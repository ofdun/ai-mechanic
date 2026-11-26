"""Benchmark of the programmatic claim verifier on the customer's manuals (hypothesis GEN-07).

Two sources of honest quotes:

* ``same_text`` — a sentence of the evidence text itself, retyped the way an LLM copies it: hyphenation
  joined, whitespace collapsed, ``ё`` → ``е``, typographic quotes and dashes replaced;
* ``other_extractor`` — a prose sentence (few numbers) from Poppler ``pdftotext`` reading order, as if the
  LLM repaired the line order of the extracted text.

Honest claims must pass. Mutations of found quotes (changed number, truncated range, dropped negation or
condition, wrong page, manual of another engine) must fail. Evidence text is the team's pdfium page export
or Poppler ``-layout`` text.

    python -m evaluation.generation.verifier_benchmark --pages DATA/pages.jsonl --pdf-dir DATA --output run.json

The manuals and their text stay local. The report contains only counts and reason codes.
"""
import argparse
import json
import random
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from ml.contracts import Evidence, EvidenceBundle, TechnicalDraft
from ml.generation.claims import GroundedClaim
from ml.generation.text import has_condition, has_negation, numbers
from ml.generation.verifier import EvidenceVerifier


SENTENCE = re.compile(r'[А-ЯЁA-Z][^.!?;]{30,260}[.!?;]')


def clean(text: str) -> str:
    text = re.sub(r'([^\W\d_])-\s*\n\s*([^\W\d_])', r'\1\2', text)
    return ' '.join(text.split())


def poppler(pdf: Path, page: int, *layout: str) -> str:
    return subprocess.run(['pdftotext', *layout, '-f', str(page), '-l', str(page), str(pdf), '-'],
                          capture_output=True, text=True, timeout=60).stdout


def sentences(text: str) -> list[str]:
    found = []
    for match in SENTENCE.finditer(clean(text)):
        sentence = match.group().strip()
        tokens = sentence.split()
        readable = sum(len(token) <= 2 and token.isalpha() for token in tokens) <= 0.2 * len(tokens)
        if len(re.findall(r'[а-яё]{3,}', sentence.lower())) >= 4 and '..' not in sentence and readable:
            found.append(sentence)
    return found


def prose(sentence: str) -> bool:
    return len(re.findall(r'[а-яё]{3,}', sentence.lower())) >= 8 and len(re.findall(r'\d+', sentence)) <= 2


def retype(sentence: str) -> str:
    """How an LLM tends to copy a sentence: typography normalised, letters and digits unchanged."""
    table = str.maketrans({'ё': 'е', 'Ё': 'Е', '«': '"', '»': '"', '–': '-', '—': '-', '\u00a0': ' '})
    return sentence.translate(table)


def bump(number: str) -> str:
    """Change the last digit: ``0,30`` → ``0,31``, ``196`` → ``197``."""
    return number[:-1] + str((int(number[-1]) + 1) % 10)


VALUE = r'(?<![\w\-./=])(\d+(?:[.,]\d+)?)(?![\w\-]|[.,]\d)'
RANGE = r'(?<![\w\-./])(\d+(?:[.,]\d+)?)\s*(?:[-–—]|\.{2,3}|…)\s*(\d+(?:[.,]\d+)?)(?![\w\-]|[.,]\d)'


def mutations(sentence: str) -> list[tuple[str, str]]:
    """(kind, mutated claim text). The quote stays the honest sentence.

    Only standalone values and ranges are mutated, not product codes or engine names such as
    ``М6-6gx30`` or ``ЯМЗ-536-10``.
    """
    out = []
    values = list(re.finditer(VALUE, sentence))
    if values:
        target = values[-1]
        out.append(('changed_number', sentence[:target.start()] + bump(target.group()) + sentence[target.end():]))
    match = re.search(RANGE, sentence)
    if match:
        out.append(('truncated_range', sentence[:match.start()] + match.group(1) + sentence[match.end():]))
    if has_negation(sentence):
        dropped = re.sub(r'\b(?:не|нельзя|запрещается|запрещено|не допускается)\s+', '', sentence, count=1,
                         flags=re.IGNORECASE)
        if dropped != sentence and not has_negation(dropped):
            out.append(('dropped_negation', dropped))
    if has_condition(sentence):
        dropped = re.sub(r'\b(?:если|при|после|только|перед|без|через)\s+[^,.;]+[,]?\s*', '', sentence, count=1,
                         flags=re.IGNORECASE)
        if dropped != sentence and not has_condition(dropped) and len(dropped) > 25:
            out.append(('dropped_condition', dropped))
    return out


def evidence(record: dict, text: str, key: str) -> Evidence:
    return Evidence(evidence_id=key, document_id=record['document_id'], manual=record['document_id'],
                    engine_models=record['models'], pdf_page=record['pdf_page'], text=text or '—')


def run(pages_path: Path, pdf_dir: Path, *, sample: int, seed: int, layout: bool) -> dict:
    records = [json.loads(line) for line in pages_path.read_text(encoding='utf-8').splitlines()]
    files = {unicodedata.normalize('NFC', path.name): path for path in pdf_dir.glob('*.pdf')}
    manifest = json.loads((pages_path.parent / 'manifest.json').read_text(encoding='utf-8'))['documents']
    pdf_of = {item['document_id']: files[unicodedata.normalize('NFC', item['file'])] for item in manifest}
    by_page = {(record['document_id'], record['pdf_page']): record for record in records}
    engines = list({tuple(record['models']): record['models'] for record in records}.values())
    rng = random.Random(seed)
    candidates = [record for record in records if len(record['text']) > 400]
    rng.shuffle(candidates)

    honest = {mode: Counter() for mode in ('same_text', 'other_extractor')}
    honest_reasons = {mode: Counter() for mode in honest}
    attacks, attack_misses, missed_attacks = Counter(), Counter(), []
    used = 0
    for record in candidates:
        if used >= sample:
            break
        pdf = pdf_of[record['document_id']]
        page_text = poppler(pdf, record['pdf_page'], '-layout') if layout else record['text']
        own_sentences = sentences(page_text)
        other_sentences = [sentence for sentence in sentences(poppler(pdf, record['pdf_page'])) if prose(sentence)]
        if not own_sentences:
            continue
        used += 1
        engine = record['models'][0]
        own = evidence(record, page_text, 'own')
        neighbour_record = by_page.get((record['document_id'], record['pdf_page'] + 1))
        neighbour = evidence(neighbour_record, neighbour_record['text'], 'next') if neighbour_record else None
        foreign_models = next(models for models in engines if engine not in models)
        foreign = Evidence(**{**own.model_dump(), 'evidence_id': 'foreign', 'engine_models': foreign_models})
        bundle = EvidenceBundle(items=[item for item in (own, neighbour, foreign) if item])
        verifier = EvidenceVerifier(engine)

        def verdict(text: str, quote: str, key: str = 'own') -> tuple[str, ...]:
            claim = GroundedClaim(kind='check', text=text, quote=quote, evidence_ids=[key])
            return verifier.inspect(TechnicalDraft(claims=[claim]), bundle).checks[0].reasons

        for mode, pool in (('same_text', own_sentences), ('other_extractor', other_sentences)):
            if not pool:
                continue
            quote = retype(rng.choice(pool))
            reasons = verdict(quote, quote)
            honest[mode]['total'] += 1
            honest[mode]['accepted'] += not reasons
            honest_reasons[mode].update(reasons)
            if reasons or mode != 'same_text':
                continue
            cases = mutations(quote)
            if neighbour and quote not in clean(neighbour_record['text']):
                cases.append(('wrong_page', quote))
            cases.append(('wrong_engine', quote))
            for kind, text in cases:
                key = {'wrong_page': 'next', 'wrong_engine': 'foreign'}.get(kind, 'own')
                if kind in {'changed_number', 'truncated_range'} and numbers(text) == numbers(quote):
                    continue
                attacks[kind] += 1
                if not verdict(text, quote, key):
                    attack_misses[kind] += 1
                    if len(missed_attacks) < 10:
                        missed_attacks.append({'kind': kind, 'document': record['document_id'],
                                               'pdf_page': record['pdf_page']})

    total_attacks = sum(attacks.values())
    return {
        'page_text': 'poppler-layout' if layout else 'pdfium-export',
        'sample_pages': used,
        'honest_acceptance': {mode: round(counter['accepted'] / counter['total'], 4) if counter['total'] else None
                              for mode, counter in honest.items()},
        'honest_cases': {mode: counter['total'] for mode, counter in honest.items()},
        'honest_rejection_reasons': {mode: dict(counter.most_common()) for mode, counter in honest_reasons.items()},
        'attack_rejection': round(1 - sum(attack_misses.values()) / total_attacks, 4) if total_attacks else None,
        'attacks': dict(attacks),
        'attack_rejection_by_kind': {kind: round(1 - attack_misses[kind] / total, 4)
                                     for kind, total in sorted(attacks.items())},
        'missed_attacks_sample': missed_attacks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pages', type=Path, required=True)
    parser.add_argument('--pdf-dir', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=400)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = [run(args.pages, args.pdf_dir, sample=args.sample, seed=args.seed, layout=layout) for layout in (False, True)]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

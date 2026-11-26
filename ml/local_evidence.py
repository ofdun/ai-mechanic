"""Read reviewed rows and complete passages from the locally installed engine manuals."""
import re

from ml.contracts import EvidenceBundle, SearchOutcome, TechnicalClaim
from ml.planner.local import is_diagnostic_followup
from ml.reviewed_checks import INSPECTIONS, RISKS, reviewed


Y236_RE = 'yamz-236_re_2010'
Y236_RK = 'yamz-236_rk_2014'
Y536_RK = 'yamz-536_rk_2013'
Z5234_RK = 'zmz-5234_rk_2025'
Z409_RK = 'zmz-40904-40905_rk_2025'
RANGE = r'(\d+(?:[.,]\d+)?\s*[-–…]\s*\d+(?:[.,]\d+)?)'

def compact(item):
    return ' '.join(item.text.split()) if item else ''


def topic_terms(text):
    words = re.findall(r'[а-я]{3,}', text.casefold().replace('ё', 'е'))
    ignored = ('как', 'нужн', 'можн', 'провер', 'опред', 'поиск', 'долж', 'руковод', 'двигат',
               'мотор', 'дальш', 'причин', 'следу', 'почему', 'сказат', 'скол', 'значен', 'номин',
               'пожал', 'найти', 'покаж', 'указа', 'этого', 'очень', 'может', 'быть', 'будет',
               'такое', 'нормат', 'назван', 'характерист', 'вместе')
    return {w[:4] for w in words if not w.startswith(ignored) and w not in {
        'при', 'что', 'для', 'чем', 'это', 'под', 'него', 'его', 'чего', 'мне', 'меня', 'или', 'тип',
    }}


def candidate_pages(retriever, plan, question):
    terms = topic_terms(re.sub(r'[яз]мз[-\s]*[\d.м]+', '', question))
    specific = terms - {'моме', 'затя', 'креп', 'болт', 'гаек', 'разм', 'норм', 'допу', 'техн'}
    if not specific:
        return []
    result = []
    for item in retriever.search(plan).items:
        if plan.engine_model not in item.engine_models:
            continue
        fragment = compact(item).casefold().replace('ё', 'е')
        windows = (fragment[i:i + 300] for i in range(0, len(fragment), 100))
        if any(all(term in window for term in specific) for window in windows):
            result.append(item)
            if len(result) == 2:
                break
    return result


class LocalEvidence:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, plan, diagnostic):
        pages = {}
        claims = []

        def page(document, number):
            item = self.retriever.page(plan.engine_model, document, number)
            if item and plan.engine_model in item.engine_models:
                pages[item.evidence_id] = item
                return item
            return None

        def cite(kind, text, *items):
            claims.append(TechnicalClaim(kind=kind, text=text, evidence_ids=[item.evidence_id for item in items]))

        question = plan.original_query.casefold().replace('ё', 'е')
        named_engines = re.findall(r'[яз]мз\s*[-‑–]?\s*\d+(?:\.\d+)?(?:м\d+)?', question)
        engine_key = lambda s: re.sub(r'[^а-я\d.]', '', s.casefold())
        other_engine = any(engine_key(name) != engine_key(plan.engine_model) for name in named_engines)
        asks_to_drive = re.search(r'перегон|перегнат|доеха|своим ходом|на базу|в рейс|можно ли ехать', question)
        if not (other_engine or asks_to_drive):
            followup = diagnostic and is_diagnostic_followup(question)
            if diagnostic and (plan.anomaly_type or followup):
                risk_only = followup and bool(re.search(r'опас|гроз|риск|плох', question))
                if diagnostic.anomaly_type == 'F08' and plan.engine_model == 'ЯМЗ-236М2':
                    injector_claims(page, cite, risk_only=risk_only)
                else:
                    groups = [('risk', RISKS)]
                    if not risk_only:
                        groups.append(('operation', INSPECTIONS))
                    for kind, records in groups:
                        for document, number, digest, instruction in records.get(diagnostic.anomaly_type, ()):
                            source = page(document, number)
                            if reviewed(source, digest):
                                cite(kind, instruction, source)
            elif all(term in question for term in ('крыш', 'коренн', 'подшипник')) and re.search(r'момент|затяж|затян', question):
                bearing_torque(plan.engine_model, page, cite)
            elif re.search(r'давлен', question) and 'форсун' in question and plan.engine_model == 'ЯМЗ-236М2':
                injector_pressure(page, cite)
            elif not re.search(r'сколько|момент|давлен|норматив|допуск|температур|зазор|размер', question):
                self.passages(plan, question, pages, claims)
            if not claims and not plan.anomaly_type and not followup:
                # Candidate pages help manual review; no unverified value from a table becomes a claim.
                candidates = candidate_pages(self.retriever, plan, question)
                return SearchOutcome(plan=plan, evidence=EvidenceBundle(items=candidates), claims=[])
        cited = {key for claim in claims for key in claim.evidence_ids}
        return SearchOutcome(plan=plan, evidence=EvidenceBundle(items=[p for k, p in pages.items() if k in cited]),
                             claims=claims)

    def passages(self, plan, source_query, pages, claims):
        # Restrict verbatim excerpts to a single complete instruction about the named component.
        terms = topic_terms(source_query)
        if not terms:
            return
        for item in self.retriever.search(plan).items:
            if plan.engine_model not in item.engine_models:
                continue
            for match in re.finditer(r'(?:^|(?<=[.!?])\s+)([А-ЯЁ][^.!?]{30,280}\.)', compact(item)):
                sentence = match[1]
                lower = sentence.casefold().replace('ё', 'е')
                if not all(t in lower for t in terms) or re.search(r'\d|[();:]|[а-я][А-Я]', sentence):
                    continue
                if re.search(r'приводит к|может привести|приводят к|вызывает|может вызвать', lower):
                    kind = 'observation'
                elif re.match(r'(?:проверить|проверьте|осмотрите|осмотреть)\b|'
                              r'перед проведением проверки .{3,80}? проверить\b', lower):
                    kind = 'check'
                else:
                    continue
                if any(c.text == sentence or c.kind == kind for c in claims):
                    continue
                pages[item.evidence_id] = item
                claims.append(TechnicalClaim(kind=kind, text=sentence, evidence_ids=[item.evidence_id]))


def injector_claims(page, cite, *, risk_only):
    from ml.pdf_baseline import ReviewedOperations

    diagnosis = page(Y236_RE, 52)
    if not diagnosis:
        return
    operations = ReviewedOperations.operations(EvidenceBundle(items=[diagnosis]))
    if not operations:
        return
    if re.search(r'мощности,\s*дымит', compact(diagnosis), re.I):
        cite('risk', 'При нарушенной регулировке или засорении форсунки двигатель может терять мощность и дымить.', diagnosis)
    if risk_only:
        return
    injector_pressure(page, cite)
    for operation in operations:
        cite('operation', operation.text, diagnosis)


def injector_pressure(page, cite):
    service = page(Y236_RE, 37)
    if not reviewed(service, '36e0d86928aaa7f4a46eda9d'):
        return
    pressure = re.search(r'давления начала впрыскивания форсунок до (\d+[,.]\d+) МПа', compact(service))
    setting = re.search(r'давление начала впрыскивания (\d+[,.]\d+\+\d+[,.]\d+) МПа', compact(service))
    if pressure and setting:
        cite('check', 'Проверьте давление начала впрыскивания форсунок.', service)
        cite('reference_value', f'После длительной работы допускается снижение давления до {pressure[1]} МПа. '
             f'При обслуживании или замене хотя бы одной форсунки все регулируют на {setting[1]} МПа.', service)


def bearing_torque(engine, page, cite):
    if engine.startswith('ЗМЗ-'):
        document, number = (Z5234_RK, 137) if engine == 'ЗМЗ-5234.10' else (Z409_RK, 230)
        table = page(document, number)
        text = compact(table)
        row = re.search(r'(Болты|Гайки) крепления крышек коренных подшипников ' + RANGE + r'\s*\(', text)
        if not row or 'Н·м (кгс·м)' not in text:
            return
        cite('reference_value', f'{row[1]} крепления крышек коренных подшипников по сводной таблице — {row[2]} Н·м.', table)
        if document == Z5234_RK:
            assembly = page(document, 117)
            remaining = page(document, 118)
            value = re.search(r'затянуть гайки крепления крышки первого коренного подшипника '
                              r'динамометрическим ключом моментом ' + RANGE + r' Н[·\uf0d7]м',
                              compact(assembly))
            rest = re.search(r'Окончательная затяжка этих гаек [^.]{0,65}моментом затяжки ' + RANGE, compact(remaining))
            if (value and rest and value[1] == rest[1] and value[1] != row[2]
                    and 'остальные крышки коренных подшипников' in compact(assembly)):
                cite('observation', f'В порядке сборки первой и остальных крышек указано {value[1]} Н·м. '
                     'Верхняя граница отличается от сводной таблицы. Уточните применимость перед затяжкой.',
                     assembly, remaining, table)
        return
    if engine == 'ЯМЗ-536':
        order, values = page(Y536_RK, 319), page(Y536_RK, 320)
        units = re.search(r'предварительный момент\s*-\s*Мкр=(\d+±\d+) Н\s*м;\s*-\s*довернуть болты '
                          r'на угол\s*-\s*(\d+±\d+)\s*[о°]', compact(values), re.I)
        if (units and 'обеспечив затяжку в два приема' in compact(order)
                and re.search(r'Смазать резьбу и торцы головок болтов[^!]{0,65}моторным маслом!', compact(order))):
            cite('reference_value', f'Смажьте резьбу и торцы головок болтов моторным маслом. '
                 f'Болты крышек затягивают в два приёма по схеме руководства. '
                 f'Предварительный момент — {units[1]} Н·м, затем доворот на {units[2]}°.', order, values)
        return
    if engine == 'ЯМЗ-236М2':
        operation, repair = page(Y236_RE, 68), page(Y236_RK, 41)
        re_text, rk_text = compact(operation), compact(repair)
        vertical = re.search(r'вертикальные\s+' + RANGE, re_text)
        horizontal = re.search(r'горизонтальные\s+' + RANGE, re_text)
        bolts = re.search(r'болтов крепления крышек производить с моментом ' + RANGE + r' Н\*м', rk_text)
        ties = re.search(r'стяжных болтов с моментом ' + RANGE + r' Н\*м', rk_text)
        if vertical and horizontal and bolts and ties and 'Н·м (кгс·м)' in re_text and 'в два приема' in rk_text:
            cite('reference_value', f'По руководству эксплуатации вертикальные болты — {vertical[1]} Н·м, '
                 f'горизонтальные — {horizontal[1]} Н·м.', operation)
            cite('reference_value', f'При сборке по руководству ремонта болты крышек затягивают в два приёма '
                 f'моментом {bolts[1]} Н·м, стяжные болты блока — {ties[1]} Н·м.', repair)
            cite('observation', 'Значения в руководствах расходятся. Для конкретной сборки уточните '
                 'применимое руководство у специалиста или производителя.', operation, repair)

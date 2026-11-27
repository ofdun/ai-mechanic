"""Suggest fleet vehicles from a photo of a Russian registration plate.

Recognition runs on the local Tesseract binary. The image stays in memory and is
never written, logged or sent to an outside service. The result is only a
suggestion: the mechanic confirms it or picks the vehicle from the list.
"""
import io
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable

from ml.contracts import AgentError

# Plate letters are the twelve Cyrillic letters that look like Latin ones.
LETTERS = 'ABEKMHOPCTYX'
CYRILLIC = str.maketrans('АВЕКМНОРСТУХ', LETTERS)
DIGITS = '0123456789'
# Common OCR confusions, applied only where the plate format expects the other kind of symbol.
AS_DIGIT = str.maketrans({'O': '0', 'D': '0', 'Q': '0', 'B': '8', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'T': '7'})
AS_LETTER = str.maketrans({'0': 'O', '8': 'B', '4': 'A', '3': 'E', '7': 'T'})
# Letter, three digits, two letters, two or three digits of the region.
SHAPE = 'LDDDLLDD'
PLATE = re.compile(f'[{LETTERS}][0-9]{{3}}[{LETTERS}]{{2}}[0-9]{{2,3}}')
MAX_PIXELS = 40_000_000
MAX_SIDE = 1600
MAX_CANDIDATES = 5


def normalize(text: str) -> str:
    return re.sub(r'[^0-9A-Z]', '', text.upper().translate(CYRILLIC))


def _coerce(window: str) -> str | None:
    """Fix symbol-class confusions in an 8 or 9 character window, or reject it."""
    shape = SHAPE + 'D' * (len(window) - len(SHAPE))
    fixed = ''.join(ch.translate(AS_LETTER) if kind == 'L' else ch.translate(AS_DIGIT) for ch, kind in zip(window, shape))
    return fixed if PLATE.fullmatch(fixed) else None


def extract_plates(text: str) -> list[str]:
    """Plate-shaped readings found in OCR output, with and without a third region digit."""
    found: list[str] = []
    for line in text.splitlines():
        stream = normalize(line)
        for size in (9, 8):
            for start in range(len(stream) - size + 1):
                plate = _coerce(stream[start:start + size])
                if plate and plate not in found:
                    found.append(plate)
    return found


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
    return row[-1]


def match_fleet(readings: Iterable[str], vehicle_ids: Iterable[str]) -> dict:
    """Rank fleet vehicles against plate readings.

    ``reliable`` is true only when a reading equals exactly one vehicle and no other
    vehicle is within one symbol of any reading. Even then the caller must confirm.
    """
    readings = list(dict.fromkeys(readings))
    fleet = {normalize(vehicle_id): vehicle_id for vehicle_id in vehicle_ids}
    best: dict[str, tuple[int, str]] = {}
    for reading in readings:
        for plate, vehicle_id in fleet.items():
            distance = _distance(reading, plate)
            if distance <= 2 and (vehicle_id not in best or distance < best[vehicle_id][0]):
                best[vehicle_id] = (distance, reading)
    ranked = sorted(best.items(), key=lambda item: (item[1][0], item[0]))[:MAX_CANDIDATES]
    candidates = [{'vehicle_id': vehicle_id, 'reading': reading, 'distance': distance}
                  for vehicle_id, (distance, reading) in ranked]
    exact = [item for item in candidates if item['distance'] == 0]
    close = [item for item in candidates if item['distance'] <= 1]
    return {
        'readings': readings,
        'candidates': candidates,
        'reliable': len(exact) == 1 and len(close) == 1,
    }


def prepare_image(data: bytes) -> list[bytes]:
    """Decode the photo and return PNG variants for OCR, without EXIF or location data."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert('L')
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise AgentError('PLATE_IMAGE_INVALID', 'Не удалось открыть фото. Сделайте снимок номера ещё раз.') from error
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    contrast = ImageOps.autocontrast(image, cutoff=2)
    binary = contrast.point(lambda value: 255 if value > 128 else 0)
    variants = []
    for variant in (contrast, binary):
        buffer = io.BytesIO()
        variant.save(buffer, format='PNG')
        variants.append(buffer.getvalue())
    return variants


def tesseract(image: bytes, psm: int) -> str:
    binary = shutil.which('tesseract')
    if binary is None:
        raise AgentError('PLATE_OCR_UNAVAILABLE', 'Распознавание номера на этом сервере не настроено. Выберите машину из списка.')
    args = [binary, 'stdin', 'stdout', '-l', 'eng', '--psm', str(psm),
            '-c', f'tessedit_char_whitelist={LETTERS}{DIGITS}']
    try:
        return subprocess.run(args, input=image, capture_output=True, timeout=10, check=True).stdout.decode('utf-8', 'replace')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise AgentError('PLATE_OCR_FAILED', 'Номер не распознан. Выберите машину из списка.', retryable=True) from error


def read_plate(data: bytes, vehicle_ids: Iterable[str], ocr: Callable[[bytes, int], str] = tesseract) -> dict:
    readings: list[str] = []
    for image in prepare_image(data):
        # 7 reads a tight crop of the plate as one line, 11 finds it on a wider shot.
        for psm in (7, 11):
            readings.extend(extract_plates(ocr(image, psm)))
    return match_fleet(readings, vehicle_ids)

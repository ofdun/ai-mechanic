import io

import pytest

from ml.contracts import AgentError
from ml.plates import extract_plates, match_fleet, normalize, read_plate

FLEET = ['P456AT178', 'M333YK47', 'B418HK98', 'P456AT78']


def test_normalize_maps_cyrillic_lookalikes():
    assert normalize('р 456 ат 178 RUS') == 'P456AT178RUS'


def test_extract_fixes_symbol_class_confusions():
    # O read instead of 0 in the digits and 8 instead of B in the letters.
    assert 'B418HK98' in extract_plates('8418HK98')
    assert 'P456AT178' in extract_plates('P456AT17B')
    assert 'E549YT198' in extract_plates('E 5 4 9 Y T | 198 RUS')
    assert extract_plates('ДВИГАТЕЛЬ 12345') == []


def test_exact_unique_reading_is_reliable_but_only_suggested():
    result = match_fleet(['M333YK47'], FLEET)
    assert result['reliable'] is True
    assert result['candidates'][0] == {'vehicle_id': 'M333YK47', 'reading': 'M333YK47', 'distance': 0}


def test_near_miss_is_not_reliable():
    result = match_fleet(['M333YK41'], FLEET)
    assert result['reliable'] is False
    assert [item['vehicle_id'] for item in result['candidates']] == ['M333YK47']


def test_lookalike_neighbour_makes_exact_reading_unreliable():
    # P456AT78 and P456AT178 differ by one symbol, so neither may be preselected.
    result = match_fleet(['P456AT178'], FLEET)
    assert result['reliable'] is False
    assert [item['vehicle_id'] for item in result['candidates']][:2] == ['P456AT178', 'P456AT78']


def test_conflicting_readings_are_not_reliable():
    assert match_fleet(['M333YK47', 'B418HK98'], FLEET)['reliable'] is False


def test_unknown_plate_has_no_candidates():
    result = match_fleet(['A777AA77'], FLEET)
    assert result == {'readings': ['A777AA77'], 'candidates': [], 'reliable': False}


def png():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (40, 20), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


def test_read_plate_runs_ocr_on_prepared_variants():
    calls = []

    def ocr(image, psm):
        calls.append(psm)
        assert image.startswith(b'\x89PNG')
        return 'B418HK 98\n'

    result = read_plate(png(), FLEET, ocr=ocr)
    assert calls == [7, 11, 7, 11]
    assert result['reliable'] is True
    assert result['candidates'][0]['vehicle_id'] == 'B418HK98'


def test_read_plate_rejects_non_images():
    with pytest.raises(AgentError) as error:
        read_plate(b'not an image', FLEET, ocr=lambda image, psm: '')
    assert error.value.code == 'PLATE_IMAGE_INVALID'

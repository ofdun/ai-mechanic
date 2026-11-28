import json
import re
import struct
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / 'app' / 'frontend'
PUBLIC = FRONTEND / 'public'


def png_size(path):
    header = path.read_bytes()[:24]
    assert header[:8] == b'\x89PNG\r\n\x1a\n', path
    return '%dx%d' % struct.unpack('>II', header[16:24])


def test_manifest_is_installable_from_root():
    manifest = json.loads((PUBLIC / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['start_url'] == manifest['scope'] == '/'
    assert manifest['display'] == 'standalone'
    assert manifest['name'] and manifest['short_name']
    purposes = set()
    for icon in manifest['icons']:
        assert icon['type'] == 'image/png'
        assert png_size(PUBLIC / icon['src'].lstrip('/')) == icon['sizes']
        purposes.add((icon['sizes'], icon['purpose']))
    assert {('192x192', 'any'), ('512x512', 'any'), ('512x512', 'maskable')} <= purposes


def test_index_links_manifest_and_icons_with_credentials():
    html = (FRONTEND / 'index.html').read_text(encoding='utf-8')
    manifest = re.search(r'<link rel="manifest"[^>]*>', html)
    assert manifest and 'href="/manifest.json"' in manifest[0]
    assert 'crossorigin="use-credentials"' in manifest[0]
    touch = re.search(r'<link rel="apple-touch-icon" href="/([^"]+)"', html)
    assert touch and png_size(PUBLIC / touch[1]) == '180x180'


def test_no_service_worker_caches_diagnoses_or_api():
    sources = [path.read_text(encoding='utf-8') for path in (FRONTEND / 'src').rglob('*.js*')]
    sources.append((FRONTEND / 'index.html').read_text(encoding='utf-8'))
    assert not any('serviceWorker' in text or 'caches.' in text for text in sources)
    assert not list(PUBLIC.rglob('*sw*.js'))

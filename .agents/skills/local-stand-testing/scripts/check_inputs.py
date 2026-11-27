"""Check the supplied local release kit before building Docker images. Read-only, stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(fleet, mocks):
    documents = json.loads((ROOT / "ml/ingestion/manifest.json").read_text())["documents"]
    files = list(fleet.iterdir())
    for doc in documents:
        matches = [p for p in files if unicodedata.normalize("NFC", p.name) == doc["file"]]
        if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
            raise ValueError(f'Нужен оригинальный PDF в FLEET_DIR: {doc["file"]}')
        if digest(matches[0]) != doc["sha256"]:
            raise ValueError(f'Редакция PDF отличается от манифеста: {doc["file"]}')
    archives = list(fleet.glob("neurodvs_hackathon_mock_*.zip"))
    if len(archives) != 1:
        raise ValueError("Для полного комплекта укажите FLEET_DIR с одним исходным ZIP моков.")
    with ZipFile(archives[0]) as archive:
        names = [n for n in archive.namelist() if n == "data/vehicles.json" or n.endswith("/data/vehicles.json")]
        if len(names) != 1:
            raise ValueError("В архиве ожидается один data/vehicles.json.")
        prefix = names[0][:-len("data/vehicles.json")]
        for rel in ("mock_ml/app.py", "mock_toir/app.py", "data/vehicles.json", "data/fault_types.json"):
            if (mocks / rel).read_bytes() != archive.read(prefix + rel):
                raise ValueError(f"Файл MOCK_DIR/{rel} отличается от исходного ZIP.")
        catalog = json.loads(archive.read(names[0]))
    override = fleet / "vehicles.json"
    if override.is_file() and json.loads(override.read_bytes()) != catalog:
        raise ValueError("vehicles.json в FLEET_DIR перекрывает каталог ZIP и отличается от него.")
    vehicles = catalog["vehicles"]
    print(f'PDF: {len(documents)}, ТС: {len(vehicles)}, замеры: {sum(len(v["sessions"]) for v in vehicles)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-dir", type=Path, required=True)
    parser.add_argument("--mock-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        check(args.fleet_dir, args.mock_dir)
    except (OSError, KeyError, ValueError, BadZipFile) as error:
        parser.exit(1, f"Проверка комплекта: {error}\n")

import json
from pathlib import Path
from typing import Dict, List, Union

from app.db import close_db, fetch_products, init_db


def _serialize(value) -> str:
    """Convert any DB value (including Decimal) to a plain string."""
    if isinstance(value, str):
        return value
    return str(value)


async def _collect_from_db() -> List[Dict]:
    try:
        await init_db()
        products = await fetch_products()
        await close_db()
    except Exception as exc:
        print(f"  Warning: DB unavailable ({exc}), skipping DB records.")
        return []

    return [
        {"source": "db_product", **{k: _serialize(v) for k, v in p.items()}}
        for p in products
    ]


def _collect_from_files(data_dir: Union[str, Path]) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    records: List[Dict] = []

    for path in sorted(data_path.iterdir()):
        if not path.is_file():
            continue

        if path.suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            for paragraph in text.split("\n\n"):
                para = paragraph.strip()
                if para:
                    records.append({"source": "txt_file", "filename": path.name, "text": para})

        elif path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping {path.name} — invalid JSON ({exc}).")
                continue

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        records.append({"source": "json_file", "filename": path.name, **item})
                    elif isinstance(item, str) and item.strip():
                        records.append({"source": "json_file", "filename": path.name, "text": item})
            elif isinstance(data, dict):
                records.append({"source": "json_file", "filename": path.name, **data})

    return records


async def collect_all(data_dir: Union[str, Path] = "data") -> List[Dict]:
    db_records = await _collect_from_db()
    file_records = _collect_from_files(data_dir)
    return db_records + file_records

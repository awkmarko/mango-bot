import json
from pathlib import Path
from typing import Dict, List, Optional, Union

_METADATA_KEYS = {"source", "filename"}


def _product_pair(record: dict) -> Optional[Dict]:
    name = record.get("name", "").strip()
    short_name = record.get("short_name", "").strip()
    price = record.get("price", "").strip()
    sale_price = record.get("sale_price", "").strip()
    in_stock = record.get("in_stock", "").strip()
    sku = record.get("sku", "").strip()

    if not name:
        return None

    prompt = f"Tell me about {name}."

    parts = [name]
    if short_name and short_name != name:
        parts.append(f"(also known as {short_name})")
    if sku:
        parts.append(f"SKU: {sku}.")

    if sale_price and sale_price not in ("0", "0.00"):
        parts.append(f"Regular price: {price}, on sale for {sale_price}.")
    elif price:
        parts.append(f"Price: {price}.")

    if in_stock and in_stock != "0":
        parts.append(f"In stock: {in_stock} units.")
    else:
        parts.append("Currently out of stock.")

    completion = " ".join(parts)
    return {"prompt": prompt, "completion": completion}


def _text_pair(record: dict) -> Optional[Dict]:
    text = record.get("text", "").strip()
    if not text:
        return None

    excerpt = text[:80].rstrip()
    if len(text) > 80:
        excerpt += "..."
    prompt = f"Summarize: {excerpt}"

    return {"prompt": prompt, "completion": text}


def _json_pair(record: dict) -> Optional[Dict]:
    # Pass through if already shaped correctly
    if "prompt" in record and "completion" in record:
        p = record["prompt"].strip()
        c = record["completion"].strip()
        if p and c:
            return {"prompt": p, "completion": c}
        return None

    # Delegate to text pair if a text field exists
    if "text" in record:
        return _text_pair(record)

    # Build from remaining fields
    filename = record.get("filename", "data")
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    content_fields = {
        k: v for k, v in record.items()
        if k not in _METADATA_KEYS and isinstance(v, str) and v.strip()
    }
    if not content_fields:
        return None

    completion = "; ".join(f"{k}: {v}" for k, v in content_fields.items())
    prompt = f"What information is available about {title}?"
    return {"prompt": prompt, "completion": completion}


_DISPATCH = {
    "db_product": _product_pair,
    "txt_file": _text_pair,
    "json_file": _json_pair,
}


def to_training_pairs(records: List[Dict]) -> List[Dict]:
    pairs: List[Dict] = []
    for record in records:
        handler = _DISPATCH.get(record.get("source", ""))
        if handler is None:
            continue
        try:
            pair = handler(record)
        except Exception:
            continue
        if pair and pair["prompt"].strip() and pair["completion"].strip():
            pairs.append(pair)
    return pairs


def export_jsonl(pairs: List[Dict], output_path: Union[str, Path]) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)

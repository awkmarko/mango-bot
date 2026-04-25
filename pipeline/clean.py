import hashlib
import re
from typing import Dict, List, Set

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_METADATA_KEYS = {"source", "filename"}


def _strip_html(text: str) -> str:
    return _HTML_TAG.sub("", text)


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _clean_value(value: str) -> str:
    return _normalize(_strip_html(value))


def _content_fingerprint(record: dict) -> str:
    content = {k: v for k, v in record.items() if k not in _METADATA_KEYS}
    return hashlib.md5(str(sorted(content.items())).encode()).hexdigest()


def _has_content(record: dict) -> bool:
    return any(
        isinstance(v, str) and v.strip()
        for k, v in record.items()
        if k not in _METADATA_KEYS
    )


def clean_records(records: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    cleaned: List[Dict] = []

    for record in records:
        cleaned_record = {
            k: _clean_value(v) if isinstance(v, str) else v
            for k, v in record.items()
        }

        if not _has_content(cleaned_record):
            continue

        fingerprint = _content_fingerprint(cleaned_record)
        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        cleaned.append(cleaned_record)

    return cleaned

import asyncio
import logging
import sys
from typing import Callable, Dict, List, Optional

# pysqlite3-binary ships a newer sqlite3 (>= 3.35.0) required by ChromaDB.
# This swap must happen before chromadb is imported.
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.config import Settings as ChromaSettings
import httpx

from app.config import settings
from app.db import fetch_all_orders, fetch_products

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_products_collection = None
_orders_collection = None
_indexed = False


def is_indexed() -> bool:
    return _indexed


def _get_collections() -> tuple:
    global _chroma_client, _products_collection, _orders_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _products_collection = _chroma_client.get_or_create_collection("products")
        _orders_collection = _chroma_client.get_or_create_collection("orders")
    return _products_collection, _orders_collection


def sanitize_metadata(meta: dict) -> dict:
    result = {}
    for k, v in meta.items():
        if v is None:
            result[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            result[k] = v
        else:
            result[k] = str(v)
    return result


# --- Text and metadata generators (module-level so _find_changed can call them) ---

def _product_text(p: dict) -> str:
    return (
        f"Product: {p.get('name') or p.get('short_name', '')}. "
        f"Price: {p.get('final_price', 0)} RSD."
    )


def _product_metadata(p: dict) -> dict:
    name = p.get("name") or p.get("short_name", "")
    return sanitize_metadata({"id": p["id"], "name": name, "price": str(p.get("final_price", 0))})


def _order_text(o: dict) -> str:
    customer_name = f"{o.get('first_name', '')} {o.get('last_name', '')}".strip()
    return (
        f"Order {o.get('id')} for {customer_name} ({o.get('email', '')}). "
        f"Status: {o.get('status', '')}. Total: {o.get('total', '')} RSD. "
        f"Date: {o.get('date_created', '')}."
    )


def _order_metadata(o: dict) -> dict:
    customer_name = f"{o.get('first_name', '')} {o.get('last_name', '')}".strip()
    return sanitize_metadata({
        "id": o.get("id"),
        "status": o.get("status", ""),
        "customer_name": customer_name,
        "customer_email": o.get("email", ""),
    })


# --- Embedding ---

async def _embed(text: str) -> Optional[List[float]]:
    """Embed text via Ollama; retries once on failure, then skips."""
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0)
            ) as client:
                response = await client.post(
                    f"{settings.ollama_base_url}/api/embeddings",
                    json={"model": settings.embedding_model, "prompt": text},
                )
                response.raise_for_status()
                return response.json()["embedding"]
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "[rag] embedding attempt 1 failed (%s: %s), retrying in 1s...",
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(1.0)
            else:
                logger.error(
                    "[rag] embedding failed after 2 attempts (%s: %s) for text: %r",
                    type(exc).__name__, exc, text[:80],
                )
    return None


async def _embed_batch(texts: List[str]) -> List:
    tasks = [_embed(text) for text in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)


# --- Indexing helpers ---

def _find_changed(
    collection,
    items: List[dict],
    text_fn: Callable[[dict], str],
) -> List[dict]:
    """Return items that are new or whose indexed document text has changed."""
    if not items:
        return []
    ids = [str(item["id"]) for item in items]
    existing = collection.get(ids=ids, include=["documents"])
    existing_docs: Dict[str, str] = dict(zip(existing["ids"], existing["documents"]))
    return [
        item for item in items
        if str(item["id"]) not in existing_docs
        or existing_docs[str(item["id"])] != text_fn(item)
    ]


async def _index_items(
    collection,
    items: List[dict],
    text_fn: Callable[[dict], str],
    metadata_fn: Callable[[dict], dict],
    label: str,
) -> int:
    """Embed and upsert items in batches. Logs item ID on per-item failures."""
    batch_size = 10
    count = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [text_fn(item) for item in batch]
        embeddings = await _embed_batch(texts)
        for item, text, emb in zip(batch, texts, embeddings):
            if isinstance(emb, Exception):
                logger.error(
                    "[rag] embedding raised for %s id=%s: %s: %s",
                    label, item.get("id"), type(emb).__name__, emb,
                )
                continue
            if emb is None:
                logger.warning(
                    "[rag] skipping %s id=%s: embedding returned None",
                    label, item.get("id"),
                )
                continue
            collection.upsert(
                ids=[str(item["id"])],
                embeddings=[emb],
                documents=[text],
                metadatas=[metadata_fn(item)],
            )
            count += 1
        print(
            f"[rag] {label}: {min(i + batch_size, len(items))}/{len(items)}",
            flush=True,
        )
        await asyncio.sleep(0.5)
    return count


# --- Public indexing API ---

async def maybe_index() -> None:
    """Incremental startup indexing: skip items that are already indexed with current text."""
    global _indexed
    products_col, orders_col = _get_collections()

    products = await fetch_products()
    changed_products = _find_changed(products_col, products, _product_text)
    if changed_products:
        print(
            f"[rag] indexing {len(changed_products)} new/changed products "
            f"(of {len(products)} total)"
        )
        await _index_items(
            products_col, changed_products, _product_text, _product_metadata, "products"
        )
    else:
        print(f"[rag] products up to date ({products_col.count()} indexed), skipping")

    orders = await fetch_all_orders()
    changed_orders = _find_changed(orders_col, orders, _order_text)
    if changed_orders:
        print(
            f"[rag] indexing {len(changed_orders)} new/changed orders "
            f"(of {len(orders)} total)"
        )
        try:
            await _index_items(
                orders_col, changed_orders, _order_text, _order_metadata, "orders"
            )
        except Exception as exc:
            logger.error("[rag] order indexing failed: %s", exc)
    else:
        print(f"[rag] orders up to date ({orders_col.count()} indexed), skipping")

    _indexed = True
    print("[rag] index ready")


async def reindex_all() -> None:
    """Full reindex: re-embed every item regardless of current index state."""
    global _indexed
    _indexed = False
    print("[rag] starting full reindex...")
    products_col, orders_col = _get_collections()

    products = await fetch_products()
    await _index_items(products_col, products, _product_text, _product_metadata, "products")

    orders = await fetch_all_orders()
    try:
        await _index_items(orders_col, orders, _order_text, _order_metadata, "orders")
    except Exception as exc:
        logger.error("[rag] order indexing failed: %s", exc)

    _indexed = True
    print("[rag] reindex complete")


# --- Search ---

async def search_products_rag(query: str, n_results: int = 5) -> List[Dict]:
    products_col, _ = _get_collections()
    embedding = await _embed(query)
    if embedding is None:
        return []
    try:
        results = products_col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas"],
        )
        return [dict(m) for m in results.get("metadatas", [[]])[0]]
    except Exception as exc:
        logger.error("[rag] product search failed: %s", exc)
        return []


async def search_orders_rag(query: str, n_results: int = 5) -> List[Dict]:
    _, orders_col = _get_collections()
    embedding = await _embed(query)
    if embedding is None:
        return []
    try:
        results = orders_col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas"],
        )
        return [dict(m) for m in results.get("metadatas", [[]])[0]]
    except Exception as exc:
        logger.error("[rag] order search failed: %s", exc)
        return []

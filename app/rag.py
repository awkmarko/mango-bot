import asyncio
import logging
import sys
from typing import Dict, List, Optional

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


async def _embed_batch(texts: List[str]) -> List:
    tasks = [_embed(text) for text in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def _embed(text: str) -> Optional[List[float]]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]
    except Exception as exc:
        logger.error("[rag] embedding failed: %s", exc)
        return None


async def index_products() -> None:
    products_col, _ = _get_collections()
    products = await fetch_products()
    count = 0
    batch_size = 10
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]
        texts = [
            f"Product: {p.get('name') or p.get('short_name', '')}. Price: {p.get('final_price', 0)} RSD."
            for p in batch
        ]
        embeddings = await _embed_batch(texts)
        for p, text, emb in zip(batch, texts, embeddings):
            if emb is None or isinstance(emb, Exception):
                continue
            name = p.get("name") or p.get("short_name", "")
            products_col.upsert(
                ids=[str(p["id"])],
                embeddings=[emb],
                documents=[text],
                metadatas=[sanitize_metadata({"id": p["id"], "name": name, "price": str(p.get("final_price", 0))})],
            )
            count += 1
        print(f"[rag] indexed {min(i + batch_size, len(products))}/{len(products)} products", flush=True)
        await asyncio.sleep(0.5)


async def index_orders() -> None:
    _, orders_col = _get_collections()
    orders = await fetch_all_orders()
    count = 0
    batch_size = 10

    def _order_text(o: dict) -> str:
        oid = o.get("id")
        customer_name = f"{o.get('first_name', '')} {o.get('last_name', '')}".strip()
        customer_email = o.get("email", "")
        status = o.get("status", "")
        total = o.get("total", "")
        date_created = str(o.get("date_created", ""))
        return (
            f"Order {oid} for {customer_name} ({customer_email}). "
            f"Status: {status}. Total: {total} RSD. Date: {date_created}."
        )

    for i in range(0, len(orders), batch_size):
        batch = orders[i:i + batch_size]
        texts = [_order_text(o) for o in batch]
        embeddings = await _embed_batch(texts)
        for o, text, emb in zip(batch, texts, embeddings):
            if emb is None or isinstance(emb, Exception):
                continue
            oid = o.get("id")
            customer_name = f"{o.get('first_name', '')} {o.get('last_name', '')}".strip()
            orders_col.upsert(
                ids=[str(oid)],
                embeddings=[emb],
                documents=[text],
                metadatas=[sanitize_metadata({
                    "id": oid,
                    "status": o.get("status", ""),
                    "customer_name": customer_name,
                    "customer_email": o.get("email", ""),
                })],
            )
            count += 1
        print(f"[rag] indexed {min(i + batch_size, len(orders))}/{len(orders)} orders", flush=True)
        await asyncio.sleep(0.5)


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


async def reindex_all() -> None:
    global _indexed
    print("[rag] starting full reindex...")
    await index_products()
    try:
        await index_orders()
    except Exception as exc:
        logger.error("[rag] index_orders failed: %s", exc)
    _indexed = True
    print("[rag] reindex complete")

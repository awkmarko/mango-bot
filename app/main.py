import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.db import (
    add_order_product,
    close_db,
    fetch_order,
    fetch_order_products,
    get_product_by_id,
    init_db,
    remove_order_product,
    update_order_address,
)
from app.rag import is_indexed, reindex_all, search_orders_rag, search_products_rag
from app.session_store import add_turn, clear_session, get_history
from app.system_prompt import build_system_prompt, format_order_context

_LOCKED_STATUSES = {"shipped", "delivered"}

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{[^`]*\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r'(\{[^{}]*"action"[^{}]*\})', re.DOTALL)


def _extract_action(text: str) -> Optional[dict]:
    for pattern in (_FENCED_JSON, _BARE_JSON):
        m = pattern.search(text)
        if m:
            try:
                data = json.loads(m.group(1))
                if "action" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None


def _strip_action_block(text: str) -> str:
    text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r'\{[^{}]*"action"[^{}]*\}', "", text)
    return text.strip()


async def _execute_action(action: dict) -> str:
    name = action.get("action")

    if name == "update_address":
        order_id = int(action["order_id"])
        order = await fetch_order(order_id)
        if order is None:
            return f"Greška: narudžbina {order_id} nije pronađena."
        if order.get("status") in _LOCKED_STATUSES:
            return f"Greška: narudžbina {order_id} ima status '{order['status']}' i ne može se menjati."
        await update_order_address(
            order_id,
            action.get("first_name", ""),
            action.get("last_name", ""),
            action.get("address", ""),
            action.get("city", ""),
            action.get("postcode", ""),
            action.get("phone", ""),
        )
        return f"Adresa za narudžbinu {order_id} je uspešno ažurirana."

    if name == "add_product":
        order_id = int(action["order_id"])
        order = await fetch_order(order_id)
        if order is None:
            return f"Greška: narudžbina {order_id} nije pronađena."
        if order.get("status") in _LOCKED_STATUSES:
            return f"Greška: narudžbina {order_id} ima status '{order['status']}' i ne može se menjati."
        product_id = int(action["product_id"])
        product = await get_product_by_id(product_id)
        if product is None:
            return "Proizvod sa tim ID-jem ne postoji u sistemu."
        await add_order_product(
            order_id,
            product_id,
            int(action.get("quantity", 1)),
            float(action.get("total", 0)),
        )
        return f"Proizvod je dodat u narudžbinu {order_id}."

    if name == "remove_product":
        op_id = int(action["order_product_id"])
        await remove_order_product(op_id)
        return f"Stavka (order_product_id={op_id}) je uklonjena iz narudžbine."

    return f"Greška: nepoznata akcija '{name}'."


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    asyncio.create_task(reindex_all())
    print("[rag] indexing started in background", flush=True)
    yield
    await close_db()


app = FastAPI(title="Mango Bot", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    indexed = is_indexed()

    system_prompt = build_system_prompt()
    history = get_history(body.session_id)

    rag_products = []
    if indexed:
        rag_products = await search_products_rag(body.message)
        print(f"[rag] product query={body.message!r} → {len(rag_products)} results")

    order_ids = re.findall(r"\d{5,}", body.message)
    order_context = None
    rag_orders = []
    if order_ids:
        order_id = int(order_ids[0])
        order = await fetch_order(order_id)
        if order:
            order_products = await fetch_order_products(order_id)
            order_context = format_order_context(order, order_products)
            print(f"[order] fetched order {order_id}: status={order.get('status')}, {len(order_products)} products")
        else:
            print(f"[order] order {order_id} not found")
    elif indexed:
        rag_orders = await search_orders_rag(body.message)
        print(f"[rag] order query={body.message!r} → {len(rag_orders)} results")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if not indexed:
        messages.append({
            "role": "system",
            "content": "SYSTEM CONTEXT: Product search index is still loading. Only answer from order data if available. Do not mention products.",
        })
    if rag_products:
        messages.append({
            "role": "system",
            "content": f"SYSTEM CONTEXT - Relevant products: {json.dumps(rag_products, ensure_ascii=False)}",
        })
    if order_context:
        messages.append({"role": "system", "content": order_context})
    if rag_orders:
        messages.append({
            "role": "system",
            "content": f"SYSTEM CONTEXT - Relevant orders: {json.dumps(rag_orders, ensure_ascii=False)}",
        })
    messages.append({"role": "user", "content": body.message})

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={"model": settings.ollama_model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        data = response.json()

    raw_reply = data["message"]["content"]
    print(f"[llm] raw reply: {raw_reply!r}")

    action = _extract_action(raw_reply)
    if action:
        print(f"[action] executing: {action}")
        result = await _execute_action(action)
        print(f"[action] result: {result}")

        confirm_messages = messages + [
            {"role": "assistant", "content": raw_reply},
            {"role": "system", "content": f"Rezultat akcije: {result}"},
            {"role": "user", "content": "Potvrdi korisniku na srpskom šta je urađeno."},
        ]
        async with httpx.AsyncClient(timeout=60.0) as client:
            confirm_response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={"model": settings.ollama_model, "messages": confirm_messages, "stream": False},
            )
            confirm_response.raise_for_status()
            confirm_data = confirm_response.json()

        final_reply = _strip_action_block(confirm_data["message"]["content"])
    else:
        final_reply = _strip_action_block(raw_reply)

    add_turn(body.session_id, body.message, final_reply)
    return ChatResponse(reply=final_reply)


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    clear_session(session_id)
    return JSONResponse({"status": "cleared"})

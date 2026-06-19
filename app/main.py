import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
from app.db import (
    add_order_product,
    close_db,
    fetch_order,
    fetch_order_products,
    get_products_by_ids,
    init_db,
    remove_order_product,
    update_order_address,
)
from app.rag import is_indexed, maybe_index, reindex_all, search_products_rag
from app.session_store import add_turn, clear_session, get_history
from app.system_prompt import build_system_prompt, format_order_card, format_product_card
import app.order_state as order_state

_LOCKED_STATUSES = {"shipped", "delivered"}

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{[^`]*\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r'(\{[^{}]*"action"[^{}]*\})', re.DOTALL)

_ACTION_REQUIRED_FIELDS = {
    "update_address": ["order_id", "first_name", "last_name", "address", "city", "postcode", "phone"],
    "add_product": ["order_id", "product_id", "quantity", "total"],
    "remove_product": ["order_id", "order_product_id"],
}

_LLM_OPTIONS = {"temperature": 0.15}

# Require an explicit order-related keyword before treating a number as an order ID.
# This prevents postcodes (e.g. "11070") from being mistaken for order IDs.
_ORDER_ID_CUE = re.compile(
    r'(?:narud[žz]bin\w*|porud[žz]bin\w*|nalog|order)'
    r'(?:\s+(?:br\.?|broj))?\s*#?\s*(\d{4,})',
    re.IGNORECASE,
)

# Phrases the bot uses when asking the user for the order ID.
_ASKED_FOR_ORDER_ID = (
    "broj narudžbine", "broj porudžbine", "broj narudzbine",
    "id narudžbine", "id porudžbine", "navedite broj",
    "unesite broj", "koji je broj", "nalog broj",
)


def _normalize_sr(text: str) -> str:
    return (
        text.lower()
        .replace("š", "s").replace("đ", "dj").replace("č", "c").replace("ć", "c").replace("ž", "z")
    )


def _in_corpus(value: str, corpus: str) -> bool:
    """Return True if value can be traced to something the user actually typed."""
    if not value or not value.strip():
        return True
    nv = _normalize_sr(value.strip())
    nc = _normalize_sr(corpus)
    if nv in nc:
        return True
    # Fallback: every significant token (3+ chars) appears somewhere in corpus.
    tokens = [t for t in re.split(r"\W+", nv) if len(t) >= 3]
    return bool(tokens) and all(t in nc for t in tokens)


def _build_user_corpus(history: List[dict], current_message: str) -> str:
    parts = [m["content"] for m in history if m.get("role") == "user"]
    parts.append(current_message)
    return " ".join(parts)


def _extract_order_id_from_message(message: str, history: List[dict]) -> Optional[int]:
    """Extract an order ID only when there is an explicit cue in the message.

    Accepts:
    1. Explicit keyword immediately before the number ("narudžbina 12345", "order #12345").
    2. A bare 4+ digit number when the previous assistant turn asked for the order ID.

    A postcode like "11070" in an address string will never match case 1 (no keyword),
    and will only match case 2 if the bot literally just asked for an order number.
    """
    m = _ORDER_ID_CUE.search(message)
    if m:
        return int(m.group(1))

    # Case 2: bare number after the bot asked for the order ID.
    if history:
        last_asst = next(
            (h["content"] for h in reversed(history) if h.get("role") == "assistant"),
            "",
        )
        if any(p in last_asst.lower() for p in _ASKED_FOR_ORDER_ID):
            bare = re.search(r'\b(\d{4,})\b', message)
            if bare:
                return int(bare.group(1))

    return None


def _detect_pending_action(message: str) -> Optional[str]:
    """Infer the user's intended order mutation from their message text."""
    lower = message.lower()
    if any(w in lower for w in ("adresu", "adresa", "adrese", "dostavu", "dostave",
                                 "promen", "izmeni", "izmenu", "azuriraj", "ažuriraj")):
        return "update_address"
    if any(w in lower for w in ("dodaj", "dodati", "dodajte", "ubaci", "ubaciti")):
        return "add_product"
    if any(w in lower for w in ("ukloni", "ukloniti", "obriši", "obrisi", "izbaci")):
        return "remove_product"
    return None


def _should_skip_product_search(session_id: str, message: str) -> bool:
    """Return True when product RAG search would not be useful for this message.

    Skips when:
    - We are in an active address-update flow (numbers in the message are address
      parts, not product queries).
    - The message is a short confirmation or denial.
    """
    if order_state.get_pending_action(session_id) == "update_address":
        return True
    lower = message.lower().strip().rstrip(".!? ")
    return lower in {
        "da", "ne", "ok", "jeste", "jest", "tacno", "tačno",
        "potvrdi", "potvrđujem", "u redu", "uredu",
        "odustani", "odustaj", "ne hvala",
    }


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


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


def _validate_action(action: dict) -> Tuple[bool, str]:
    name = action.get("action")
    if name not in _ACTION_REQUIRED_FIELDS:
        return False, f"nepoznata akcija '{name}'"
    missing = [f for f in _ACTION_REQUIRED_FIELDS[name] if f not in action]
    if missing:
        return False, f"nedostaju polja: {', '.join(missing)}"
    return True, ""


async def _execute_action(
    action: dict, user_corpus: str
) -> Tuple[str, Optional[str]]:
    """Returns (result_message, fresh_order_card_or_None).

    user_corpus: all user messages in the session concatenated — used to verify
    that field values in the action were actually stated by the user, not invented.
    """
    valid, err = _validate_action(action)
    if not valid:
        return f"Greška u akciji: {err}.", None

    name = action["action"]

    if name == "update_address":
        order_id = int(action["order_id"])
        order = await fetch_order(order_id)
        if order is None:
            return f"Greška: narudžbina {order_id} nije pronađena.", None
        if order.get("status") in _LOCKED_STATUSES:
            return (
                f"Greška: narudžbina {order_id} ima status "
                f"'{order['status']}' i ne može se menjati.",
                None,
            )

        # first_name / last_name always come from DB — never trust model output.
        fn = order.get("first_name", "")
        ln = order.get("last_name", "")

        # For each changeable field: if model value differs from DB, it must
        # appear somewhere in what the user typed; otherwise the model invented it.
        db_vals = {
            "address": str(order.get("address") or ""),
            "city": str(order.get("city") or ""),
            "postcode": str(order.get("postcode") or ""),
            "phone": str(order.get("phone") or ""),
        }
        invented = [
            f for f in ("address", "city", "postcode", "phone")
            if str(action.get(f, "")).strip() != db_vals[f].strip()
            and not _in_corpus(str(action.get(f, "")), user_corpus)
        ]
        if invented:
            return (
                f"Greška: model je pokušao da postavi vrednosti za "
                f"{', '.join(invented)} koje kupac nije naveo. "
                "Pitaj kupca da eksplicitno navede te podatke.",
                None,
            )

        # No field differs from DB — nothing to update.
        changed = any(
            str(action.get(f, "")).strip() != db_vals[f].strip()
            for f in ("address", "city", "postcode", "phone")
        )
        if not changed:
            return (
                "Podaci o adresi su identični trenutnim u sistemu. "
                "Nisu napravljene izmene.",
                None,
            )

        await update_order_address(
            order_id, fn, ln,
            action["address"], action["city"], action["postcode"], action["phone"],
        )
        fresh_order = await fetch_order(order_id)
        fresh_items = await fetch_order_products(order_id)
        card = format_order_card(fresh_order, fresh_items) if fresh_order else ""
        return f"Adresa za narudžbinu {order_id} je uspešno ažurirana.", card

    if name == "add_product":
        order_id = int(action["order_id"])
        product_id = int(action["product_id"])

        order, products = await asyncio.gather(
            fetch_order(order_id),
            get_products_by_ids([product_id]),
        )
        if order is None:
            return f"Greška: narudžbina {order_id} nije pronađena.", None
        if order.get("status") in _LOCKED_STATUSES:
            return (
                f"Greška: narudžbina {order_id} ima status "
                f"'{order['status']}' i ne može se menjati.",
                None,
            )
        if not products:
            return "Greška: Proizvod sa tim ID-jem ne postoji u sistemu.", None

        product = products[0]
        qty = int(action["quantity"])
        if qty <= 0:
            return f"Greška: nevažeća količina {qty}.", None

        # Validate that total ≈ unit_price × qty (allow 5% or 1 RSD tolerance).
        unit = float(product.get("sale_price") or product.get("price") or 0)
        if unit > 0:
            expected = round(unit * qty, 2)
            actual = round(float(action["total"]), 2)
            if abs(actual - expected) > max(1.0, expected * 0.05):
                return (
                    f"Greška: ukupna cena {actual} RSD ne odgovara "
                    f"ceni proizvoda ({unit} × {qty} = {expected} RSD).",
                    None,
                )

        await add_order_product(order_id, product_id, qty, float(action["total"]))
        fresh_order = await fetch_order(order_id)
        fresh_items = await fetch_order_products(order_id)
        card = format_order_card(fresh_order, fresh_items) if fresh_order else ""
        return f"Proizvod je dodat u narudžbinu {order_id}.", card

    if name == "remove_product":
        order_id = int(action["order_id"])
        op_id = int(action["order_product_id"])

        order = await fetch_order(order_id)
        if order is None:
            return f"Greška: narudžbina {order_id} nije pronađena.", None
        if order.get("status") in _LOCKED_STATUSES:
            return (
                f"Greška: narudžbina {order_id} ima status "
                f"'{order['status']}' i ne može se menjati.",
                None,
            )

        order_items = await fetch_order_products(order_id)
        valid_op_ids = {item["id"] for item in order_items}
        if op_id not in valid_op_ids:
            return (
                f"Greška: stavka {op_id} ne postoji u narudžbini {order_id}.",
                None,
            )

        await remove_order_product(op_id)
        fresh_order = await fetch_order(order_id)
        fresh_items = await fetch_order_products(order_id)
        card = format_order_card(fresh_order, fresh_items) if fresh_order else ""
        return (
            f"Stavka (order_product_id={op_id}) je uklonjena iz narudžbine {order_id}.",
            card,
        )

    return f"Greška: nepoznata akcija '{name}'.", None


def _sse(text: str) -> str:
    # Encode literal newlines so SSE framing is not broken; frontend decodes \n back.
    return "data: " + text.replace("\n", "\\n") + "\n\n"


async def _lookup_order(
    message: str, session_id: str, history: List[dict]
) -> Tuple[Optional[str], Optional[int]]:
    """Stateful order lookup.

    Uses the session-locked order_id when one exists. Only extracts a new order ID
    from the message if there is an explicit keyword cue (or the bot just asked for one),
    preventing postcodes or house numbers from being mistaken for order IDs.

    Switches to a different order if the user references one explicitly.
    Returns (order_card, None) on success, (None, not_found_id) when the order
    doesn't exist, or (None, None) when there is no order context.
    """
    locked_id = order_state.get_locked_order_id(session_id)
    extracted_id = _extract_order_id_from_message(message, history)

    if extracted_id is not None and extracted_id != locked_id:
        # User referenced a new or different order — switch context.
        order_state.clear_order_state(session_id)
        order_id = extracted_id
    elif locked_id is not None:
        order_id = locked_id
    else:
        return None, None

    order = await fetch_order(order_id)
    if order:
        items = await fetch_order_products(order_id)
        card = format_order_card(order, items)
        order_state.lock_order(session_id, order_id)
        print(f"[order] fetched order {order_id}: status={order.get('status')}, {len(items)} products")
        return card, None

    if locked_id == order_id:
        order_state.clear_order_state(session_id)
    print(f"[order] order {order_id} not found")
    return None, order_id


async def _lookup_products(
    message: str, indexed: bool, session_id: str
) -> Optional[str]:
    """RAG discovery followed by fresh DB fetch for authoritative product data.

    Skips entirely when the session is in an address-update flow or the message
    is a short confirmation — those messages contain no product queries.
    """
    if not indexed:
        return None
    if _should_skip_product_search(session_id, message):
        return None
    rag_results = await search_products_rag(message)
    print(f"[rag] product query={message!r} → {len(rag_results)} results")
    if not rag_results:
        return None
    ids = [int(r["id"]) for r in rag_results if r.get("id")]
    if not ids:
        return None
    # Preserve RAG relevance order after DB re-fetch.
    id_to_product = {row["id"]: row for row in await get_products_by_ids(ids)}
    products = [id_to_product[i] for i in ids if i in id_to_product]
    if not products:
        return None
    return format_product_card(products)


def _build_messages(
    system_prompt: str,
    history: List[dict],
    indexed: bool,
    product_card: Optional[str],
    order_card: Optional[str],
    order_not_found_id: Optional[int],
    user_message: str,
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if not indexed:
        messages.append({
            "role": "system",
            "content": (
                "SISTEMSKI KONTEKST: Indeks proizvoda se još učitava. "
                "Odgovaraj samo na osnovu podataka o narudžbini ako su dostupni. "
                "Ne pominjaj proizvode."
            ),
        })
    if product_card:
        messages.append({
            "role": "system",
            "content": (
                f"SISTEMSKI KONTEKST — Dostupni proizvodi:\n{product_card}\n\n"
                "Koristi ove podatke TAČNO kako su navedeni. Nemoj ih dopunjavati ni menjati."
            ),
        })
    if order_card:
        messages.append({
            "role": "system",
            "content": (
                f"SISTEMSKI KONTEKST — Podaci o narudžbini:\n{order_card}\n\n"
                "Prikaži ove podatke TAČNO kako su navedeni. Nemoj ih prepričavati ni dopunjavati."
            ),
        })
    if order_not_found_id is not None:
        messages.append({
            "role": "system",
            "content": (
                f"SISTEMSKI KONTEKST: Narudžbina sa ID {order_not_found_id} "
                "nije pronađena u sistemu."
            ),
        })
    messages.append({"role": "user", "content": user_message})
    return messages


async def _generate_stream(body: ChatRequest) -> AsyncGenerator[str, None]:
    indexed = is_indexed()
    system_prompt = build_system_prompt()
    history = get_history(body.session_id)
    user_corpus = _build_user_corpus(history, body.message)

    order_card, order_not_found_id = await _lookup_order(body.message, body.session_id, history)
    product_card = await _lookup_products(body.message, indexed, body.session_id)

    if order_card and order_state.get_locked_order_id(body.session_id):
        hint = _detect_pending_action(body.message)
        if hint:
            order_state.set_pending_action(body.session_id, hint)

    messages = _build_messages(
        system_prompt, history, indexed,
        product_card, order_card, order_not_found_id, body.message,
    )

    full_reply = ""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0)
        ) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": _LLM_OPTIONS,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_reply += content
                        yield _sse(content)
                    if chunk.get("done"):
                        break
    except httpx.TimeoutException:
        yield _sse("Trenutno mi treba malo više vremena. Pokušajte ponovo.")
        return
    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"
        return

    print(f"[llm] stream done, full reply: {full_reply!r}")

    action = _extract_action(full_reply)
    if action:
        print(f"[action] executing: {action}")
        result_msg, fresh_card = await _execute_action(action, user_corpus)
        print(f"[action] result: {result_msg}")

        if fresh_card is not None:
            order_state.set_pending_action(body.session_id, None)

        confirm_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"Sistemski rezultat: {result_msg} "
                    "Odgovori kupcu jednom kratkom rečenicom na srpskom. "
                    "Nemoj ponavljati detalje narudžbine."
                ),
            },
        ]
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0)
            ) as client:
                cr = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": confirm_messages,
                        "stream": False,
                        "options": _LLM_OPTIONS,
                    },
                )
                cr.raise_for_status()
                confirmation = _strip_action_block(cr.json()["message"]["content"])
        except Exception:
            confirmation = result_msg

        reply_suffix = confirmation + ("\n\n" + fresh_card if fresh_card else "")
        yield _sse("\n\n")
        yield _sse(reply_suffix)
        final_reply = _strip_action_block(full_reply) + "\n\n" + reply_suffix
    else:
        final_reply = _strip_action_block(full_reply)

    add_turn(body.session_id, body.message, final_reply)
    yield "data: [DONE]\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    if settings.force_reindex:
        asyncio.create_task(reindex_all())
        print("[rag] force reindex started in background", flush=True)
    else:
        asyncio.create_task(maybe_index())
        print("[rag] incremental index check started in background", flush=True)
    yield
    await close_db()


app = FastAPI(title="Mango Bot", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    indexed = is_indexed()
    system_prompt = build_system_prompt()
    history = get_history(body.session_id)
    user_corpus = _build_user_corpus(history, body.message)

    order_card, order_not_found_id = await _lookup_order(body.message, body.session_id, history)
    product_card = await _lookup_products(body.message, indexed, body.session_id)

    if order_card and order_state.get_locked_order_id(body.session_id):
        hint = _detect_pending_action(body.message)
        if hint:
            order_state.set_pending_action(body.session_id, hint)

    messages = _build_messages(
        system_prompt, history, indexed,
        product_card, order_card, order_not_found_id, body.message,
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0)
        ) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": _LLM_OPTIONS,
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return ChatResponse(reply="Trenutno mi treba malo više vremena. Pokušajte ponovo.")

    raw_reply = data["message"]["content"]
    print(f"[llm] raw reply: {raw_reply!r}")

    action = _extract_action(raw_reply)
    if action:
        print(f"[action] executing: {action}")
        result_msg, fresh_card = await _execute_action(action, user_corpus)
        print(f"[action] result: {result_msg}")

        if fresh_card is not None:
            order_state.set_pending_action(body.session_id, None)

        confirm_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"Sistemski rezultat: {result_msg} "
                    "Odgovori kupcu jednom kratkom rečenicom na srpskom. "
                    "Nemoj ponavljati detalje narudžbine."
                ),
            },
        ]
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0)
            ) as client:
                confirm_response = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": confirm_messages,
                        "stream": False,
                        "options": _LLM_OPTIONS,
                    },
                )
                confirm_response.raise_for_status()
                confirmation = _strip_action_block(confirm_response.json()["message"]["content"])
        except Exception:
            confirmation = result_msg

        final_reply = confirmation + ("\n\n" + fresh_card if fresh_card else "")
    else:
        final_reply = _strip_action_block(raw_reply)

    add_turn(body.session_id, body.message, final_reply)
    return ChatResponse(reply=final_reply)


@app.post("/admin/reindex")
async def admin_reindex():
    asyncio.create_task(reindex_all())
    return JSONResponse({"status": "full reindex started"})


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    return StreamingResponse(
        _generate_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    clear_session(session_id)
    order_state.clear_order_state(session_id)
    return JSONResponse({"status": "cleared"})

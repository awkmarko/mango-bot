from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OrderFlowState:
    order_id: Optional[int] = None        # locked once confirmed in DB
    pending_action: Optional[str] = None  # "update_address" | "add_product" | "remove_product"


_store: Dict[str, OrderFlowState] = {}


def get_order_state(session_id: str) -> OrderFlowState:
    return _store.setdefault(session_id, OrderFlowState())


def lock_order(session_id: str, order_id: int) -> None:
    get_order_state(session_id).order_id = order_id


def get_locked_order_id(session_id: str) -> Optional[int]:
    return get_order_state(session_id).order_id


def set_pending_action(session_id: str, action: Optional[str]) -> None:
    get_order_state(session_id).pending_action = action


def get_pending_action(session_id: str) -> Optional[str]:
    return get_order_state(session_id).pending_action


def clear_order_state(session_id: str) -> None:
    _store.pop(session_id, None)

from collections import defaultdict
from typing import Dict, List
from app.config import settings

# Each value is a list of {"role": "user"|"assistant", "content": str}
_store: Dict[str, List[Dict]] = defaultdict(list)


def get_history(session_id: str) -> List[Dict]:
    return list(_store[session_id])


def add_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    history = _store[session_id]
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})

    # Keep only the last MAX_HISTORY_TURNS pairs (each pair = 2 messages)
    max_messages = settings.max_history_turns * 2
    if len(history) > max_messages:
        _store[session_id] = history[-max_messages:]


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)

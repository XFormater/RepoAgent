"""Small, auditable query expansion for lexical memory retrieval."""

from __future__ import annotations

from nanobot.memory_system.storage.lexical import lexical_tokens

_EXPANSIONS = {
    "activity": (
        "activities hobby hobbies leisure enjoys exercise sport art class trip "
        "camping painting swimming pottery running reading playing volunteering"
    ),
    "feel": "feeling felt mood emotion emotional happy sad upset alone solitude",
    "reason": "reason because caused motivation motivated decided decision",
    "prefer": "preference prefers preferred choose chose choice rather favorite enjoys",
    "time": "date day month year before after recently last next when",
    "place": "place location city country where moved visited went",
    "work": "work job career profession business company studio employed",
    "relationship": "relationship family friend partner spouse bride groom wedding together",
    "plan": "plan plans planned intend goal future upcoming will",
}

_TRIGGERS = {
    "activity": {"activity", "activities", "hobby", "hobbies", "partake", "spend"},
    "feel": {"feel", "feeling", "felt", "mood"},
    "reason": {"why", "reason", "decide", "decided"},
    "prefer": {"prefer", "preference", "rather", "favorite", "choose"},
    "time": {"when", "date", "before", "after"},
    "place": {"where", "location", "place"},
    "work": {"work", "job", "career", "business"},
    "relationship": {"family", "friend", "partner", "bride", "groom", "wedding"},
    "plan": {"plan", "plans", "future", "upcoming"},
}


def expand_memory_query(query: str) -> str:
    """Append bounded intent synonyms while leaving the original query intact."""
    terms = set(lexical_tokens(query))
    additions = [
        expansion
        for intent, expansion in _EXPANSIONS.items()
        if terms.intersection(_TRIGGERS[intent])
    ]
    return f"{query}\nRetrieval synonyms: {' '.join(additions)}" if additions else query

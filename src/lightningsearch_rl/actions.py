from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedAction:
    type: str
    content: str
    valid: bool
    raw: str


_ACTION_RE = re.compile(r"^\s*<(think|search|answer)>(.*?)</\1>\s*$", re.DOTALL)


def parse_action(raw: str) -> ParsedAction:
    match = _ACTION_RE.match(raw)
    if not match:
        return ParsedAction(type="invalid", content="", valid=False, raw=raw)
    action_type, content = match.group(1), match.group(2).strip()
    return ParsedAction(
        type=action_type,
        content=content,
        valid=bool(content),
        raw=raw,
    )

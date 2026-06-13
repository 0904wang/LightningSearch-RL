from __future__ import annotations

from lightningsearch_rl.data import Passage
from lightningsearch_rl.runtime import EpisodeTrace


def format_observation(passages: list[Passage]) -> str:
    lines = ["<observation>"]
    for index, passage in enumerate(passages, start=1):
        prefix = f"{passage.title}: " if passage.title else ""
        lines.append(f"[{index}] {prefix}{passage.text}")
    lines.append("</observation>")
    return "\n".join(lines)


def format_assistant_trace(trace: EpisodeTrace) -> str:
    search_step = next((step for step in trace.steps if step.action_type == "search"), None)
    observation = search_step.observation if search_step and search_step.observation else []
    search_action = search_step.action if search_step else "<search></search>"

    return "\n".join(
        [
            "<think>I should search for evidence.</think>",
            search_action,
            format_observation(observation),
            "<think>The retrieved evidence supports the answer.</think>",
            f"<answer>{trace.final_answer}</answer>",
        ]
    )

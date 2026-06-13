from __future__ import annotations

from dataclasses import asdict, dataclass

from lightningsearch_rl.runtime import EpisodeTrace


@dataclass(frozen=True)
class Transition:
    question_id: str
    step_index: int
    state: str
    action: str
    action_type: str
    observation: list[dict] | None
    terminal: bool
    reward: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_transitions(trace: EpisodeTrace) -> list[Transition]:
    transitions: list[Transition] = []
    for index, step in enumerate(trace.steps):
        transitions.append(
            Transition(
                question_id=trace.question_id,
                step_index=index,
                state=step.state,
                action=step.action,
                action_type=step.action_type,
                observation=[asdict(p) for p in step.observation] if step.observation else None,
                terminal=step.terminal,
                reward=trace.reward if step.terminal else None,
            )
        )
    return transitions

from __future__ import annotations

from collections.abc import Mapping

# Exact GoEmotions-compatible output order used by proxy3d/multi-motions-28.
EMOTIONS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def vector(values: Mapping[str, float], *, clip: bool = True) -> dict[str, float]:
    """Return a dense 28-D vector in classifier label order.

    Multi-Motions 28 is multi-label: its sigmoid confidence scores are not a
    categorical probability distribution and must *not* be renormalized to sum
    to one.  This helper therefore only fills missing coordinates and, when
    requested, clips values to the common [0, 1] scale.
    """
    result: dict[str, float] = {}
    for name in EMOTIONS:
        value = float(values.get(name, 0.0))
        if clip:
            value = max(0.0, min(1.0, value))
        result[name] = value
    return result


def mix_state(
    user_state: Mapping[str, float],
    agent_state: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mixed = {
        name: alpha * float(user_state.get(name, 0.0))
        + (1.0 - alpha) * float(agent_state.get(name, 0.0))
        for name in EMOTIONS
    }
    return vector(mixed)


def mix_state_with_memory(
    user_state: Mapping[str, float],
    agent_state: Mapping[str, float],
    memory_state: Mapping[str, float],
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Mix current user, agent and associative-memory signals on one scale.

    U_t and E_t are in [0, 1]. A_t is kept in the same bounded scale. The
    additive memory term can push a coordinate above one, so the educational
    demo clips the mixed vector back to [0, 1]. It does not perform
    sum-normalization because the 28 emotion coordinates are independent
    sigmoid confidences rather than mutually exclusive class probabilities.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if beta < 0.0:
        raise ValueError("beta must be >= 0")
    mixed = {
        name: alpha * float(user_state.get(name, 0.0))
        + (1.0 - alpha) * float(agent_state.get(name, 0.0))
        + beta * float(memory_state.get(name, 0.0))
        for name in EMOTIONS
    }
    return vector(mixed)


def apply_character_weights(
    communication_state: Mapping[str, float],
    character_weights: Mapping[str, float],
) -> dict[str, float]:
    """Apply the article's W_character elementwise to S_t.

    The weights belong to the character policy, not to the communication-style
    microdialogue table. They are deliberately not clipped to [0, 1]: the
    article allows weights that strengthen a coordinate (for example 1.5),
    while zero makes that reaction unavailable.
    """
    scores: dict[str, float] = {}
    for name in EMOTIONS:
        weight = float(character_weights.get(name, 1.0))
        if weight < 0.0:
            raise ValueError(f"character weight for {name!r} must be >= 0")
        scores[name] = float(communication_state.get(name, 0.0)) * weight
    return scores


def update_toy_agent_state(
    agent_state: Mapping[str, float],
    user_state: Mapping[str, float],
    inertia: float = 0.8,
) -> dict[str, float]:
    """Transparent stand-in for the article's external recurrent state model."""
    if not 0.0 <= inertia <= 1.0:
        raise ValueError("inertia must be in [0, 1]")
    updated = {
        name: inertia * float(agent_state.get(name, 0.0))
        + (1.0 - inertia) * float(user_state.get(name, 0.0))
        for name in EMOTIONS
    }
    return vector(updated)

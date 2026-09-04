"""Logged-support checks for teacher-forced and closed-loop trajectories."""


def first_unsupported_step(logged_actions: list[str], generated_actions: list[str]) -> int | None:
    """Return the first generated action whose continuation is absent from the log."""
    for index, action in enumerate(generated_actions):
        if index >= len(logged_actions) or action != logged_actions[index]:
            return index
    return None


def support_trace(logged_actions: list[str], generated_actions: list[str]) -> list[dict]:
    unsupported = first_unsupported_step(logged_actions, generated_actions)
    return [{"step": i, "generated_action": a, "logged_action": logged_actions[i] if i < len(logged_actions) else None,
             "on_logged_support": unsupported is None or i < unsupported}
            for i, a in enumerate(generated_actions)]

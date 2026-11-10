"""Small, ROS-independent helpers for drive-mode arbitration."""

from typing import Optional


class ManualOverrideGate:
    """Keep autonomous output muted while manual commands are fresh.

    The F1TENTH joystick stack publishes ``/teleop`` in manual/neutral mode
    and stops publishing it while the RB autonomous deadman is active.  The
    gate starts muted so that an autonomous command cannot leak out before the
    first joystick message arrives.
    """

    def __init__(self, enabled: bool, timeout: float):
        if timeout <= 0.0:
            raise ValueError('manual override timeout must be positive')
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self._last_manual_command = None  # type: Optional[float]

    def observe_manual_command(self, now: float) -> None:
        """Record receipt of a command from the manual-control input."""
        self._last_manual_command = float(now)

    def manual_active(self, now: float) -> bool:
        """Return whether autonomous output must currently be suppressed."""
        if not self.enabled:
            return False
        if self._last_manual_command is None:
            return True
        age = float(now) - self._last_manual_command
        # Treat a backwards clock jump as fresh manual input as the safe state.
        return age < 0.0 or age <= self.timeout

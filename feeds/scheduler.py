"""Polling scheduler with dynamic cadence (Stage 6, Sec 8).

Registers per-source tasks with intervals, per-source rate limits, backoff and
jitter. Switches between RESEARCH mode (bounded iterations, for one-shot studies)
and PRODUCTION mode (continuous). High-alert mode shortens cadence for affected
markets when a news event or anchor move fires.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class Task:
    name: str
    fn: Callable[[], object]
    interval: float                 # seconds (baseline)
    alert_interval: float | None = None
    last_run: float = 0.0
    failures: int = 0
    runs: int = 0
    errors: int = 0


@dataclass
class Scheduler:
    mode: str = "research"          # "research" | "production"
    high_alert: bool = False
    tasks: list[Task] = field(default_factory=list)
    jitter: float = 0.3

    def add(self, name, fn, interval, alert_interval=None):
        self.tasks.append(Task(name, fn, interval, alert_interval))

    def _due(self, t: Task, now: float) -> bool:
        iv = t.alert_interval if (self.high_alert and t.alert_interval) else t.interval
        backoff = min(2 ** t.failures, 16)            # exponential backoff on failure
        return now - t.last_run >= iv * backoff

    def run_once(self) -> dict:
        now = time.time()
        ran = {}
        for t in self.tasks:
            if not self._due(t, now):
                continue
            time.sleep(random.uniform(0, self.jitter))  # anti-robotic jitter
            t.last_run = time.time()
            t.runs += 1
            try:
                ran[t.name] = t.fn()
                t.failures = 0
            except Exception as exc:  # noqa: BLE001
                t.failures += 1
                t.errors += 1
                log.warning("task %s failed: %s", t.name, exc)
                ran[t.name] = {"error": str(exc)[:120]}
        return ran

    def run(self, iterations: int | None = None, tick=1.0):
        """RESEARCH: bounded iterations. PRODUCTION: iterations=None -> loop."""
        i = 0
        while iterations is None or i < iterations:
            self.run_once()
            i += 1
            time.sleep(tick)

    def health(self) -> list[dict]:
        return [{"task": t.name, "runs": t.runs, "errors": t.errors,
                 "fail_streak": t.failures} for t in self.tasks]

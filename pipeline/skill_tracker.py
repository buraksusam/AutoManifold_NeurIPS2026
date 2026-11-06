"""
Skill Tracking System for PUA and Superpowers Integration.

Tracks when PUA/Superpowers skills are invoked during agent execution,
including timestamps, parameters, and results.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from threading import Lock

from .config import OUTPUT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class SkillInvocation:
    """Record of a single skill invocation."""
    skill_name: str
    skill_type: str  # "pua" or "superpowers"
    agent_name: str
    agent_id: int
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    result_summary: str = ""
    success: bool = True
    error_message: str = ""


@dataclass
class SkillUsageLog:
    """Container for all skill invocations in a session/run."""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    pua_plugin_active: Optional[bool] = None
    invocations: List[SkillInvocation] = field(default_factory=list)

    def add_invocation(self, invocation: SkillInvocation):
        """Add a skill invocation to the log."""
        self.invocations.append(invocation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert invocations to dict (handle both dataclass and regular dict)
        invocations_list = []
        for inv in self.invocations:
            if hasattr(inv, '__dataclass_fields__'):
                invocations_list.append(asdict(inv))
            else:
                # Convert to dict manually for regular objects
                invocations_list.append({
                    "skill_name": inv.skill_name,
                    "skill_type": inv.skill_type,
                    "agent_name": inv.agent_name,
                    "agent_id": inv.agent_id,
                    "timestamp": inv.timestamp,
                    "parameters": inv.parameters,
                    "duration_ms": inv.duration_ms,
                    "result_summary": inv.result_summary,
                    "success": inv.success,
                    "error_message": inv.error_message,
                })

        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pua_plugin_active": self.pua_plugin_active,
            "total_invocations": len(self.invocations),
            "invocations_by_type": {
                "pua": len([i for i in self.invocations if i.skill_type == "pua"]),
                "superpowers": len([i for i in self.invocations if i.skill_type == "superpowers"]),
            },
            "invocations_by_agent": self._count_by_agent(),
            "invocations_by_skill": self._count_by_skill(),
            "invocations": invocations_list,
        }

    def _count_by_agent(self) -> Dict[str, int]:
        """Count invocations per agent."""
        counts: Dict[str, int] = {}
        for inv in self.invocations:
            key = f"{inv.agent_name} (Agent {inv.agent_id})"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_skill(self) -> Dict[str, int]:
        """Count invocations per skill."""
        counts: Dict[str, int] = {}
        for inv in self.invocations:
            key = f"{inv.skill_name} ({inv.skill_type})"
            counts[key] = counts.get(key, 0) + 1
        return counts


class SkillTracker:
    """
    Central tracker for all PUA/Superpowers skill invocations.

    Thread-safe singleton pattern for global access across all agents.
    """

    _instance: Optional["SkillTracker"] = None
    _lock: Lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._current_log: Optional[SkillUsageLog] = None
        self._log_lock = Lock()
        self._output_dir = os.path.join(OUTPUT_ROOT, "skill_logs")
        os.makedirs(self._output_dir, exist_ok=True)

        logger.info("SkillTracker initialized")

    def start_run(self, run_id: str) -> None:
        """Start a new run log."""
        with self._log_lock:
            self._current_log = SkillUsageLog(
                run_id=run_id,
                started_at=datetime.now().isoformat(),
            )
            logger.info(f"Skill tracking started for run: {run_id}")

    def end_run(self) -> Optional[str]:
        """End current run and save log. Returns path to saved log file."""
        with self._log_lock:
            if self._current_log is None:
                return None

            self._current_log.completed_at = datetime.now().isoformat()
            log_path = self._save_log(self._current_log)
            self._current_log = None
            logger.info(f"Skill tracking ended. Log saved to: {log_path}")
            return log_path

    def track_skill(
        self,
        skill_name: str,
        skill_type: str,
        agent_name: str,
        agent_id: int,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional["_SkillInvocationContext"]:
        """
        Track a skill invocation. Returns a context manager for tracking duration and result.

        Usage:
            with tracker.track_skill("brainstorming", "superpowers", "DataCharacterization", 1, {"topic": "design"}):
                # ... execute skill ...
                pass
        """
        if self._current_log is None:
            # Auto-start a run if none exists
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.start_run(run_id)

        invocation = SkillInvocation(
            skill_name=skill_name,
            skill_type=skill_type,
            agent_name=agent_name,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            parameters=parameters or {},
        )

        return _SkillInvocationContext(self, invocation)

    def log_invocation(self, invocation: SkillInvocation) -> None:
        """Directly log a completed skill invocation."""
        with self._log_lock:
            if self._current_log:
                self._current_log.add_invocation(invocation)

        # Also log to standard logger
        logger.info(
            f"[SKILL] {invocation.skill_type.upper()}/{invocation.skill_name} "
            f"by {invocation.agent_name} (Agent {invocation.agent_id}) "
            f"duration={invocation.duration_ms}ms success={invocation.success}"
        )

    def _save_log(self, log: SkillUsageLog) -> str:
        """Save skill usage log to file."""
        filename = f"{log.run_id}_skills.json"
        log_path = os.path.join(self._output_dir, filename)

        with open(log_path, "w") as f:
            json.dump(log.to_dict(), f, indent=2)

        return log_path

    def get_current_summary(self) -> Dict[str, Any]:
        """Get summary of current run's skill usage."""
        with self._log_lock:
            if self._current_log:
                return self._current_log.to_dict()
            return {"error": "No active run"}

    def is_skill_active(self, skill_name: str) -> bool:
        """Check if a specific skill has been used in current run."""
        with self._log_lock:
            if self._current_log:
                return any(inv.skill_name == skill_name for inv in self._current_log.invocations)
            return False


class _SkillInvocationContext:
    """Context manager for tracking skill execution duration and result."""

    def __init__(self, tracker: SkillTracker, invocation: SkillInvocation):
        self.tracker = tracker
        self.invocation = invocation
        self._start_time = None

    def __enter__(self) -> "_SkillInvocationContext":
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            self.invocation.duration_ms = None
        else:
            duration = (time.time() - self._start_time) * 1000  # Convert to ms
            self.invocation.duration_ms = round(duration, 2)
        self.invocation.success = exc_type is None

        if exc_type is not None:
            self.invocation.error_message = str(exc_val)

        self.tracker.log_invocation(self.invocation)


# Global convenience functions
def get_tracker() -> SkillTracker:
    """Get the global SkillTracker instance."""
    return SkillTracker()


def track_skill(
    skill_name: str,
    skill_type: str,
    agent_name: str,
    agent_id: int,
    parameters: Optional[Dict[str, Any]] = None,
) -> Optional[_SkillInvocationContext]:
    """
    Convenience function to track a skill invocation.

    Usage:
        from pipeline.skill_tracker import track_skill
        with track_skill("brainstorming", "superpowers", "MyAgent", 1):
            # execute skill
            pass
    """
    return get_tracker().track_skill(
        skill_name=skill_name,
        skill_type=skill_type,
        agent_name=agent_name,
        agent_id=agent_id,
        parameters=parameters,
    )


def get_skill_summary() -> Dict[str, Any]:
    """Get the current skill usage summary."""
    return get_tracker().get_current_summary()

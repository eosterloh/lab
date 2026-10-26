"""GPU mutex: one Train commit on the box at a time."""

from __future__ import annotations

from lab.config import LabConfig
from lab.supervisor import Supervisor
from tests.helpers import arm_for_train


def test_gpu_lock_blocks_second_supervisor_until_first_releases(tmp_path, transport) -> None:
    lock = tmp_path / "gpu.lock"
    a = Supervisor(LabConfig(run_dir=tmp_path / "a", gpu_lock_path=lock))
    b = Supervisor(LabConfig(run_dir=tmp_path / "b", gpu_lock_path=lock), transport=transport)

    arm_for_train(a)
    started = a.call("enter_train")
    assert started["ok"] is True
    assert started["job"]["status"] == "succeeded"

    arm_for_train(b)
    blocked = b.call("enter_train")
    assert blocked["ok"] is False
    assert "gpu" in blocked["error"]

    assert a.call("enter_eval")["ok"]
    assert b.call("enter_train")["ok"] is True


def test_run_policy_backs_off_when_enter_train_hits_a_held_gpu_lock(tmp_path, transport, monkeypatch) -> None:
    lock = tmp_path / "gpu.lock"
    holder = Supervisor(LabConfig(run_dir=tmp_path / "holder", gpu_lock_path=lock), transport=transport)
    waiter = Supervisor(LabConfig(run_dir=tmp_path / "waiter", gpu_lock_path=lock), transport=transport)
    arm_for_train(holder)
    assert holder.call("enter_train")["ok"]
    arm_for_train(waiter)
    sleeps: list[float] = []

    def sleeper(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 2:
            assert holder.call("enter_eval")["ok"]

    monkeypatch.setattr("lab.supervisor.time.sleep", sleeper)
    monkeypatch.setattr("lab.supervisor.GPU_LOCK_BACKOFF_S", (0.5, 1.0, 2.0))

    class Policy:
        def act(self, obs: dict) -> tuple[str, dict]:
            phase = obs.get("phase")
            if phase == "research":
                return "enter_train", {}
            if phase == "train":
                return "enter_eval", {}
            if phase == "eval" and not obs.get("eval_ran"):
                return "run_eval", {}
            return "halt", {"reason": "done"}

    result = waiter.run_policy(Policy(), max_cycles=1, max_steps=20, max_error_streak=4)
    assert sleeps[:2] == [0.5, 1.0]
    assert result["halt_reason"] == "max_cycles reached"
    assert result["completed_cycles"] == 1
    assert "consecutive" not in (result["halt_reason"] or "")


def test_exhausted_gpu_backoff_halts_with_the_lock_reason(tmp_path, transport, monkeypatch) -> None:
    lock = tmp_path / "gpu.lock"
    holder = Supervisor(LabConfig(run_dir=tmp_path / "holder", gpu_lock_path=lock), transport=transport)
    waiter = Supervisor(LabConfig(run_dir=tmp_path / "waiter", gpu_lock_path=lock), transport=transport)
    arm_for_train(holder)
    assert holder.call("enter_train")["ok"]
    arm_for_train(waiter)
    monkeypatch.setattr("lab.supervisor.time.sleep", lambda _delay: None)
    monkeypatch.setattr("lab.supervisor.GPU_LOCK_BACKOFF_S", (0, 0))

    class Policy:
        def act(self, obs: dict) -> tuple[str, dict]:
            return "enter_train", {}

    result = waiter.run_policy(Policy(), max_cycles=1, max_steps=30, max_error_streak=12)
    assert result["halted"] is True
    assert "gpu lock" in result["halt_reason"]
    assert "consecutive" not in result["halt_reason"]
    assert result["steps"] < 12

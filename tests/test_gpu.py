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

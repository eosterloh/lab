from __future__ import annotations

from lab.config import LabConfig
from lab.policy import dummy_pack
from lab.supervisor import Supervisor


def test_gpu_lock_blocks_second_train(tmp_path, transport) -> None:
    a = Supervisor(LabConfig(run_dir=tmp_path / "a", gpu_lock_path=tmp_path / "gpu.lock"))
    b = Supervisor(
        LabConfig(run_dir=tmp_path / "b", gpu_lock_path=tmp_path / "gpu.lock"),
        transport=transport,
    )
    assert a.call("enter_research")["ok"]
    pack = dummy_pack(a.observe())
    assert a.call("write_pack", {"pack": pack})["ok"]
    started = a.call("enter_train")
    assert started["ok"] is True
    assert started["job"]["status"] == "succeeded"
    assert b.call("enter_research")["ok"]
    pack_b = dummy_pack(b.observe())
    assert b.call("write_pack", {"pack": pack_b})["ok"]
    blocked = b.call("enter_train")
    assert blocked["ok"] is False
    assert "gpu" in blocked["error"]
    assert a.call("enter_eval")["ok"]
    assert b.call("enter_train")["ok"] is True

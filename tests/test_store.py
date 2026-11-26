import pytest

from wm.store import LocalStore, Store, make_store, read_candidates, write_name


def _file(tmp_path, text="hello"):
    p = tmp_path / "payload.txt"
    p.write_text(text)
    return p


def test_naming_rules():
    assert write_name("ckpt-wm", "smoke") == "ckpt-wm-smoke"
    assert write_name("ckpt-wm", "full") == "ckpt-wm"
    assert read_candidates("ckpt-wm", "smoke") == ["ckpt-wm", "ckpt-wm-smoke"]
    assert read_candidates("ckpt-wm", "full") == ["ckpt-wm"]


def test_local_roundtrip(tmp_path):
    store = make_store("local", "smoke")
    assert not store.exists("thing")
    store.push("thing", _file(tmp_path), {"complete": True, "config_hash": "abc"})
    assert store.exists("thing")
    assert (tmp_path / "store" / "thing-smoke" / "meta.json").exists()
    pulled = store.pull("thing")
    assert (pulled / "payload.txt").read_text() == "hello"


def test_metadata_always_has_required_keys(tmp_path):
    store = make_store("local", "smoke")
    store.push("thing", _file(tmp_path), {"complete": False, "config_hash": "abc", "step": 5})
    meta = store.metadata("thing")
    assert meta["complete"] is False and meta["config_hash"] == "abc" and meta["step"] == 5
    assert "git" in meta
    with pytest.raises(KeyError):
        store.push("thing", _file(tmp_path), {"complete": True})


def test_smoke_prefers_full_artifact(tmp_path):
    backend = LocalStore()
    full, smoke = Store(backend, "full"), Store(backend, "smoke")
    smoke.push("data", _file(tmp_path, "smoke"), {"complete": True, "config_hash": "x"})
    assert (smoke.pull("data") / "payload.txt").read_text() == "smoke"
    assert not full.exists("data")  # full never reads -smoke
    full.push("data", _file(tmp_path, "full"), {"complete": True, "config_hash": "x"})
    assert (smoke.pull("data") / "payload.txt").read_text() == "full"
    # skip/resume looks only at what this mode writes
    assert smoke.own_metadata("data")["config_hash"] == "x"


def test_wandb_store_needs_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        make_store("kaggle", "smoke")

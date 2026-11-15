"""Codec checkpoint loading accesses only the intended section through CPU mmap."""
from __future__ import annotations

import pytest
import torch


def test_loader_uses_cpu_mmap_and_selected_section(tmp_path, monkeypatch):
    from sglang_omni.models.zonos3.components import codec
    checkpoint = tmp_path / "codec.pt"
    w = torch.randn((16, 32), dtype=torch.bfloat16)
    torch.save({
        "generator": {"test.weight": w},
        "unrelated_training_section": {"other.weight": torch.ones(1024 * 1024)}
    }, checkpoint)
    calls = []
    original = torch.load

    def inspect_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(codec.torch, "load", inspect_load)
    monkeypatch.setattr(codec, "fold_weight_norm", lambda weights: weights)
    result = codec.load_checkpoint_state(checkpoint, "generator", device="cpu")
    assert len(calls) == 1
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is True
    assert calls[0]["mmap"] is True
    assert list(result) == ["test.weight"]
    assert result["test.weight"].dtype == torch.float32
    assert result["test.weight"].device.type == "cpu"
    torch.testing.assert_close(result["test.weight"], w.float(), rtol=0, atol=0)


def test_missing_section_errors(tmp_path):
    from sglang_omni.models.zonos3.components import codec
    path = tmp_path / "codec.pt"
    torch.save({"generator": {"test.weight": torch.zeros(1)}}, path)
    with pytest.raises(KeyError):
        codec.load_checkpoint_state(path, "not_found", device="cpu")

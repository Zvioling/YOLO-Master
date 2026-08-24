# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import csv
<<<<<<< HEAD
import json
import urllib
from copy import copy
from pathlib import Path
from unittest import mock
=======
import os
import shutil
import tarfile
import urllib
import zipfile
from copy import copy
from pathlib import Path
>>>>>>> origin/main

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

<<<<<<< HEAD
from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA, TASKS
from ultralytics.data.build import load_inference_source
from ultralytics.data.utils import check_det_dataset
=======
import ultralytics.data.build as data_build
from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, load_inference_source
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
>>>>>>> origin/main
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
<<<<<<< HEAD
=======
    DATASETS_DIR,
>>>>>>> origin/main
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    IS_JETSON,
    IS_RASPBERRYPI,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    YAML,
    checks,
    is_github_action_running,
)
<<<<<<< HEAD
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13


def make_stub_yolo():
    """Create a lightweight YOLO instance without loading weights from disk or network."""
    model = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(model)
    model._check_is_pytorch_model = lambda: None
    model.trainer = mock.Mock()
    model.model = mock.Mock()
    return model
=======
from ultralytics.utils.downloads import download, safe_download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13


def test_dataloader_caps_workers_to_batches():
    """Test tiny datasets do not spawn persistent workers beyond useful batch count."""
    single_batch = build_dataloader(range(4), batch=4, workers=8)
    drop_last_single_batch = build_dataloader(range(5), batch=4, workers=8, drop_last=True)
    two_batches = build_dataloader(range(8), batch=4, workers=8)
    try:
        assert single_batch.num_workers == 0
        assert drop_last_single_batch.num_workers == 0
        assert two_batches.num_workers <= 2
    finally:
        single_batch.close()
        drop_last_single_batch.close()
        two_batches.close()


def test_dataloader_cap_preserves_distributed_drop_last(monkeypatch):
    """Test worker cap follows distributed sampler size without changing global drop_last behavior."""
    sampler_cls = data_build.distributed.DistributedSampler

    def distributed_sampler(dataset, shuffle):
        return sampler_cls(dataset, num_replicas=3, rank=0, shuffle=shuffle)

    monkeypatch.setattr(data_build.distributed, "DistributedSampler", distributed_sampler)
    loader = build_dataloader(range(8), batch=4, workers=8, rank=0, drop_last=True)
    try:
        assert len(loader) == 1
        assert loader.num_workers == 0
    finally:
        loader.close()


def test_dataloader_empty_dataset_uses_dataloader_validation():
    """Test empty datasets fail through DataLoader validation instead of worker-cap math."""
    with pytest.raises(ValueError, match="positive integer"):
        build_dataloader([], batch=4, workers=2)


def test_cfg_rejects_fuzzed_values():
    """Test invalid overrides fail in config validation."""
    with pytest.raises(TypeError, match="degrees"):
        get_cfg(overrides={"degrees": None})
    with pytest.raises(ValueError, match="cls_pw"):
        get_cfg(overrides={"cls_pw": 10})
    for key, value in (
        ("split", []),
        ("split", -0.0),
        ("optimizer", []),
        ("copy_paste_mode", {}),
        ("optimizer", None),
        ("split", None),
        ("copy_paste_mode", None),
    ):
        with pytest.raises((TypeError, ValueError), match=key):
            get_cfg(overrides={key: value})
    assert get_cfg(overrides={"auto_augment": None}).auto_augment is None


def skip_rpi_semantic():
    """Skip semantic segmentation tests on Raspberry Pi due to memory constraints."""
    if IS_RASPBERRYPI:
        pytest.skip("Semantic segmentation tests are skipped on Raspberry Pi due to memory constraints.")


def test_select_device(monkeypatch):
    """The same device string must resolve to the same GPU on every call, and the environment is never mutated."""
    from ultralytics.utils import torch_utils

    set_calls = []
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch_utils.torch.cuda, "set_device", set_calls.append)
    monkeypatch.setattr(torch_utils, "get_gpu_info", lambda i: f"Mock GPU {i}, 1MiB")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert str(torch_utils.select_device("", verbose=False)) == "cuda:0"
    assert not set_calls  # default '' request must never move the current device, e.g. diagnostics like check_yolo()
    for _ in range(2):  # repeated calls are idempotent, e.g. Trainer.__init__ then final_eval, or predict() twice
        assert str(torch_utils.select_device("1", verbose=False)) == "cuda:1"
        with pytest.raises(ValueError):
            torch_utils.select_device("3", verbose=False)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is None  # CUDA_VISIBLE_DEVICES never written
    assert set_calls == [1, 1]  # explicit single-GPU requests set the default device for indexless 'cuda' operations
    assert str(torch_utils.select_device("0,1", verbose=False)) == "cuda:0"
    assert set_calls == [1, 1]  # multi-GPU requests never move the current device; DDP ranks pin theirs in _setup_ddp
    monkeypatch.setattr(torch_utils.torch.cuda, "current_device", lambda: 1)
    assert str(torch_utils.select_device("", verbose=False)) == "cuda:1"  # default '' resolves to the current device
    assert str(torch_utils.select_device(torch.device("cuda", 1), verbose=False)) == "cuda:1"
    with pytest.raises(ValueError):  # torch.device inputs are validated like strings, no raw CUDA errors
        torch_utils.select_device(torch.device("cuda", 3), verbose=False)
    set_calls.clear()
    assert str(torch_utils.select_device(torch.device("cuda"), verbose=False)) == "cuda:1"
    assert not set_calls  # indexless torch.device('cuda') means the current device and never moves it
    assert torch_utils.parse_device([0, 1]) == "0,1"
    assert torch_utils.parse_device("00,01") == "0,1"  # leading zeros stripped for valid torch device strings
    assert torch_utils.parse_device(torch.device("cuda")) == ""  # indexless 'cuda' stays the '' default request
    # Physical GPU ids under an external CUDA_VISIBLE_DEVICES restriction translate to torch indices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert str(torch_utils.select_device("3", verbose=False)) == "cuda:1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 1)
    assert str(torch_utils.select_device("3", verbose=False)) == "cuda:0"  # e.g. pods launched with CVD preset
    assert torch_utils.parse_device(torch_utils.parse_device("3")) == "0"  # idempotent: trainer + select_device parse
    # '-1' idle-GPU auto-selection searches only externally visible GPUs and translates physical ids to torch indices
    from ultralytics.utils import autodevice

    monkeypatch.setattr(autodevice.GPUInfo, "__init__", lambda self: self.__dict__.update(nvml_available=False))
    monkeypatch.setattr(
        autodevice.GPUInfo,
        "select_idle_gpu",
        lambda self, count=1, indices=None, **kw: [i for i in (0, 1, 3) if i in indices][:count],
    )
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert torch_utils.parse_device("-1") == "0"  # idle physical GPU 1 is torch index 0 under CVD='1,3'; 0 is hidden
    assert torch_utils.parse_device("1") == "1"  # in-range ids are torch indices, so repeated parses are stable
    assert torch_utils.parse_device("-1,3") == "0,1"  # mixed: idle physical GPU 1 + physical GPU 3 as torch indices
    assert torch_utils.parse_device("0,1") == "0,1"  # already-translated outputs re-parse unchanged (idempotent)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,9,1")  # malformed: CUDA stops at the invalid id 9, 2 usable GPUs
    assert torch_utils.parse_device("9") == "9"  # unusable id is not translated, so select_device rejects it
    assert torch_utils.parse_device("2") == "1"  # physical GPU 2 is torch index 1 of the usable prefix
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "01,03")  # leading zeros are valid for CUDA's atoi-style parsing
    assert torch_utils.parse_device("3") == "1"  # visible ids normalize like requested ids
    assert torch_utils.parse_device("-1") == "0"  # idle physical GPU 1 found via normalized visible ids
>>>>>>> origin/main


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment


def test_model_methods():
    """Test various methods and properties of the YOLO model to ensure correct functionality."""
    model = YOLO(MODEL)

    # Model methods
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # Model properties
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


<<<<<<< HEAD
def test_save_lora_only_uses_live_trainer_model(tmp_path):
    """Test that LoRA adapter export uses the live trainer model when available."""
    model = make_stub_yolo()
    model.trainer.model = mock.Mock(lora_enabled=True)

    with mock.patch("ultralytics.utils.lora.save_lora_adapters", return_value=True) as save_mock:
        assert model.save_lora_only(tmp_path / "lora_adapter")
        save_mock.assert_called_once_with(model.trainer.model, tmp_path / "lora_adapter")


def test_load_lora_delegates_to_adapter_loader(tmp_path):
    """Test that LoRA adapter loading is delegated to the shared utility."""
    model = make_stub_yolo()

    with mock.patch("ultralytics.utils.lora.load_lora_adapters", return_value=True) as load_mock:
        assert model.load_lora(tmp_path / "lora_adapter")
        load_mock.assert_called_once_with(model.model, tmp_path / "lora_adapter", merge=False, trainable=False)


def test_load_lora_supports_trainable_reload(tmp_path):
    """Test that load_lora can request a trainable PEFT reload for continued fine-tuning."""
    model = make_stub_yolo()

    with mock.patch("ultralytics.utils.lora.load_lora_adapters", return_value=True) as load_mock:
        assert model.load_lora(tmp_path / "lora_adapter", trainable=True)
        load_mock.assert_called_once_with(model.model, tmp_path / "lora_adapter", merge=False, trainable=True)


def test_train_preserves_active_lora_model():
    """Test that training keeps an already-loaded LoRA model instead of rebuilding it."""
    model = make_stub_yolo()
    model.model.lora_enabled = True
    model.model.yaml = {}
    model.overrides = {"model": "yolo11n.pt"}
    model.ckpt = None
    model.task = "detect"
    model.session = None
    model.callbacks = {}
    model._has_active_lora_model = mock.Mock(return_value=True)

    trainer = mock.Mock()
    trainer.get_model = mock.Mock()
    trainer.model = None
    trainer.train.side_effect = RuntimeError("stop")

    with mock.patch.object(model, "_smart_load", return_value=lambda overrides, _callbacks: trainer), \
         mock.patch("ultralytics.utils.checks.check_pip_update_available"), \
         pytest.raises(RuntimeError, match="stop"):
        model.train(data="coco8.yaml", epochs=1)

    assert trainer.get_model.call_count == 0
    assert trainer.model is model.model


def test_save_lora_only_supports_fallback_backend(tmp_path):
    """Test that fallback-backed adapters still use the shared save utility."""
    model = make_stub_yolo()
    model.trainer.model = mock.Mock(lora_enabled=True, lora_backend="fallback")

    with mock.patch("ultralytics.utils.lora.save_lora_adapters", return_value=True) as save_mock:
        assert model.save_lora_only(tmp_path / "fallback_adapter")
        save_mock.assert_called_once_with(model.trainer.model, tmp_path / "fallback_adapter")


def test_merge_lora_weights_clears_peft_runtime_state():
    """Test that PEFT merges remove stale LoRA runtime metadata."""
    from ultralytics.utils import lora as lora_utils

    class DummyModel:
        pass

    merged_base = object()
    model = DummyModel()
    model.model = mock.Mock()
    model.model.merge_and_unload.return_value = merged_base
    model.lora_enabled = True
    model.lora_backend = "peft"
    model.lora_variant = "ia3"
    model.lora_include_head = False
    model.lora_freeze_bn = True
    model.lora_target_modules = ["0.conv"]
    model.lora_target_audit = {"selected_count": 1}
    model.lora_runtime_metadata = {"effective_backend": "peft"}
    model.use_gradient_checkpointing = True

    with mock.patch.object(lora_utils, "_find_original_model_class", return_value=DummyModel):
        assert lora_utils.merge_lora_weights(model)

    assert model.model is merged_base
    for attr in (
        "lora_enabled",
        "lora_backend",
        "lora_variant",
        "lora_include_head",
        "lora_freeze_bn",
        "lora_target_modules",
        "lora_target_audit",
        "lora_runtime_metadata",
        "use_gradient_checkpointing",
    ):
        assert not hasattr(model, attr)


def test_load_fallback_adapter_state_restores_runtime_metadata(tmp_path):
    """Test that fallback adapter loading restores saved top-level runtime metadata."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    wrapped = lora_utils.ManualLoRAConv(torch.nn.Conv2d(3, 4, kernel_size=3, padding=1), r=2, alpha=4, dropout=0.1)
    torch.save(
        {
            "modules": {"conv": {"r": 2, "alpha": 4, "dropout": 0.1}},
            "state": {
                "conv": {
                    "lora_A": wrapped.lora_A.detach().clone(),
                    "lora_B": wrapped.lora_B.detach().clone(),
                }
            },
        },
        tmp_path / "fallback_adapter.pt",
    )

    payload = {
        "backend": "fallback",
        "variant": "lora",
        "weight_file": "fallback_adapter.pt",
        "include_head": True,
        "freeze_bn": True,
        "target_modules": ["conv"],
        "target_audit": {"selected_count": 1},
        "runtime_metadata": {"effective_backend": "fallback"},
    }

    loaded = lora_utils._load_fallback_adapter_state(model, tmp_path, payload)

    assert getattr(loaded, "lora_enabled", False) is True
    assert loaded.lora_backend == "fallback"
    assert loaded.lora_variant == "lora"
    assert loaded.lora_include_head is True
    assert loaded.lora_freeze_bn is True
    assert loaded.lora_target_modules == ["conv"]
    assert loaded.lora_target_audit["selected_count"] == 1
    assert loaded.lora_runtime_metadata == {"effective_backend": "fallback"}


def test_load_lora_force_replace_clears_stale_fallback_runtime_state(tmp_path):
    """Test that force-replacing fallback adapters clears stale top-level LoRA markers before reload."""
    from ultralytics.utils import lora as lora_utils

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "fallback_meta.json").write_text('{"backend":"fallback","weight_file":"fallback_adapter.pt"}')

    model = torch.nn.Module()
    model.model = torch.nn.Sequential()
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_include_head = True
    model.lora_freeze_bn = True
    model.lora_target_modules = ["old.conv"]
    model.lora_target_audit = {"stale": True}
    model.lora_runtime_metadata = {"stale": True}
    model.use_gradient_checkpointing = True

    def _fake_loader(current_model, path, payload):
        assert not hasattr(current_model, "lora_include_head")
        assert not hasattr(current_model, "lora_freeze_bn")
        assert not hasattr(current_model, "lora_target_modules")
        assert not hasattr(current_model, "lora_target_audit")
        assert not hasattr(current_model, "lora_runtime_metadata")
        assert not hasattr(current_model, "use_gradient_checkpointing")
        return current_model

    with mock.patch.object(lora_utils, "_load_fallback_adapter_state", side_effect=_fake_loader):
        assert lora_utils.load_lora_adapters(model, adapter_dir, force_replace=True)


def test_load_lora_adapters_restores_peft_runtime_metadata(tmp_path):
    """Test that PEFT adapter loading restores saved runtime metadata onto the model."""
    from ultralytics.utils import lora as lora_utils

    runtime_payload = {
        "backend": "peft",
        "variant": "ia3",
        "include_head": True,
        "freeze_bn": True,
        "target_modules": ["0.conv"],
        "target_audit": {"selected_count": 1},
        "runtime_metadata": {"effective_backend": "peft"},
    }
    (tmp_path / "runtime_metadata.json").write_text(json.dumps(runtime_payload))

    model = torch.nn.Module()
    model.model = object()

    class _DummyProxy(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 1))
            self.lora_A = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, base_model, path, is_trainable=False):
            return cls()

        def __len__(self):
            return len(self.layers)

        def __getitem__(self, idx):
            return self.layers[idx]

        def children(self):
            return self.layers.children()

    def _fake_wrap(current_model, config):
        current_model.lora_enabled = True
        return current_model

    with mock.patch.object(lora_utils, "PEFT_AVAILABLE", True), \
         mock.patch.object(lora_utils, "PeftModel", _DummyProxy), \
         mock.patch.object(lora_utils, "PeftProxy", _DummyProxy), \
         mock.patch.object(lora_utils, "_wrap_top_level_lora_model", side_effect=_fake_wrap):
        assert lora_utils.load_lora_adapters(model, tmp_path)

    assert getattr(model, "lora_enabled", False) is True
    assert model.lora_backend == "peft"
    assert model.lora_variant == "ia3"
    assert model.lora_include_head is True
    assert model.lora_freeze_bn is True
    assert model.lora_target_modules == ["0.conv"]
    assert model.lora_target_audit == {"selected_count": 1}
    assert model.lora_runtime_metadata == {"effective_backend": "peft"}


def test_load_lora_adapters_merge_true_calls_merge_for_peft(tmp_path):
    """Test that PEFT adapter loading delegates to merge_lora_weights when merge=True."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.model = object()

    class _DummyProxy(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 1))
            self.lora_A = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, base_model, path, is_trainable=False):
            return cls()

        def __len__(self):
            return len(self.layers)

        def __getitem__(self, idx):
            return self.layers[idx]

        def children(self):
            return self.layers.children()

    def _fake_wrap(current_model, config):
        current_model.lora_enabled = True
        return current_model

    with mock.patch.object(lora_utils, "PEFT_AVAILABLE", True), \
         mock.patch.object(lora_utils, "PeftModel", _DummyProxy), \
         mock.patch.object(lora_utils, "PeftProxy", _DummyProxy), \
         mock.patch.object(lora_utils, "_wrap_top_level_lora_model", side_effect=_fake_wrap), \
         mock.patch.object(lora_utils, "merge_lora_weights", return_value=True) as merge_mock:
        assert lora_utils.load_lora_adapters(model, tmp_path, merge=True)
        merge_mock.assert_called_once_with(model)


def test_load_lora_compatible_state_dict_restores_adapter_params():
    """Test that LoRA-aware checkpoint loading restores compatible adapter tensors."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.base = torch.nn.Linear(2, 2)
    model.lora_A = torch.nn.Parameter(torch.zeros(2, 2))

    source_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    stats = lora_utils.load_lora_compatible_state_dict(model, source_state, context="unit-test")

    assert stats["adapter_loaded"] == 1
    assert torch.equal(model.lora_A, torch.ones_like(model.lora_A))


def test_load_lora_compatible_state_dict_rejects_adapter_shape_mismatch():
    """Test that changed LoRA rank/targets fail with a clear compatibility error."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.zeros(2, 2))
    source_state = {"lora_A": torch.zeros(3, 2)}

    with pytest.raises(RuntimeError, match="adapter topology"):
        lora_utils.load_lora_compatible_state_dict(model, source_state, context="unit-test")


def test_build_lora_target_audit_counts_filters():
    """Test compact target audit accounting for user filtering and grouped-conv exclusions."""
    from ultralytics.utils import lora as lora_utils

    audit = lora_utils.build_lora_target_audit(
        valid_targets=["0.conv", "1.conv", "2.cv1"],
        selected_targets=["1.conv"],
        requested_targets=["conv"],
        incompatible_layers=["3.cv2"],
        peft_type="lora",
        rank=8,
    )

    assert audit["valid_count"] == 3
    assert audit["selected_count"] == 1
    assert audit["filtered_by_user_count"] == 2
    assert audit["incompatible_grouped_conv_count"] == 1


def test_merge_manual_lora_conv_preserves_forward_output():
    """Test that merging a fallback LoRA conv reproduces the wrapped forward numerically."""
    from ultralytics.utils.lora import ManualLoRAConv, _merge_manual_lora_conv

    torch.manual_seed(0)
    base_conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    wrapped = ManualLoRAConv(base_conv, r=2, alpha=4, dropout=0.0)
    wrapped.lora_A.data.normal_(mean=0.0, std=0.2)
    wrapped.lora_B.data.normal_(mean=0.0, std=0.2)

    x = torch.randn(2, 3, 8, 8)
    expected = wrapped(x)

    merged = _merge_manual_lora_conv(wrapped)
    actual = merged(x)

    assert isinstance(merged, torch.nn.Conv2d)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_unfreeze_detection_head_only_unfreezes_head_params():
    """Test that LoRA head unfreeze only touches the detection head, not frozen backbone params."""
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.lora import _unfreeze_detection_head

    class _ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.detect = Detect(nc=2, ch=(8,))

    model = _ToyModel()
    for _, param in model.named_parameters():
        param.requires_grad = False

    unfrozen = _unfreeze_detection_head(model)
    head_params = {
        name for name, _ in model.named_parameters()
        if name == "detect" or name.startswith("detect.")
    }
    trainable_params = {name for name, param in model.named_parameters() if param.requires_grad}

    assert unfrozen > 0
    assert trainable_params
    assert trainable_params == head_params
    assert all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("stem."))
=======
def test_model_load_remaps_cls_head_by_names():
    """Test class-name remap is limited to closed-set class-logit heads."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.nn.tasks import DetectionModel, OBBModel, PoseModel, SegmentationModel, YOLOEModel

    src = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    tgt = DetectionModel("yolo26n.yaml", nc=2, verbose=False)
    src.names, tgt.names = {0: "cat", 1: "dog", 2: "car"}, {0: "dog", 1: "cat"}
    for seq in src.model[-1].cv3:
        seq[-1].bias.data.copy_(torch.tensor([10.0, 20.0, 30.0]))
    tgt.load(src, verbose=False)
    assert all(seq[-1].bias.tolist() == [20.0, 10.0] for seq in tgt.model[-1].cv3)

    src = DetectionModel("yolo26n.yaml", nc=6, verbose=False)
    tgt = DetectionModel("yolo26n.yaml", nc=6, verbose=False)
    src.names = {
        0: "airplane",
        1: "motorcycle",
        2: "couch",
        3: "tv",
        4: "dining table",
        5: "potted plant",
    }
    tgt.names = {
        0: "aeroplane",
        1: "motorbike",
        2: "sofa",
        3: "tvmonitor",
        4: "diningtable",
        5: "pottedplant",
    }
    for seq in src.model[-1].cv3:
        seq[-1].bias.data.copy_(torch.arange(6, dtype=seq[-1].bias.dtype))
    for seq in tgt.model[-1].cv3:
        seq[-1].bias.data.fill_(-1)
    tgt.load(src, verbose=False)
    assert all(seq[-1].bias.tolist() == list(range(6)) for seq in tgt.model[-1].cv3)

    src = YOLOEModel("yoloe-26n.yaml", nc=3, verbose=False)
    tgt = YOLOEModel("yoloe-26n.yaml", nc=2, verbose=False)
    src.names, tgt.names = {0: "cat", 1: "dog", 2: "car"}, {0: "dog", 1: "cat"}
    tgt.load(src, verbose=False)  # YOLOE cv3 outputs embeddings, not class rows

    names = {0: "dog", 1: "cat"}
    for trainer_cls, model in (
        (DetectionTrainer, DetectionModel("yolo26n.yaml", nc=2, verbose=False)),
        (SegmentationTrainer, SegmentationModel("yolo26n-seg.yaml", nc=2, verbose=False)),
        (PoseTrainer, PoseModel("yolo26n-pose.yaml", nc=2, data_kpt_shape=[17, 3], verbose=False)),
        (OBBTrainer, OBBModel("yolo26n-obb.yaml", nc=2, verbose=False)),
    ):
        trainer = object.__new__(trainer_cls)
        trainer.args = SimpleNamespace(cls_remap=True)
        trainer.data = {"names": names}
        assert trainer.set_model_names_for_load(model).names == names
>>>>>>> origin/main


def test_model_profile():
    """Test profiling of the YOLO model with `profile=True` to assess performance and resource usage."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # build model
    im = torch.randn(1, 3, 64, 64)  # requires min imgsz=64
    _ = model.predict(im, profile=True)


def test_predict_txt(tmp_path):
    """Test YOLO predictions with file, directory, and pattern sources listed in a text file."""
    file = tmp_path / "sources_multi_row.txt"
    with open(file, "w") as f:
        for src in SOURCES_LIST:
            f.write(f"{src}\n")
    results = YOLO(MODEL)(source=file, imgsz=32)
<<<<<<< HEAD
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images
=======
    assert len(results) == 7, f"Expected 7 results from source list, got {len(results)}"
>>>>>>> origin/main


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_multi_row(tmp_path):
    """Test YOLO predictions with sources listed in multiple rows of a CSV file."""
    file = tmp_path / "sources_multi_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"])
        writer.writerows([[src] for src in SOURCES_LIST])
    results = YOLO(MODEL)(source=file, imgsz=32)
<<<<<<< HEAD
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images
=======
    assert len(results) == 7, f"Expected 7 results from multi-row CSV, got {len(results)}"
>>>>>>> origin/main


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_single_row(tmp_path):
    """Test YOLO predictions with sources listed in a single row of a CSV file."""
    file = tmp_path / "sources_single_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCES_LIST)
    results = YOLO(MODEL)(source=file, imgsz=32)
<<<<<<< HEAD
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images
=======
    assert len(results) == 7, f"Expected 7 results from single-row CSV, got {len(results)}"
>>>>>>> origin/main


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """Test YOLO model predictions on various image input types and sources, including online images."""
<<<<<<< HEAD
=======
    if IS_RASPBERRYPI and model_name == "yolo26n-sem.pt":
        skip_rpi_semantic()
>>>>>>> origin/main
    channels = 1 if model_name == "yolo11n-grayscale.pt" else 3
    model = YOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE), flags=cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)  # uint8 NumPy array
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(torch.rand((2, channels, 32, 32)), imgsz=32)) == 2  # batch-size 2 Tensor, FP32 0.0-1.0 RGB order
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, channels).numpy().astype(np.uint8), imgsz=32)) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
<<<<<<< HEAD
        f"{ASSETS_URL}/zidane.jpg?token=123" if ONLINE else SOURCE,  # URI
=======
        "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/im/zidane.jpg?token=123" if ONLINE else SOURCE,  # URI
>>>>>>> origin/main
        im,  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, channels), dtype=np.uint8),  # numpy
    ]
    assert len(model(batch, imgsz=32, classes=0)) == len(batch)  # multiple sources in a batch


<<<<<<< HEAD
@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model prediction methods with 'visualize=True' to generate and display prediction visualizations."""
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


=======
@pytest.mark.parametrize("model_name", ["yolo26n.pt", "yolo11n.pt"])  # end2end and NMS-based models
def test_predict_classes_with_max_det(model_name):
    """Test that the classes filter applies before max_det truncation in both end2end and NMS-based models."""
    boxes = YOLO(WEIGHTS_DIR / model_name)(SOURCE, classes=[0], max_det=300, verbose=False)[0].boxes
    assert len(boxes) > 1  # bus.jpg contains multiple persons
    top1 = YOLO(WEIGHTS_DIR / model_name)(SOURCE, classes=[0], max_det=1, verbose=False)[0].boxes  # fresh model
    assert len(top1) == 1 and int(top1.cls) == 0
    assert float(top1.conf) == pytest.approx(float(boxes.conf.max()))  # best person kept, not an arbitrary one


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model prediction methods with 'visualize=True' to generate prediction visualizations."""
    if IS_RASPBERRYPI and model == "yolo26n-sem.pt":
        skip_rpi_semantic()
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_load_tensor_uint8():
    """Test that tensor normalization supports uint8 while preserving floating-point epsilon tolerance."""
    from ultralytics.data.loaders import LoadTensor

    loaded = LoadTensor(torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)).im0
    assert loaded.dtype == torch.float32 and loaded.max() == 1
    normalized = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    normalized[..., 0, 0] += torch.finfo(normalized.dtype).eps
    assert LoadTensor(normalized).im0.max() > 1


>>>>>>> origin/main
def test_predict_gray_and_4ch(tmp_path):
    """Test YOLO prediction on SOURCE converted to grayscale and 4-channel images with various filenames."""
    im = Image.open(SOURCE)

    source_grayscale = tmp_path / "grayscale.jpg"
    source_rgba = tmp_path / "4ch.png"
    source_non_utf = tmp_path / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = tmp_path / "image with spaces.jpg"

    im.convert("L").save(source_grayscale)  # grayscale
    im.convert("RGBA").save(source_rgba)  # 4-ch PNG with alpha
    im.save(source_non_utf)  # non-UTF characters in filename
    im.save(source_spaces)  # spaces in filename

    # Inference
    model = YOLO(MODEL)
    for f in source_rgba, source_grayscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
<<<<<<< HEAD
            assert len(results) == 1  # verify that an image was run
        f.unlink()  # cleanup


=======
            assert len(results) == 1, f"Expected 1 result for {f.name}, got {len(results)}"
        f.unlink()  # cleanup


def test_predict_ndarray_channels():
    """Test NumPy channel normalization for grayscale and color models."""
    from ultralytics.data.loaders import LoadPilAndNumpy

    model = YOLO(MODEL)  # default 3-channel model
    gray = np.asarray(Image.open(SOURCE).convert("L"))  # genuine 2D (H, W) uint8 array
    assert gray.ndim == 2, "Expected a 2D grayscale array for this test"
    assert len(model(source=gray, imgsz=32, verbose=False)) == 1  # 2D ndarray auto-expanded to 3 channels
    assert len(model(source=gray.astype("float64"), imgsz=32, verbose=False)) == 1  # non-OpenCV dtype also works
    for source_channels, model_channels in ((1, 3), (2, 1), (2, 3), (3, 1), (4, 1), (4, 3)):
        im = np.zeros((8, 8, source_channels), dtype=np.uint8)
        assert LoadPilAndNumpy(im, channels=model_channels).im0[0].shape == (8, 8, model_channels)


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_predict_all_image_formats():
    """Predict on the 12 image format extensions in COCO12-Formats (AVIF, BMP, DNG, HEIC, JP2, JPEG, JPG, MPO, PNG, TIF,
    TIFF, WebP).
    """
    # Download dataset if needed
    data = check_det_dataset("coco12-formats.yaml")
    dataset_path = Path(data["path"])

    # Collect all images from train and val
    expected = {"avif", "bmp", "dng", "heic", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
    images = [im for im in (dataset_path / "images" / "train").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    images += [im for im in (dataset_path / "images" / "val").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    assert len(images) == 12, f"Expected 12 images, found {len(images)}"

    # Verify all format extensions are represented
    extensions = {img.suffix.lower().lstrip(".") for img in images}
    assert extensions == expected, f"Missing formats: {expected - extensions}"

    # Run inference on all images
    model = YOLO(MODEL)
    results = model(images, imgsz=32)
    assert len(results) == 12, f"Expected 12 results, got {len(results)}"


>>>>>>> origin/main
@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(is_github_action_running(), reason="No auth https://github.com/JuanBindez/pytubefix/issues/166")
def test_youtube():
    """Test YOLO model on a YouTube video stream, handling potential network-related errors."""
    model = YOLO(MODEL)
    try:
<<<<<<< HEAD
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
=======
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=32, save=True)
>>>>>>> origin/main
    # Handle internet connection errors and 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.error(f"YouTube Test Error: {e}")


<<<<<<< HEAD
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.parametrize("model", MODELS)
def test_track_stream(model, tmp_path):
    """Test streaming tracking on a short 10 frame video using ByteTrack tracker and different GMC methods.

    Note imgsz=160 required for tracking for higher confidence and better matches.
    """
    if model == "yolo11n-cls.pt":  # classification model not supported for tracking
        return
    video_url = f"{ASSETS_URL}/decelera_portrait_min.mov"
    model = YOLO(model)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # test frame saving also

    # Test Global Motion Compensation (GMC) methods and ReID
    for gmc, reidm in zip(["orb", "sift", "ecc"], ["auto", "auto", "yolo11n-cls.pt"]):
=======
def test_track_second_association_indices():
    """Low-confidence detections matched in second association keep full detection-set indices."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    args = IterableSimpleNamespace(**{**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml"), "fuse_score": False})
    tracker = BYTETracker(args)
    boxes = [[10, 10, 50, 50], [200, 200, 260, 260], [400, 400, 480, 480]]
    for confs in ([0.9, 0.9, 0.9], [0.9, 0.9, 0.2]):  # third detection drops to low confidence on frame 2
        data = torch.tensor([[*b, c, 0] for b, c in zip(boxes, confs)], dtype=torch.float32)
        tracks = tracker.update(Boxes(data, (640, 640)))
    low = tracks[np.isclose(tracks[:, 5], 0.2)]  # columns are [x1, y1, x2, y2, id, score, cls, idx]
    assert len(low) == 1 and int(low[0, -1]) == 2, f"second-association idx not preserved:\n{tracks}"


@pytest.mark.parametrize("tracker_type", ["bytetrack", "fasttrack"])
def test_track_second_association_low_conf_keeps_id(tracker_type):
    """Low-confidence detection is recovered by the second association under the default fuse_score=True."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    args = IterableSimpleNamespace(**YAML.load(ROOT / f"cfg/trackers/{tracker_type}.yaml"))  # default fuse_score=True
    tracker = TRACKER_MAP[tracker_type](args)
    box = [100, 100, 200, 200]  # same box on both frames, so IoU is 1.0
    # frame 1: high score starts the track; frame 2: score drops into the low band (track_low_thresh < 0.15 < track_high_thresh)
    frame1 = tracker.update(Boxes(torch.tensor([[*box, 0.9, 0]], dtype=torch.float32), (640, 640)))
    frame2 = tracker.update(Boxes(torch.tensor([[*box, 0.15, 0]], dtype=torch.float32), (640, 640)))
    assert len(frame1) == 1, f"expected one track on frame 1:\n{frame1}"
    tid = int(frame1[0, 4])
    # the low-score box must be kept and mapped to the same id via the second association
    assert len(frame2) == 1, f"low-confidence detection lost by second association:\n{frame2}"
    assert int(frame2[0, 4]) == tid, f"id switched on low-confidence frame: {tid} -> {int(frame2[0, 4])}\n{frame2}"


@pytest.mark.parametrize("tracker_type", ["botsort", "deepocsort", "tracktrack"])
def test_track_reid_auto_user_detections(tracker_type):
    """Native ReID (model='auto') must degrade to motion-only with user-supplied detections, not encode the raw frame."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = {**YAML.load(ROOT / f"cfg/trackers/{tracker_type}.yaml"), "with_reid": True, "model": "auto"}
    tracker = TRACKER_MAP[tracker_type](IterableSimpleNamespace(**cfg))
    img = np.full((640, 640, 3), 128, dtype=np.uint8)  # nonzero so bogus frame-derived features would not be dropped
    data = torch.tensor([[10, 10, 50, 50, 0.9, 0], [200, 200, 260, 260, 0.9, 0]], dtype=torch.float32)
    for _ in range(3):  # frame 2 used to crash in embedding_distance after storing image rows as track features
        tracks = tracker.update(Boxes(data, (640, 640)), img)
    assert len(tracks) == 2, f"native-ReID tracker must keep tracking without feats:\n{tracks}"


def test_reid_invalid_crops():
    """Test ReID skips out-of-bounds detection crops while preserving feature alignment."""
    from types import SimpleNamespace

    from ultralytics.trackers.utils.reid import ReID

    encoder = ReID.__new__(ReID)
    encoder.is_pt = True
    encoder.model = SimpleNamespace(predictor=lambda crops: [torch.ones(4) for _ in crops])
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    feats = encoder(img, np.array([[30, 30, 40, 40], [1100, 1100, 200, 200]], dtype=np.float32))
    assert feats[0] is not None and feats[1] is None


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.parametrize("model", MODELS)
def test_track_stream(model, tmp_path):
    """Test streaming tracking on a short video with all built-in trackers and various GMC/ReID configurations.

    Note imgsz=160 required for tracking for higher confidence and better matches.
    """
    if model in {"yolo26n-cls.pt", "yolo26n-sem.pt"}:  # classification and semantic segmentation not supported
        return
    from ultralytics.trackers.track import TRACKER_MAP

    video_url = f"{ASSETS_URL}/decelera_portrait_min.mov"
    model = YOLO(model)

    # Default end-to-end run for all built-in trackers
    for tracker_type in TRACKER_MAP:
        kwargs = {"save_frames": True} if tracker_type == "botsort" else {}
        model.track(video_url, imgsz=160, tracker=f"{tracker_type}.yaml", **kwargs)

    # Test Global Motion Compensation (GMC) methods and ReID on botsort
    for gmc, reidm in zip(["orb", "sift", "ecc"], ["auto", "auto", "yolo26n-cls.pt"]):
>>>>>>> origin/main
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / f"botsort-{gmc}.yaml"
        YAML.save(custom_yaml, {**default_args, "gmc_method": gmc, "with_reid": True, "model": reidm})
        model.track(video_url, imgsz=160, tracker=custom_yaml)

<<<<<<< HEAD
=======
    # Test ONNX ReID encoder auto-download
    if model == "yolo26n.pt":
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / "botsort-reid-onnx.yaml"
        YAML.save(custom_yaml, {**default_args, "with_reid": True, "model": "yolo26n-reid.onnx"})
        model.track(video_url, imgsz=160, tracker=custom_yaml)

>>>>>>> origin/main

@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_val(task: str, weight: str, data: str) -> None:
    """Test the validation mode of the YOLO model."""
<<<<<<< HEAD
=======
    if IS_RASPBERRYPI and task == "semantic":
        skip_rpi_semantic()
>>>>>>> origin/main
    model = YOLO(weight)
    for plots in {True, False}:  # Test both cases i.e. plots=True and plots=False
        metrics = model.val(data=data, imgsz=32, plots=plots)
        metrics.to_df()
        metrics.to_csv()
        metrics.to_json()
        # Tests for confusion matrix export
        metrics.confusion_matrix.to_df()
        metrics.confusion_matrix.to_csv()
        metrics.confusion_matrix.to_json()
<<<<<<< HEAD


@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_scratch():
    """Test training the YOLO model from scratch using the provided configuration."""
    model = YOLO(CFG)
    model.train(data="coco8.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
=======
        cm = metrics.confusion_matrix
        expected = cm.nc if task in {"classify", "semantic"} else cm.nc + 1  # detection-style tasks include background
        assert cm.matrix.shape == (expected, expected), f"{task} confusion matrix is {cm.matrix.shape}"
        assert len(cm.tp_fp()[0]) == cm.nc  # per-class TP/FP never include background


def test_val_save_txt_pose(tmp_path):
    """Test that pose keypoints saved by val(save_txt=True) and val(save_json=True) are in the original image space."""
    model = YOLO(WEIGHTS_DIR / "yolo26n-pose.pt")
    # imgsz=640 (not the imgsz=32 used elsewhere): coco8-pose images are non-square, so the letterbox offset is only
    # large enough to push mis-scaled keypoints outside [0, 1] at full resolution; at small imgsz they would stay in
    # range and hide the regression. save_json=True also exercises pred_to_json, the other consumer of the scaled key.
    metrics = model.val(
        data="coco8-pose.yaml", imgsz=640, conf=0.25, save_txt=True, save_json=True, project=tmp_path, name="val"
    )
    txt_files = list((Path(metrics.save_dir) / "labels").glob("*.txt"))
    assert txt_files, "val(save_txt=True) saved no label files"
    assert (Path(metrics.save_dir) / "predictions.json").exists(), "val(save_json=True) saved no predictions.json"
    for txt_file in txt_files:
        for line in txt_file.read_text().splitlines():
            values = [float(v) for v in line.split()]
            x, y, w, h = values[1:5]  # normalized xywh box
            kpts = torch.tensor(values[5:]).view(-1, 3)  # (17, 3) of normalized (x, y, conf) keypoints
            assert ((kpts[:, :2] >= 0) & (kpts[:, :2] <= 1)).all(), f"keypoints not in [0, 1] in {txt_file.name}"
            # Keypoints scaled into the wrong (letterbox) space also land off the person, so check that visible
            # keypoints cluster on the box; the 0.05 margin allows joints (wrists, ankles) just outside a tight box.
            visible = kpts[kpts[:, 2] > 0.5, :2]
            if len(visible):
                cx, cy = visible.mean(0)
                assert abs(cx - x) < w / 2 + 0.05 and abs(cy - y) < h / 2 + 0.05, "keypoints misaligned with box"


def test_pose_metrics_curves():
    """Test that pose curve labels contain four unique box and pose series."""
    from ultralytics.utils.metrics import PoseMetrics

    curves = PoseMetrics().curves
    assert len(curves) == len(set(curves)) == 8


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_multi():
    """Test fine-tuning a base model across a dataset collection, which triggers MultiTrainer for list/tuple data."""
    model = YOLO(MODEL)
    results = model.train(data=["coco8.yaml", "coco8.yaml"], epochs=1, imgsz=32)
    assert isinstance(results, dict) and len(results) == 2  # one entry per run (coco8, coco8-2), no duplicate collapse
    assert all(m and "fitness" in m for m in results.values())  # checkpoint train metrics per run
    assert len(model.trainer.trainers) == 2  # both list entries fine-tuned in series
    sweep_dir = model.trainer.save_dir
    assert sweep_dir.name.startswith("multitrain")  # all runs grouped under one sweep directory
    assert (sweep_dir / "multitrain_results.json").exists()  # results JSON for post-processing
    assert (sweep_dir / "multitrain_results.png").exists()  # cross-dataset results plot


def test_normalize_platform_uri():
    """Test Platform web URLs are rewritten to ul:// URIs so datasets/models load directly from a pasted URL."""
    from ultralytics.utils.checks import normalize_platform_uri

    base = "https://platform.ultralytics.com/glenn-jocher"
    assert normalize_platform_uri(f"{base}/datasets/coco8") == "ul://glenn-jocher/datasets/coco8"
    assert normalize_platform_uri(f"{base}/project/model/") == "ul://glenn-jocher/project/model"
    assert normalize_platform_uri("coco8.yaml") == "coco8.yaml"  # non-Platform inputs unchanged


def test_convert_signed_ndjson(monkeypatch):
    """Test signed NDJSON URLs are converted before dataset YAML validation."""
    from ultralytics.data import converter, utils

    captured = []

    async def convert(path):
        captured.append(path)
        return "dataset.ndjson.yaml"

    monkeypatch.setattr(converter, "convert_ndjson_to_yolo", convert)
    url = "https://storage.googleapis.com/bucket/dataset-v1.ndjson?X-Goog-Signature=abc"
    assert utils.convert_ndjson_to_yolo_if_needed(url) == "dataset.ndjson.yaml"
    assert captured == [url]


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_ndjson_conversion_concurrency_and_resume(monkeypatch, tmp_path, task):
    """Test concurrent conversions share work and interrupted conversions resume before publishing completion."""
    import asyncio
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import aiohttp

    from ultralytics.data import converter

    counts, failures, conversions, count_lock = {}, set(), 0, threading.Lock()

    class Response:
        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        async def read(self):
            await asyncio.sleep(0.01)
            with count_lock:
                counts[self.url] = counts.get(self.url, 0) + 1
                fail = self.url in failures
                failures.discard(self.url)
            if fail:
                raise OSError("interrupted")
            return b"image"

    class Session:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def get(self, url, **_):
            return Response(url)

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    original_convert = converter._convert_ndjson_to_yolo

    async def track_conversion(*args):
        nonlocal conversions
        with count_lock:
            conversions += 1
        return await original_convert(*args)

    monkeypatch.setattr(converter, "_convert_ndjson_to_yolo", track_conversion)
    annotations = {"classification": [7]} if task == "classify" else {"boxes": [[0, 0.5, 0.5, 1, 1]]}

    def write_ndjson(name):
        path = tmp_path / f"{name}.ndjson"
        records = [
            {"type": "dataset", "task": task, "class_names": {"7": "item"}},
            {
                "file": "train.jpg",
                "url": f"https://example.com/{name}-train.jpg",
                "split": "train",
                "annotations": annotations,
            },
            {
                "file": "val.jpg",
                "url": f"https://example.com/{name}-val.jpg",
                "split": "val",
                "annotations": annotations,
            },
        ]
        path.write_text("\n".join(json.dumps(record) for record in records))
        return path

    concurrent = write_ndjson("concurrent")
    jobs = 2
    barrier = threading.Barrier(jobs)

    def convert(path):
        barrier.wait()
        return asyncio.run(converter.convert_ndjson_to_yolo(path, tmp_path))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(convert, [concurrent] * jobs))
    assert len(set(results)) == 1
    assert conversions == 1
    assert sum(counts.values()) == 2

    resume = write_ndjson("resume")
    failed_url = "https://example.com/resume-val.jpg"
    failures.add(failed_url)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(convert, resume) for _ in range(jobs)]
        errors, results = [], []
        for future in futures:
            try:
                results.append(future.result())
            except RuntimeError as error:
                errors.append(str(error))
    assert len(errors) == len(results) == 1 and "Downloaded 1/2 images" in errors[0]
    result = results[0]
    marker = result / ".ndjson.yaml" if task == "classify" else result
    assert YAML.load(marker)["complete"] is True
    assert counts["https://example.com/resume-train.jpg"] == 1
    assert counts[failed_url] == 2
    request_count = sum(counts.values())
    asyncio.run(converter.convert_ndjson_to_yolo(resume, tmp_path))
    assert conversions == 4
    assert sum(counts.values()) == request_count


def test_platform_job_transport(monkeypatch, tmp_path):
    """Test configurable Platform transport with an existing local checkpoint."""
    from types import SimpleNamespace

    from ultralytics import SETTINGS, cfg
    from ultralytics.utils.callbacks import platform

    monkeypatch.setattr(cfg, "TESTS_RUNNING", False)
    monkeypatch.setitem(SETTINGS, "runs_dir", str(tmp_path))
    args = SimpleNamespace(
        save_dir=None, project="user/project", task="detect", name="model", mode="train", exist_ok=True
    )
    assert cfg.get_save_dir(args) == tmp_path / "detect/user/project/model"

    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {"received": True}, raise_for_status=lambda: None)

    monkeypatch.setattr(platform, "requests", SimpleNamespace(post=post), raising=False)
    monkeypatch.setattr(platform, "_api_key", "api-key")
    monkeypatch.setattr(platform, "PLATFORM_API_URL", "https://example.test/api/webhooks")
    assert platform._send("epoch_end", {"epoch": 0}, "user/project", "model") == {"received": True}
    assert captured["url"] == "https://example.test/api/webhooks/training/metrics"
    assert captured["json"]["data"] == {"epoch": 0}
    assert captured["headers"] == {"Authorization": "Bearer api-key"}

    model = tmp_path / "models" / "best.pt"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    monkeypatch.setenv("PLATFORM_API_URL", "http://127.0.0.1:8765")
    assert platform._upload_model(model, "user/project", "model") == {
        "modelPath": str(model),
        "modelSize": 7,
    }


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_scratch():
    """Test training the YOLO model from scratch on 12 different image types in the COCO12-Formats dataset."""
    model = YOLO(CFG)
    model.train(data="coco12-formats.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
>>>>>>> origin/main
    model(SOURCE)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
<<<<<<< HEAD
def test_train_ndjson():
    """Test training the YOLO model using NDJSON format dataset."""
    model = YOLO(WEIGHTS_DIR / "yolo11n.pt")
=======
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_ndjson():
    """Test training the YOLO model using NDJSON format dataset."""
    model = YOLO(WEIGHTS_DIR / "yolo26n.pt")
>>>>>>> origin/main
    model.train(data=f"{ASSETS_URL}/coco8-ndjson.ndjson", epochs=1, imgsz=32)


@pytest.mark.parametrize("scls", [False, True])
<<<<<<< HEAD
def test_train_pretrained(scls):
    """Test training of the YOLO model starting from a pre-trained checkpoint."""
    model = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
=======
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_pretrained(scls):
    """Test training of the YOLO model starting from a pre-trained checkpoint."""
    model = YOLO(WEIGHTS_DIR / "yolo26n-seg.pt")
>>>>>>> origin/main
    model.train(
        data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0, single_cls=scls
    )
    model(SOURCE)


def test_all_model_yamls():
    """Test YOLO model creation for all available YAML configurations in the `cfg/models` directory."""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_11:
<<<<<<< HEAD
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # must be 640
=======
                _ = RTDETR(m.name)(SOURCE, imgsz=160)
>>>>>>> origin/main
        else:
            YOLO(m.name)


@pytest.mark.skipif(WINDOWS, reason="Windows slow CI export bug https://github.com/ultralytics/ultralytics/pull/16003")
<<<<<<< HEAD
def test_workflow():
    """Test the complete workflow including training, validation, prediction, and exporting."""
    model = YOLO(MODEL)
=======
def test_workflow(isolated_model):
    """Test the complete workflow including training, validation, prediction, and exporting."""
    model = YOLO(isolated_model)
>>>>>>> origin/main
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")  # WARNING: Windows slow CI export bug


def test_predict_callback_and_setup():
    """Test callback functionality during YOLO prediction setup and execution."""

    def on_predict_batch_end(predictor):
        """Callback function that handles operations at the end of a prediction batch."""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results is list[batch_size]

    model = YOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # access predictor properties
<<<<<<< HEAD
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
=======
    results = model.predict(dataset, stream=True, imgsz=32)  # source already setup
>>>>>>> origin/main
    for r, im0, bs in results:
        print("test_callback", im0.shape)
        print("test_callback", bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model: str, tmp_path):
    """Test YOLO model results processing and output in various formats."""
<<<<<<< HEAD
    im = f"{ASSETS_URL}/boats.jpg" if model == "yolo11n-obb.pt" else SOURCE
    results = YOLO(WEIGHTS_DIR / model)([im, im], imgsz=160)
    for r in results:
        assert len(r), f"'{model}' results should not be empty!"
=======
    if IS_RASPBERRYPI and model == "yolo26n-sem.pt":
        skip_rpi_semantic()
    im = "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/im/boats.jpg" if model == "yolo26n-obb.pt" else SOURCE
    is_semantic = "semantic" in model or "-sem" in model
    results = YOLO(WEIGHTS_DIR / model)([im, im], imgsz=32 if is_semantic else 160)
    for r in results:
        if is_semantic:
            assert r.semantic_mask is not None and r.semantic_mask.shape == r.orig_shape, (
                f"'{model}' semantic_mask should match the original image shape!"
            )
            assert r.semantic_mask.data.dtype == torch.uint8, f"'{model}' semantic_mask should use compact class IDs!"
        else:
            assert len(r), f"'{model}' results should not be empty!"
>>>>>>> origin/main
        r = r.cpu().numpy()
        print(r, len(r), r.path)  # print numpy attributes
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=tmp_path / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=tmp_path / "runs/tests/crops/")
        r.to_df(decimals=3)  # Align to_ methods: https://docs.ultralytics.com/modes/predict/#working-with-results
        r.to_csv()
        r.to_json(normalize=True)
        r.plot(pil=True, save=True, filename=tmp_path / "results_plot_save.jpg")
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)  # print after methods


<<<<<<< HEAD
def test_labels_and_crops():
    """Test output from prediction args for saving YOLO detection labels and crops."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    results = YOLO(WEIGHTS_DIR / "yolo11n.pt")(imgs, imgsz=160, save_txt=True, save_crop=True)
=======
def test_results_plot_without_boxes():
    """Test that plotting a masks-only Results (boxes=None) does not raise an AttributeError."""
    from ultralytics.engine.results import Results

    orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
    masks = torch.zeros((2, 640, 640), dtype=torch.float32)
    r = Results(orig_img, path="image.jpg", names={0: "a", 1: "b"}, masks=masks)
    assert r.boxes is None
    for color_mode in ("class", "instance"):
        assert r.plot(color_mode=color_mode).shape == orig_img.shape


def test_results_update_probs():
    """Test that Results.update(probs=...) wraps the tensor in Probs like the sibling attributes."""
    from ultralytics.engine.results import Probs, Results

    orig_img = np.zeros((32, 32, 3), dtype=np.uint8)
    r = Results(orig_img, path="image.jpg", names={i: f"c{i}" for i in range(5)}, probs=torch.rand(5))
    r.update(probs=torch.rand(5))
    assert isinstance(r.probs, Probs), "update(probs=) should wrap the tensor in Probs, not store a raw Tensor"
    assert r.verbose() and r.summary(), "verbose()/summary() raise AttributeError on a raw Tensor probs"


def test_labels_and_crops(tmp_path):
    """Test output from prediction args for saving YOLO detection labels and crops."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    model = YOLO(WEIGHTS_DIR / "yolo26n.pt")
    results = model(imgs, imgsz=160, save_txt=True, save_crop=True)
>>>>>>> origin/main
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
<<<<<<< HEAD
        # Check correct detections
        assert cls_idxs == ([0, 7, 0, 0] if r.path.endswith("bus.jpg") else [0, 0, 0])  # bus.jpg and zidane.jpg classes
        # Check label path
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists()
        # Check detections match label count
        assert len(r.boxes.data) == len([line for line in labels.read_text().splitlines() if line])
=======
        # Check that detections are made (at least 2 detections per image expected)
        assert len(cls_idxs) >= 2, f"Expected at least 2 detections, got {len(cls_idxs)}"
        # Check label path
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists(), f"Label file {labels} does not exist"
        # Check detections match label count
        label_count = len([line for line in labels.read_text().splitlines() if line])
        assert len(r.boxes.data) == label_count, f"Box count {len(r.boxes.data)} != label count {label_count}"
>>>>>>> origin/main
        # Check crops path and files
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # Crop directories match detections
<<<<<<< HEAD
        assert all(r.names.get(c) in {d.name for d in crop_dirs} for c in cls_idxs)
        # Same number of crops as detections
        assert len([f for f in crop_files if im_name in f.name]) == len(r.boxes.data)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_utils(tmp_path):
    """Test utility functions in ultralytics/data/utils.py, including dataset stats and auto-splitting."""
    from ultralytics.data.split import autosplit
    from ultralytics.data.utils import HUBDatasetStats
    from ultralytics.utils.downloads import zip_directory

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    for task in TASKS:
        file = Path(TASK2DATA[task]).with_suffix(".zip")  # i.e. coco8.zip
        download(f"https://github.com/ultralytics/hub/raw/main/example_datasets/{file}", unzip=False, dir=tmp_path)
        stats = HUBDatasetStats(tmp_path / file, task=task)
        stats.get_json(save=True)
        stats.process_images()

    autosplit(tmp_path / "coco8")
    zip_directory(tmp_path / "coco8/images/val")  # zip
=======
        crop_dir_names = {d.name for d in crop_dirs}
        assert all(r.names.get(c) in crop_dir_names for c in cls_idxs), (
            f"Crop dirs {crop_dir_names} don't match classes {cls_idxs}"
        )
        # Same number of crops as detections
        crop_count = len([f for f in crop_files if im_name in f.name])
        assert crop_count == len(r.boxes.data), f"Crop count {crop_count} != detection count {len(r.boxes.data)}"

    model(SOURCE, imgsz=160, save_crop=True, verbose=False, project=tmp_path, name="crop", exist_ok=True)
    assert any((tmp_path / "crop/crops").rglob("*.jpg")), "save_crop=True alone must write crop files"


def test_data_utils(tmp_path):
    """Test data utility functions including auto-splitting and zip archiving."""
    from ultralytics.data.split import autosplit
    from ultralytics.utils.downloads import zip_directory

    images_dir = tmp_path / "coco8/images/val"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images_dir / "test.jpg")

    autosplit(tmp_path / "coco8/images")
    assert any((tmp_path / "coco8").glob("autosplit_*.txt"))
    assert zip_directory(images_dir).is_file()
    with pytest.raises(ValueError, match="split"):
        check_cls_dataset("imagenet10", split="invalid")
    with pytest.raises(FileNotFoundError, match="'test:' images not found"):
        check_det_dataset("coco8.yaml", split="test")
    data_yaml = tmp_path / "coco8.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [item]\n")
    with pytest.raises(FileNotFoundError, match="images not found"):
        check_det_dataset(data_yaml, split="test")

    # polygons2masks_overlap must not overflow uint8 on the transient `masks + mask` sum (reaches 2 * i + 1):
    # with more than 128 overlapping instances every instance must keep a distinct index in the overlap mask
    from ultralytics.data.utils import polygons2masks_overlap

    segments = [
        np.array([[150 - s, 150 - s], [150 + s, 150 - s], [150 + s, 150 + s], [150 - s, 150 + s]], dtype=np.float32)
        for s in range(140, 10, -1)  # 130 concentric squares, all overlapping the center
    ]
    overlap, _ = polygons2masks_overlap((300, 300), segments)
    assert len(np.unique(overlap)) == len(segments) + 1  # background + 130 instances, no uint8 wraparound


def test_safe_download_unzips_local_path_archive(tmp_path):
    """Test safe_download() unzips local archive paths without treating them like remote URLs."""
    dataset_dir = tmp_path / "coco8 local"
    archive = tmp_path / "coco8 local.zip"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")

    with zipfile.ZipFile(archive, "w") as zf:
        for path in dataset_dir.rglob("*"):
            zf.write(path, arcname=path.relative_to(tmp_path))

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)
    expected_path = tmp_path / "datasets" / dataset_dir.name
    assert extracted == expected_path, f"Extracted path {extracted} != expected {expected_path}"
    assert (extracted / "data.yaml").is_file(), f"data.yaml not found in {extracted}"
    assert (extracted / "images" / "val").is_dir(), f"images/val not found in {extracted}"


def test_safe_download_skips_unsafe_archive_members(tmp_path):
    """Test safe_download() skips archive members that would extract outside the target directory."""
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../unsafe.txt", "bad")
        zf.writestr("safe/file.txt", "ok")

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)

    assert not (tmp_path / "unsafe.txt").exists()
    assert (extracted / "safe/file.txt").is_file()


def test_safe_download_skips_unsafe_tar_members(tmp_path):
    """Test safe_download() skips tar members that would extract outside the target directory."""
    source = tmp_path / "safe.txt"
    source.write_text("ok")
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source, arcname="../unsafe.txt")
        tar.add(source, arcname="safe.txt")

    extracted = safe_download(archive, dir=tmp_path / "datasets", unzip=True, progress=False)

    assert not (tmp_path / "unsafe.txt").exists()
    assert (extracted / "safe.txt").is_file()
>>>>>>> origin/main


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_converter(tmp_path):
    """Test dataset conversion functions from COCO to YOLO format and class mappings."""
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

<<<<<<< HEAD
    download(f"{ASSETS_URL}/instances_val2017.json", dir=tmp_path)
=======
    cached_file = DATASETS_DIR / "annotations" / "instances_val2017.json"
    if cached_file.exists():
        shutil.copy2(cached_file, tmp_path / cached_file.name)
    else:
        download(f"{ASSETS_URL}/instances_val2017.json", dir=tmp_path)
>>>>>>> origin/main
    convert_coco(
        labels_dir=tmp_path, save_dir=tmp_path / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True
    )
    coco80_to_coco91_class()


def test_data_annotator(tmp_path):
    """Test automatic annotation of data using detection and segmentation models."""
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(
        ASSETS,
<<<<<<< HEAD
        det_model=WEIGHTS_DIR / "yolo11n.pt",
=======
        det_model=WEIGHTS_DIR / "yolo26n.pt",
>>>>>>> origin/main
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=tmp_path / "auto_annotate_labels",
    )


def test_events():
    """Test event sending functionality."""
    from ultralytics.utils.events import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = "test"
    events(cfg)


def test_cfg_init():
    """Test configuration initialization utilities from the 'ultralytics.cfg' module."""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)

    # Test smart_value() with comprehensive cases
    # Test None conversion
    assert smart_value("none") is None
    assert smart_value("None") is None
    assert smart_value("NONE") is None

    # Test boolean conversion
    assert smart_value("true") is True
    assert smart_value("True") is True
    assert smart_value("TRUE") is True
    assert smart_value("false") is False
    assert smart_value("False") is False
    assert smart_value("FALSE") is False

    # Test numeric conversion (ast.literal_eval)
    assert smart_value("42") == 42
    assert smart_value("-42") == -42
    assert smart_value("3.14") == 3.14
    assert smart_value("-3.14") == -3.14
    assert smart_value("1e-3") == 0.001

    # Test list/tuple conversion (ast.literal_eval)
    assert smart_value("[1, 2, 3]") == [1, 2, 3]
    assert smart_value("(1, 2, 3)") == (1, 2, 3)
    assert smart_value("[640, 640]") == [640, 640]

    # Test dict conversion (ast.literal_eval)
    assert smart_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    # Test string fallback (when ast.literal_eval fails)
    assert smart_value("some_string") == "some_string"
    assert smart_value("path/to/file") == "path/to/file"
    assert smart_value("hello world") == "hello world"

    # Test that code injection is prevented (ast.literal_eval safety)
    # These should return strings, not execute code
    assert smart_value("__import__('os').system('ls')") == "__import__('os').system('ls')"
    assert smart_value("eval('1+1')") == "eval('1+1')"
    assert smart_value("exec('x=1')") == "exec('x=1')"

<<<<<<< HEAD
=======
    assert smart_value("zipfile.ZIP_DEFLATED") == zipfile.ZIP_DEFLATED
    assert smart_value("zipfile.Path") == "zipfile.Path"

>>>>>>> origin/main

def test_utils_init():
    """Test initialization utilities in the Ultralytics library."""
    from ultralytics.utils import get_ubuntu_version, is_github_action_running

    get_ubuntu_version()
    is_github_action_running()


<<<<<<< HEAD
def test_utils_checks():
    """Test various utility checks for filenames, git status, requirements, image sizes, and versions."""
    checks.check_yolov5u_filename("yolov5n.pt")
    checks.check_requirements("numpy")  # check requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    checks.check_imshow(warn=True)
    checks.check_version("ultralytics", "8.0.0")
=======
def test_utils_checks(monkeypatch):
    """Test various utility checks for filenames, requirements, image sizes, display capabilities, and versions."""

    def package_version(name):
        if name == "v2":
            return "1.0"
        raise checks.metadata.PackageNotFoundError

    checks.check_yolov5u_filename("yolov5n.pt")
    checks.check_requirements("numpy")  # check requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    with pytest.raises(ValueError):
        checks.check_imgsz("640x480")  # malformed imgsz string raises a helpful ValueError, not a raw SyntaxError
    checks.check_imshow(warn=True)
    checks.check_suffix("https://example.com/model.pt?token=abc", ".pt")
    checks.check_version("ultralytics", "8.0.0")
    # parse_version must pad to at least 3 components and keep all segments so any version pair compares correctly
    assert checks.parse_version("2") == (2, 0, 0)
    assert checks.parse_version("4.13.0.92") == (4, 13, 0, 92)
    assert checks.parse_version("2.0.1+cu118") == (2, 0, 1)  # numeric local/build suffixes are not release segments
    assert checks.parse_version("1.0.0rc1") == (1, 0, 0)
    assert checks.parse_version("v2.1") == (2, 1, 0)
    assert checks.parse_version("1.0rc1") == (1, 0, 0)  # documented non-PEP-440 tradeoff: pre-releases equal the final
    monkeypatch.setattr(checks.metadata, "version", package_version)
    assert not checks.check_version("v2", ">=2.0")  # installed version-shaped package keeps metadata precedence
    versions = ("v2.1-rc.1", "v2.1-beta1", "v2.1rev1", "v2.1-dev1", "v2.1+cu118")
    assert all(checks.check_version(v, ">=2.0") for v in versions)
    with pytest.raises(ModuleNotFoundError):
        checks.check_version("v2-missing", ">=2.0", hard=True)
    assert checks.check_version("10.3.0.30", ">=10.3.0,<10.4.0")  # Jetson TensorRT family pin
    assert checks.check_version("6.0", ">=6.0.0")  # 2-component current must satisfy 3-component requirement
    assert checks.check_version("2.1", "==2.1.0")
    assert checks.check_version("4.13.0.92", "!=4.13.0.90")  # 4-segment pins must not be truncated
    assert not checks.check_version("4.13.0.90", "!=4.13.0.90")
    assert checks.check_version("2.0.1", "<2.0.1.5")
>>>>>>> origin/main
    checks.print_args()


@pytest.mark.skipif(WINDOWS, reason="Windows profiling is extremely slow (cause unknown)")
def test_utils_benchmarks():
    """Benchmark model performance using 'ProfileModels' from 'ultralytics.utils.benchmarks'."""
    from ultralytics.utils.benchmarks import ProfileModels

<<<<<<< HEAD
    ProfileModels(["yolo11n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).run()
=======
    ProfileModels(["yolo26n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).run()
>>>>>>> origin/main


def test_utils_torchutils():
    """Test Torch utility functions including profiling and FLOP calculations."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile_ops, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile_ops(x, [m], n=3)
    get_flops_with_torch_profiler(m)
    time_sync()


<<<<<<< HEAD
=======
@pytest.mark.parametrize("nc", [1, 3])
def test_semantic_loss_all_ignore(nc):
    """SemanticSegmentationLoss must stay finite when the whole batch is ignore (255), e.g. unlabeled/void frames."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.utils.loss import SemanticSegmentationLoss

    model = SemanticSegmentationModel(cfg="yolo26-sem.yaml", nc=nc, verbose=False)
    model.args = get_cfg()
    loss_fn = SemanticSegmentationLoss(model)
    preds = torch.randn(1, nc, 64, 64, requires_grad=True)
    aux = torch.randn(1, nc, 32, 32, requires_grad=True)
    loss, items = loss_fn((preds, aux), {"semantic_mask": torch.full((1, 64, 64), 255, dtype=torch.long)})
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.backward()
    assert preds.grad is not None and aux.grad is not None


>>>>>>> origin/main
def test_utils_ops():
    """Test utility operations for coordinate transformations and normalizations."""
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
<<<<<<< HEAD
=======
        segment2box,
>>>>>>> origin/main
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

    make_divisible(17, torch.tensor([8]))

    boxes = torch.rand(10, 4)  # xywh
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    boxes = torch.rand(10, 5)  # xywhr for OBB
    boxes[:, 4] = torch.randn(10) * 30
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)

<<<<<<< HEAD

def test_utils_files(tmp_path):
    """Test file handling utilities including file age, date, and paths with spaces."""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path
=======
    # segment2box must not drop a polygon lying on the left image edge (all x == 0) to a zero box
    assert segment2box(np.array([[0, 100], [0, 150], [0, 200]]), 640, 640).tolist() == [0, 100, 0, 200]

    # segment2box must keep the visible extent when edge points shift out of frame after augmentation (issue #24935)
    seg = np.array([[550.0, 100.0], [690.0, 100.0], [690.0, 200.0], [550.0, 200.0]])
    assert segment2box(seg, 640, 640).tolist() == [550, 100, 640, 200]
    seg = np.array([[-10.0, 100.0], [650.0, 100.0], [650.0, 200.0], [-10.0, 200.0]])
    assert segment2box(seg, 640, 640).tolist() == [0, 100, 640, 200]
    assert segment2box(np.array([[100.0, 100.0], [200.0, 100.0], [700.0, -100.0]]), 640, 640).tolist() == [
        100,
        0,
        450,
        100,
    ]
    assert segment2box(np.array([[700.0, 100.0], [750.0, 150.0]]), 640, 640).tolist() == [0, 0, 0, 0]
    assert segment2box(np.empty((0, 2)), 640, 640).tolist() == [0, 0, 0, 0]
    seg = np.array([[-100.0, -100.0], [740.0, -100.0], [740.0, 740.0], [-100.0, 740.0]])  # surrounds the image
    assert segment2box(seg, 640, 640).tolist() == [0, 0, 640, 640]


def test_nms_end2end_classes_before_max_det():
    """The end-to-end NMS branch must filter classes before truncating to max_det, like the NMS-based branch."""
    from ultralytics.utils.nms import non_max_suppression

    # (2, 4, 6) end2end predictions sorted by descending confidence: [x1, y1, x2, y2, conf, cls]
    pred = torch.tensor(
        [
            [[0, 0, 9, 9, 0.9, 5], [1, 1, 9, 9, 0.8, 0], [2, 2, 9, 9, 0.7, 0], [3, 3, 9, 9, 0.6, 0]],
            [[0, 0, 9, 9, 0.9, 0], [1, 1, 9, 9, 0.8, 5], [2, 2, 9, 9, 0.7, 5], [3, 3, 9, 9, 0.6, 0]],
        ],
        dtype=torch.float32,
    )
    outputs, indices = non_max_suppression(pred, conf_thres=0.25, classes=[0], max_det=2, return_idxs=True)
    for out, idx, confs, expected in zip(outputs, indices, ([0.8, 0.7], [0.9, 0.6]), ([1, 2], [0, 3])):
        assert out.shape[0] == 2 and (out[:, 5] == 0).all()  # top-2 class-0 boxes kept, not truncated away
        assert torch.allclose(out[:, 4], torch.tensor(confs))
        assert idx.tolist() == expected
    out = non_max_suppression(pred, conf_thres=0.25, max_det=2)[0]  # without classes, top-2 overall unchanged
    assert torch.allclose(out[:, 4], torch.tensor([0.9, 0.8]))


def test_process_mask_empty():
    """Process_mask/process_mask_native/scale_masks must handle 0 detections without crashing."""
    from ultralytics.utils import ops

    protos, coeffs, bboxes = torch.rand(32, 160, 160), torch.zeros(0, 32), torch.zeros(0, 4)
    assert ops.process_mask(protos, coeffs, bboxes, (640, 640), upsample=True).shape == (0, 640, 640)
    assert ops.process_mask(protos, coeffs, bboxes, (640, 640)).shape == (0, 160, 160)  # prototype res when no upsample
    assert ops.process_mask_native(protos, coeffs, bboxes, (640, 640)).shape == (0, 640, 640)
    assert ops.scale_masks(torch.zeros(1, 0, 160, 160), (640, 640)).shape == (1, 0, 640, 640)


def test_utils_files(tmp_path):
    """Test file handling utilities including file age, date, and paths with spaces."""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, increment_path, spaces_in_path
>>>>>>> origin/main

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / "runs")

    path = tmp_path / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)

<<<<<<< HEAD
=======
    exp_dir = tmp_path / "runs" / "exp"
    exp_dir.mkdir(parents=True)
    assert increment_path(exp_dir) == tmp_path / "runs" / "exp-2"

    results_file = exp_dir / "results.txt"
    results_file.touch()
    assert increment_path(results_file) == exp_dir / "results-2.txt"

>>>>>>> origin/main

@pytest.mark.slow
def test_utils_patches_torch_save(tmp_path):
    """Test torch_save backoff when _torch_save raises RuntimeError."""
    from unittest.mock import MagicMock, patch

    from ultralytics.utils.patches import torch_save

    mock = MagicMock(side_effect=RuntimeError)

    with patch("ultralytics.utils.patches._torch_save", new=mock):
        with pytest.raises(RuntimeError):
            torch_save(torch.zeros(1), tmp_path / "test.pt")

    assert mock.call_count == 4, "torch_save was not attempted the expected number of times"


def test_nn_modules_conv():
    """Test Convolutional Neural Network modules including CBAM, Conv2, and ConvTranspose."""
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    DWConvTranspose2d(c1, c2)(x)
    ConvTranspose(c1, c2)(x)
    Focus(c1, c2)(x)
    CBAM(c1)(x)

    # Fuse ops
    m = Conv2(c1, c2)
    m.fuse_convs()
    m(x)


def test_nn_modules_block():
    """Test various neural network block modules."""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_hub():
    """Test Ultralytics HUB functionalities."""
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()
    logout()
    smart_request("GET", "https://github.com", progress=True)


@pytest.fixture
def image():
    """Load and return an image from a predefined source (OpenCV BGR)."""
    return cv2.imread(str(SOURCE))


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """Test classification transforms during training with various augmentations."""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)
    assert torch.is_tensor(transformed_image)
    assert transformed_image.dtype == torch.float32


@pytest.mark.slow
<<<<<<< HEAD
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance improvement."""
    YOLO("yolo11n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolo11n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
=======
@pytest.mark.skipif(IS_RASPBERRYPI or IS_JETSON, reason="Edge devices not intended for tuning")
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance improvement."""
    YOLO("yolo26n.pt").tune(
        data=["coco8.yaml", "coco8-grayscale.yaml"], plots=False, imgsz=32, epochs=1, iterations=2, device="cpu"
    )
    YOLO("yolo26n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolo26n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


@pytest.mark.slow
@pytest.mark.skipif(IS_RASPBERRYPI or IS_JETSON, reason="Edge devices not intended for tuning")
@pytest.mark.skipif(not ONLINE or not checks.IS_PYTHON_MINIMUM_3_10, reason="environment is offline")
@pytest.mark.skipif(not checks.check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_tune_ray():
    """Tune YOLO model for performance improvement."""
    YOLO("yolo26n-cls.pt").tune(
        data="imagenet10",
        use_ray=True,
        plots=False,
        imgsz=32,
        epochs=1,
        iterations=2,
        search_alg="random",
        device="cpu",
    )
>>>>>>> origin/main


def test_model_embeddings():
    """Test YOLO model embeddings extraction functionality."""
    model_detect = YOLO(MODEL)
<<<<<<< HEAD
    model_segment = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
=======
    model_segment = YOLO(WEIGHTS_DIR / "yolo26n-seg.pt")
>>>>>>> origin/main

    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)

<<<<<<< HEAD

=======
    model_classify = YOLO(WEIGHTS_DIR / "yolo26n-cls.pt")
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None
    assert isinstance(model_classify.embed(SOURCE, imgsz=32)[0], torch.Tensor)
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None
    assert isinstance(model_classify.predict(SOURCE, imgsz=32, embed=[-2])[0], torch.Tensor)
    assert model_classify.predict(SOURCE, imgsz=32)[0].probs is not None


def test_process_mask_native_chunked():
    """Chunked native upsampling is identical to upsampling all masks at once."""
    from ultralytics.utils import ops

    torch.manual_seed(0)
    protos, masks_in = torch.randn(32, 160, 160), torch.randn(70, 32)
    bboxes = torch.rand(70, 4) * 900 + 5  # fractional boxes exercise the crop edge handling
    bboxes[:, 2:] += bboxes[:, :2]
    out = ops.process_mask_native(protos, masks_in, bboxes, (1000, 1000))  # large shape forces multiple chunks
    ref = ops.scale_masks((masks_in @ protos.float().view(32, -1)).view(-1, 160, 160)[None], (1000, 1000))[0]
    ref = ops.crop_mask(ref, bboxes).gt_(0.0).byte()  # single-shot upsample-crop-threshold
    assert torch.equal(out, ref)


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for CLIP-based models")
>>>>>>> origin/main
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOWorld with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOWorld with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yolo_world():
    """Test YOLO world models with CLIP support."""
    model = YOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # no YOLO11n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # no YOLO11n-world model yet
    # Training from a pretrained model. Eval is included at the final stage of training.
    # Use dota8.yaml which has fewer categories to reduce the inference time of CLIP model
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # test WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # no YOLO11n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


<<<<<<< HEAD
=======
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for heavy CLIP-based models")
>>>>>>> origin/main
@pytest.mark.skipif(not TORCH_1_13, reason="YOLOE with CLIP requires torch>=1.13")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOE with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOE with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
<<<<<<< HEAD
def test_yoloe():
    """Test YOLOE models with MobileClip support."""
    # Predict
    # text-prompts
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    names = ["person", "bus"]
    model.set_classes(names, model.get_text_pe(names))
=======
def test_yoloe(tmp_path):
    """Test YOLOE models with MobileCLIP support."""
    # Predict
    # text-prompts
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    model.set_classes(["person", "bus"])
>>>>>>> origin/main
    model(SOURCE, conf=0.01)

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    # visual-prompts
    visuals = dict(
        bboxes=np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),
        cls=np.array([0, 1]),
    )
    model.predict(
        SOURCE,
        visual_prompts=visuals,
        predictor=YOLOEVPSegPredictor,
    )

    # Val
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    # text prompts
    model.val(data="coco128-seg.yaml", imgsz=32)
    # visual prompts
    model.val(data="coco128-seg.yaml", load_vp=True, imgsz=32)

    # Train, fine-tune
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOESegTrainerFromScratch

    model = YOLOE("yoloe-11s-seg.pt")
    model.train(
        data="coco128-seg.yaml",
        epochs=1,
        close_mosaic=1,
        trainer=YOLOEPESegTrainer,
        imgsz=32,
    )
    # Train, from scratch
<<<<<<< HEAD
    model = YOLOE("yoloe-11s-seg.yaml")
    model.train(
        data=dict(train=dict(yolo_data=["coco128-seg.yaml"]), val=dict(yolo_data=["coco128-seg.yaml"])),
        epochs=1,
        close_mosaic=1,
        trainer=YOLOESegTrainerFromScratch,
        imgsz=32,
    )
=======
    data_dict = dict(train=dict(yolo_data=["coco128-seg.yaml"]), val=dict(yolo_data=["coco128-seg.yaml"]))
    data_yaml = tmp_path / "yoloe-data.yaml"
    YAML.save(data=data_dict, file=data_yaml)
    for data in [data_dict, data_yaml]:
        model = YOLOE("yoloe-11s-seg.yaml")
        model.train(
            data=data,
            epochs=1,
            close_mosaic=1,
            trainer=YOLOESegTrainerFromScratch,
            imgsz=32,
        )
>>>>>>> origin/main

    # prompt-free
    # predict
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg-pf.pt")
    model.predict(SOURCE)
    # val
    model = YOLOE("yoloe-11s-seg.pt")  # or select yoloe-m/l-seg.pt for different sizes
    model.val(data="coco128-seg.yaml", imgsz=32)


<<<<<<< HEAD
=======
def test_yoloe_visual_prompt_verbose_false(capfd):
    """Verify that YOLOE visual prompting respects verbose=False."""
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")

    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    visuals = {
        "bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]),
        "cls": np.array([0]),
    }

    # Ignore any output produced while loading the model
    capfd.readouterr()

    model.predict(
        SOURCE,
        refer_image=SOURCE,
        visual_prompts=visuals,
        predictor=YOLOEVPSegPredictor,
        verbose=False,
    )

    captured = capfd.readouterr()
    output = captured.out + captured.err

    assert "Ultralytics" not in output


>>>>>>> origin/main
def test_yolov10():
    """Test YOLOv10 model training, validation, and prediction functionality."""
    model = YOLO("yolov10n.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    model(SOURCE)


def test_multichannel():
    """Test YOLO model multi-channel training, validation, and prediction functionality."""
<<<<<<< HEAD
    model = YOLO("yolo11n.pt")
=======
    model = YOLO("yolo26n.pt")
>>>>>>> origin/main
    model.train(data="coco8-multispectral.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8-multispectral.yaml")
    im = np.zeros((32, 32, 10), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    model.export(format="onnx")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_grayscale(task: str, model: str, data: str, tmp_path) -> None:
    """Test YOLO model grayscale training, validation, and prediction functionality."""
<<<<<<< HEAD
=======
    if IS_RASPBERRYPI and task == "semantic":
        skip_rpi_semantic()
>>>>>>> origin/main
    if task == "classify":  # not support grayscale classification yet
        return
    grayscale_data = tmp_path / f"{Path(data).stem}-grayscale.yaml"
    data = check_det_dataset(data)
    data["channels"] = 1  # add additional channels key for grayscale
    YAML.save(data=data, file=grayscale_data)
    # remove npy files in train/val splits if exists, might be created by previous tests
    for split in {"train", "val"}:
        for npy_file in (Path(data["path"]) / data[split]).glob("*.npy"):
            npy_file.unlink()

    model = YOLO(model)
<<<<<<< HEAD
    model.train(data=grayscale_data, epochs=1, imgsz=32, close_mosaic=1)
=======
    model.train(data=grayscale_data, epochs=1, imgsz=32, close_mosaic=1, cache="ram")
>>>>>>> origin/main
    model.val(data=grayscale_data)
    im = np.zeros((32, 32, 1), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    export_model = model.export(format="onnx")

    model = YOLO(export_model, task=task)
    model.predict(source=im, imgsz=32)
<<<<<<< HEAD
=======


def test_semantic_polygon_data():
    """Test YOLO semantic segmentation model with polygon data."""
    skip_rpi_semantic()
    model = YOLO("yolo26n-sem.pt")
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=32, close_mosaic=1)
    model.val(data="coco8-seg.yaml")
>>>>>>> origin/main

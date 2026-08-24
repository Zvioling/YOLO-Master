# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
<<<<<<< HEAD
from types import SimpleNamespace
from unittest import mock

import torch

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR, YAML


def test_func(*args):
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


def test_export():
    """Test model exporting functionality by adding a callback and verifying its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo11n.yaml").model)
    YOLO(f)(ASSETS)  # exported model inference


def test_save_trainer_args_yaml_persists_runtime_lora_total_step(tmp_path):
    """Test that saved trainer args reflect runtime-updated AdaLoRA total_step."""
    args = SimpleNamespace(augmentations=None, lora_total_step=7, save_dir=str(tmp_path))

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["lora_total_step"] == 7


def test_save_trainer_args_yaml_serializes_augmentations_repr(tmp_path):
    """Test that saved trainer args serialize augmentation objects via repr for resumability."""

    class _DummyAug:
        def __repr__(self):
            return "DummyAug(p=0.5)"

    args = SimpleNamespace(augmentations=[_DummyAug()], lora_total_step=1, save_dir=str(tmp_path))

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["augmentations"] == ["DummyAug(p=0.5)"]


def test_save_trainer_args_yaml_persists_effective_lora_backend(tmp_path):
    args = SimpleNamespace(
        augmentations=None,
        lora_total_step=7,
        requested_lora_backend="auto",
        effective_lora_backend="fallback",
        requested_lora_init_lora_weights="pissa",
        effective_lora_init_lora_weights="gaussian",
        save_dir=str(tmp_path),
    )

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["requested_lora_backend"] == "auto"
    assert saved["effective_lora_backend"] == "fallback"
    assert saved["requested_lora_init_lora_weights"] == "pissa"
    assert saved["effective_lora_init_lora_weights"] == "gaussian"


def test_update_args_with_lora_runtime_metadata_sets_requested_and_effective_fields():
    args = SimpleNamespace()
    model = SimpleNamespace(
        lora_runtime_metadata={
            "requested_backend": "auto",
            "effective_backend": "fallback",
            "requested_init_lora_weights": "pissa",
            "effective_init_lora_weights": "gaussian",
            "safety_profile": "rtdetr_lora",
            "safety_overrides": {"lora_lr_mult": {"from": 2.0, "to": 1.0}},
        }
    )

    trainer_module.update_args_with_lora_runtime_metadata(args, model)

    assert args.requested_lora_backend == "auto"
    assert args.effective_lora_backend == "fallback"
    assert args.requested_lora_init_lora_weights == "pissa"
    assert args.effective_lora_init_lora_weights == "gaussian"
    assert args.lora_safety_profile == "rtdetr_lora"
    assert args.lora_safety_overrides == {"lora_lr_mult": {"from": 2.0, "to": 1.0}}


def test_rtdetr_lora_safety_guard_mutates_training_args():
    from ultralytics.utils.lora import LoRAConfig, _apply_rtdetr_lora_safety

    class RTDETRDecoder(torch.nn.Module):
        pass

    args = SimpleNamespace(amp=True, lora_alpha_warmup=0, lora_lr_mult=2.0, lora_include_attention=False)
    config = LoRAConfig(
        r=16,
        alpha=32,
        lr_mult=2.0,
        alpha_warmup=0,
        include_attention=False,
        use_dora=True,
    )
    kwargs = {}

    changes = _apply_rtdetr_lora_safety(torch.nn.Sequential(RTDETRDecoder()), args, config, kwargs)

    assert args.amp is True
    assert args.lora_alpha_warmup == 3
    assert args.lora_lr_mult == 1.0
    assert args.lora_use_dora is False
    assert args.lora_include_attention is True
    assert config.alpha_warmup == 3
    assert config.lr_mult == 1.0
    assert config.use_dora is False
    assert config.include_attention is True
    assert "amp" not in kwargs
    assert kwargs["lora_alpha_warmup"] == 3
    assert kwargs["lora_lr_mult"] == 1.0
    assert kwargs["lora_use_dora"] is False
    assert kwargs["lora_include_attention"] is True
    assert changes == {
        "lora_alpha_warmup": {"from": 0, "to": 3},
        "lora_lr_mult": {"from": 2.0, "to": 1.0},
        "lora_include_attention": {"from": False, "to": True},
        "lora_use_dora": {"from": True, "to": False},
    }


def test_build_optimizer_separates_lora_params_with_lr_multiplier():
    class _AdapterBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.bias = torch.nn.Parameter(torch.zeros(2))
            self.lora_A = torch.nn.Parameter(torch.ones(1, 2))
            self.lora_B = torch.nn.Parameter(torch.ones(2, 1))

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _AdapterBlock()
            self.bn = torch.nn.BatchNorm1d(2)

    trainer = trainer_module.BaseTrainer.__new__(trainer_module.BaseTrainer)
    trainer.args = SimpleNamespace(lora_lr_mult=3.0)
    trainer.data = {}
    model = _TinyModel()

    optimizer = trainer_module.BaseTrainer.build_optimizer(
        trainer, model, name="SGD", lr=0.01, momentum=0.9, decay=1e-4
    )

    lora_params = {id(model.block.lora_A), id(model.block.lora_B)}
    lora_groups = [
        pg for pg in optimizer.param_groups
        if {id(param) for param in pg["params"]} == lora_params
    ]

    assert len(lora_groups) == 1
    assert lora_groups[0]["lr"] == pytest.approx(0.03)
    assert lora_groups[0]["initial_lr"] == pytest.approx(0.03)
    assert lora_groups[0]["weight_decay"] == 0.0


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_segment():
    """Test image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo11n-seg.yaml",
=======
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL, SOURCE, TASK_MODEL_DATA
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, detect, obb, pose, segment, semantic
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.torch_utils import unwrap_model


def test_func(*args, **kwargs):
    """Test function used as a callback stub to verify callback registration."""
    print("callback test passed")


def test_export(monkeypatch, tmp_path):
    """Test model exporting functionality by adding a callback and verifying its execution."""
    monkeypatch.chdir(tmp_path)
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "on_export_start callback not registered"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


@pytest.mark.parametrize(
    "trainer_cls,validator_cls,predictor_cls,data,model,weights",
    [
        (
            detect.DetectionTrainer,
            detect.DetectionValidator,
            detect.DetectionPredictor,
            "coco8.yaml",
            "yolo26n.yaml",
            MODEL,
        ),
        (
            segment.SegmentationTrainer,
            segment.SegmentationValidator,
            segment.SegmentationPredictor,
            "coco8-seg.yaml",
            "yolo26n-seg.yaml",
            WEIGHTS_DIR / "yolo26n-seg.pt",
        ),
        (
            classify.ClassificationTrainer,
            classify.ClassificationValidator,
            classify.ClassificationPredictor,
            "imagenet10",
            "yolo26n-cls.yaml",
            None,
        ),
        (obb.OBBTrainer, obb.OBBValidator, obb.OBBPredictor, "dota8.yaml", "yolo26n-obb.yaml", None),
        (pose.PoseTrainer, pose.PoseValidator, pose.PosePredictor, "coco8-pose.yaml", "yolo26n-pose.yaml", None),
        (
            semantic.SemanticSegmentationTrainer,
            semantic.SemanticSegmentationValidator,
            semantic.SemanticSegmentationPredictor,
            "cityscapes8.yaml",
            "yolo26n-sem.yaml",
            None,
        ),
    ],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_task(trainer_cls, validator_cls, predictor_cls, data, model, weights):
    """Test YOLO training, validation, and prediction for various tasks."""
    overrides = {
        "data": data,
        "model": model,
>>>>>>> origin/main
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }
<<<<<<< HEAD
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo11n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo11n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"
=======

    # Trainer
    trainer = trainer_cls(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "on_train_start callback not registered"
    trainer.train()

    # Validator
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data
    cfg.imgsz = 32
    val = validator_cls(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "on_val_start callback not registered"
    val(model=trainer.best)

    # Predictor
    pred = predictor_cls(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "on_predict_start callback not registered"

    # Determine model path for prediction
    model_path = weights if weights else trainer.best
    if model == "yolo26n.yaml":  # only for detection
        # Confirm there is no issue with sys.argv being empty
        with mock.patch.object(sys, "argv", []):
            result = pred(source=ASSETS, model=model_path)
            assert len(result) > 0, f"Predictor returned no results for {model}"
    else:
        result = pred(source=ASSETS, model=model_path)
        assert len(result) > 0, f"Predictor returned no results for {model}"

    # Test resume functionality
    with pytest.raises(AssertionError):
        trainer_cls(overrides={**overrides, "resume": trainer.last}).train()


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_resume_incomplete(task, weight, data, tmp_path):
    """Test training resumes from an incomplete checkpoint."""
    train_args = {
        "data": data,
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": task,
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO(weight)
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**train_args)
    last_path = model.trainer.last
    _, ckpt = load_checkpoint(last_path)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # Resume training using the checkpoint
    resume_model = YOLO(last_path)
    resume_model.train(resume=True, **train_args)
    assert resume_model.trainer.start_epoch == resume_model.trainer.epoch == 1, "resume test failed"


def test_distill_resume(tmp_path: Path):
    """Test knowledge distillation resumes from an incomplete checkpoint."""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "distill_model": WEIGHTS_DIR / "yolo26s.pt",
        "imgsz": 32,
        "multi_scale": 0.5,  # vary per-batch image size to exercise dynamic distillation score splitting
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill",
        "exist_ok": True,
    }

    # Train for one epoch then interrupt to produce a resumable checkpoint
    trainer = detect.DetectionTrainer(overrides=overrides)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"
    assert isinstance(ckpt["ema"], DistillationModel), "distillation EMA wraps the student model"
    assert ckpt["ema"].teacher_model is None, "teacher should be stripped from the EMA/checkpoint"
    assert ckpt["ema"].projector is not None, "the distillation projector should be persisted in the EMA checkpoint"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationModel), "resume should rebuild the DistillationModel"
    assert model.teacher_model is not None, "resume should rebuild the teacher from the distill_model path"
    assert trainer.start_epoch == trainer.epoch == 1, "resume test failed"


def test_distill_grayscale(tmp_path: Path):
    """Test knowledge distillation on a single-channel dataset (https://github.com/ultralytics/ultralytics/issues/25066)."""
    teacher = DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher}, teacher_path)
    student = DetectionModel("yolo26n.yaml", ch=1, nc=80, verbose=False)
    student.args = SimpleNamespace(imgsz=32, dis=1.0)
    model = DistillationModel(teacher_model=teacher_path, student_model=student)
    assert isinstance(model, DistillationModel)
    assert model.teacher_model.yaml["channels"] == 1


@pytest.mark.parametrize(
    "ckpt",
    [
        {"model": OrderedDict([("a", torch.zeros(1))])},  # state_dict saved under the "model" key
        {"model": {"a": torch.zeros(1)}},  # plain-dict "model" value
        OrderedDict([("a", torch.zeros(1))]),  # bare state_dict, no "model" key
    ],
)
def test_load_checkpoint_state_dict_rejected(ckpt, tmp_path):
    """Test a state_dict checkpoint raises a clear TypeError instead of a cryptic AttributeError/KeyError."""
    weight = tmp_path / "bad.pt"
    torch.save(ckpt, weight)
    with pytest.raises(TypeError, match="supported Ultralytics checkpoint format"):
        load_checkpoint(weight)
>>>>>>> origin/main


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

<<<<<<< HEAD
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 3}
=======
    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
>>>>>>> origin/main
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"
<<<<<<< HEAD
=======


def test_checkpoint_fp16_overflow():
    """Test a finite model whose weights overflow fp16 is still checkpointed (clamped) instead of skipped."""

    def inflate_ema(trainer):
        """Push an EMA weight above the fp16 max (65504) so its fp16 snapshot would otherwise become Inf."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = 1.0e5

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", inflate_ema)
    trainer.train()
    assert trainer.last.exists(), "checkpoint not saved for a finite model with fp16-overflowing weights"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )
    # Validation must leave the live EMA fp32 and unchanged; checkpoint serialization may clamp its fp16 copy.
    ema_param = next(iter(trainer.ema.ema.parameters()))
    assert ema_param.dtype == torch.float32 and torch.isfinite(ema_param).all() and ema_param.flatten()[0] == 1.0e5, (
        "validation corrupted the live EMA"
    )


def test_checkpoint_nonfinite_ema_resync():
    """Test a non-finite EMA on a finite model is resynced (not skipped) so the run still produces a checkpoint."""

    def poison_ema(trainer):
        """Make the live fp32 EMA genuinely non-finite while the model stays finite (sticky-NaN on a finite-loss run)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when the EMA went non-finite on a finite model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


def test_checkpoint_nonfinite_ema_and_model_sanitized():
    """Test a tensor non-finite in both EMA and model is sanitized (not skipped) so the run still produces a checkpoint."""

    def poison_ema_and_model(trainer):
        """Force the first parameter non-finite in both the live EMA and the model (finite-loss sticky-NaN)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")
            next(iter(unwrap_model(trainer.model).parameters())).data.flatten()[0] = float("nan")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema_and_model)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when a tensor went non-finite in both EMA and model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


@pytest.mark.parametrize(
    "kwargs,uses_weights",
    [({}, True), ({"pretrained": True}, True), ({"pretrained": False}, False), ({"pretrained": MODEL}, True)],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_reuses_loaded_checkpoint_model(monkeypatch, kwargs, uses_weights):
    """Test training reuses loaded checkpoint config while respecting the pretrained argument."""
    model = YOLO("yolo26n.yaml")
    model.ckpt = {"checkpoint": True}
    model.ckpt_path = "/tmp/fake.pt"
    model.overrides["model"] = "ul://glenn-jocher/m2/exp-14"
    model.overrides["pretrained"] = False
    original_model = model.model
    captured = {}

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = None
            self.validator = SimpleNamespace(metrics=None)
            self.best = MODEL.parent / "nonexistent-best.pt"
            self.last = MODEL
            captured["trainer"] = self

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["cfg"] = cfg
            captured["weights"] = weights
            return original_model

        def train(self):
            return None

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    monkeypatch.setattr(model, "_smart_load", lambda key: FakeTrainer)
    monkeypatch.setattr(
        "ultralytics.engine.model.load_checkpoint",
        lambda path: (original_model, {"checkpoint": True}),
    )

    model.train(data="coco8.yaml", epochs=1, **kwargs)

    assert captured["trainer"].model is original_model, "Trainer model does not match original"
    assert captured["cfg"] == original_model.yaml, f"Config mismatch: {captured['cfg']} != {original_model.yaml}"
    assert captured["weights"] is (original_model if uses_weights else None), "Unexpected weights loaded"


def test_train_multi_custom_trainer_metrics_and_failure_keys(monkeypatch, tmp_path):
    """Test custom multi-dataset runs keep memory metrics and unique failure keys."""
    model = YOLO(MODEL)
    calls = 0

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            pass

        def get_model(self, cfg=None, weights=None, verbose=True):
            return model.model

        def train(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("failed repeated dataset")
            self.validator = SimpleNamespace(metrics=SimpleNamespace(results_dict={"fitness": 1.0}))

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    results = model.train(
        data=["coco8.yaml", "coco8.yaml"],
        project=tmp_path,
        plots=False,
        save=False,
        trainer=FakeTrainer,
    )

    assert model.trainer.trainer is FakeTrainer
    assert results == {"coco8": {"fitness": 1.0}, "coco8-2": None}


@pytest.mark.parametrize("pretrained,uses_weights", [(True, True), (False, False), (MODEL, True)])
def test_setup_model_respects_pretrained_arg_for_pt_models(monkeypatch, pretrained, uses_weights):
    """Test .pt models use checkpoint config while respecting the pretrained argument."""
    captured = {}
    checkpoint_model = SimpleNamespace(yaml={"nc": 80})
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n.pt"
    trainer.args = SimpleNamespace(pretrained=pretrained)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(
        "ultralytics.engine.trainer.load_checkpoint", lambda path: (checkpoint_model, {"checkpoint": True})
    )

    trainer.setup_model()

    assert captured["cfg"] == checkpoint_model.yaml, "Checkpoint config was not used"
    assert captured["weights"] is (checkpoint_model if uses_weights else None), "Unexpected weights loaded"
>>>>>>> origin/main

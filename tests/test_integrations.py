# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
<<<<<<< HEAD
import os
=======
>>>>>>> origin/main
import subprocess
import time
from pathlib import Path

import pytest

<<<<<<< HEAD
from tests import MODEL, SOURCE
=======
from tests import SOURCE
>>>>>>> origin/main
from ultralytics import YOLO, download
from ultralytics.utils import ASSETS_URL, DATASETS_DIR, SETTINGS
from ultralytics.utils.checks import check_requirements


@pytest.mark.slow
def test_tensorboard():
    """Test training with TensorBoard logging enabled."""
    SETTINGS["tensorboard"] = True
<<<<<<< HEAD
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
=======
    YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
>>>>>>> origin/main
    SETTINGS["tensorboard"] = False


@pytest.mark.skipif(not check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_ray_tune():
    """Tune YOLO model using Ray for hyperparameter optimization."""
<<<<<<< HEAD
    YOLO("yolo11n-cls.yaml").tune(
=======
    YOLO("yolo26n-cls.yaml").tune(
>>>>>>> origin/main
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
<<<<<<< HEAD
def test_mlflow():
    """Test training with MLflow tracking enabled."""
    SETTINGS["mlflow"] = True
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    SETTINGS["mlflow"] = False


@pytest.mark.skipif(True, reason="Test failing in scheduled CI https://github.com/ultralytics/ultralytics/pull/8868")
@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active():
    """Ensure MLflow run status matches MLFLOW_KEEP_RUN_ACTIVE environment variable settings."""
    import mlflow

    SETTINGS["mlflow"] = True
    run_name = "Test Run"
    os.environ["MLFLOW_RUN"] = run_name

    # Test with MLFLOW_KEEP_RUN_ACTIVE=True
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.active_run().info.status
    assert status == "RUNNING", "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"

    run_id = mlflow.active_run().info.run_id

    # Test with MLFLOW_KEEP_RUN_ACTIVE=False
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "False"
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"

    # Test with MLFLOW_KEEP_RUN_ACTIVE not set
    os.environ.pop("MLFLOW_KEEP_RUN_ACTIVE", None)
    YOLO("yolo11n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is not set"
    SETTINGS["mlflow"] = False


@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] not installed")
def test_triton(tmp_path):
=======
def test_mlflow(tmp_path, monkeypatch):
    """Test training with MLflow tracking enabled."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_mlflow")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active(tmp_path, monkeypatch):
    """Ensure MLFLOW_KEEP_RUN_ACTIVE controls whether new MLflow runs remain active."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "keep_run_active")
    monkeypatch.setenv("MLFLOW_RUN", "Test Run")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "True")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        active = mlflow.active_run()
        assert active is not None and active.info.status == "RUNNING", (
            "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"
        )
        mlflow.end_run()

        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "False")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"

        monkeypatch.delenv("MLFLOW_KEEP_RUN_ACTIVE", raising=False)
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is unset"
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] not installed")
def test_triton(tmp_path, isolated_model):
>>>>>>> origin/main
    """Test NVIDIA Triton Server functionalities with YOLO model."""
    check_requirements("tritonclient[all]")
    from tritonclient.http import InferenceServerClient

    # Create variables
    model_name = "yolo"
    triton_repo = tmp_path / "triton_repo"  # Triton repo path
    triton_model = triton_repo / model_name  # Triton model path

    # Export model to ONNX
<<<<<<< HEAD
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
=======
    f = YOLO(isolated_model).export(format="onnx", dynamic=True)
>>>>>>> origin/main

    # Prepare Triton repo
    (triton_model / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model / "1" / "model.onnx")
    (triton_model / "config.pbtxt").touch()

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB

    # Pull the image
    subprocess.call(f"docker pull {tag}", shell=True)

    # Run the Triton server and capture the container ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_repo}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    # Wait for the Triton server to start
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    # Wait until model is ready
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)

    # Check Triton inference
    YOLO(f"http://localhost:8000/{model_name}", "detect")(SOURCE)  # exported model inference

    # Kill and remove the container at the end of the test
    subprocess.call(f"docker kill {container_id}", shell=True)


@pytest.mark.skipif(not check_requirements("faster-coco-eval", install=False), reason="faster-coco-eval not installed")
def test_faster_coco_eval():
    """Validate YOLO model predictions on COCO dataset using faster-coco-eval."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

<<<<<<< HEAD
    args = {"model": "yolo11n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
=======
    args = {"model": "yolo26n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
>>>>>>> origin/main
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    _ = validator.eval_json(validator.stats)

<<<<<<< HEAD
    args = {"model": "yolo11n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
=======
    args = {"model": "yolo26n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
>>>>>>> origin/main
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    _ = validator.eval_json(validator.stats)

<<<<<<< HEAD
    args = {"model": "yolo11n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
=======
    args = {"model": "yolo26n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
>>>>>>> origin/main
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    _ = validator.eval_json(validator.stats)

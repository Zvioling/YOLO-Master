# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
<<<<<<< HEAD
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
OpenVINO                | `openvino`                | yolo11n_openvino_model/
TensorRT                | `engine`                  | yolo11n.engine
CoreML                  | `coreml`                  | yolo11n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo11n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo11n.pb
TensorFlow Lite         | `tflite`                  | yolo11n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo11n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo11n_web_model/
PaddlePaddle            | `paddle`                  | yolo11n_paddle_model/
MNN                     | `mnn`                     | yolo11n.mnn
NCNN                    | `ncnn`                    | yolo11n_ncnn_model/
IMX                     | `imx`                     | yolo11n_imx_model/
RKNN                    | `rknn`                    | yolo11n_rknn_model/
ExecuTorch              | `executorch`              | yolo11n_executorch_model/
Axelera                 | `axelera`                 | yolo11n_axelera_model/
=======
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
DEEPX                   | `deepx`                   | yolo26n_deepx_model/
Qualcomm QNN            | `qnn`                     | yolo26n_qnn.onnx
LiteRT                  | `litert`                  | yolo26n.tflite
Hailo                   | `hailo`                   | yolo26n_hailo_model/
>>>>>>> origin/main

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
<<<<<<< HEAD
    model = YOLO('yolo11n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo11n.pt format=onnx

Inference:
    $ yolo predict model=yolo11n.pt                 # PyTorch
                         yolo11n.torchscript        # TorchScript
                         yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo11n_openvino_model     # OpenVINO
                         yolo11n.engine             # TensorRT
                         yolo11n.mlpackage          # CoreML (macOS-only)
                         yolo11n_saved_model        # TensorFlow SavedModel
                         yolo11n.pb                 # TensorFlow GraphDef
                         yolo11n.tflite             # TensorFlow Lite
                         yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo11n_paddle_model       # PaddlePaddle
                         yolo11n.mnn                # MNN
                         yolo11n_ncnn_model         # NCNN
                         yolo11n_imx_model          # IMX
                         yolo11n_rknn_model         # RKNN
                         yolo11n_executorch_model   # ExecuTorch
                         yolo11n_axelera_model      # Axelera

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo11n_web_model public/yolo11n_web_model
    $ npm start
"""

import json
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
=======
    model = YOLO('yolo26n.pt')
    results = model.export(format='onnx')
    results = model.export(format='onnx', quantize=8, data='coco8.yaml')  # INT8 ONNX

CLI:
    $ yolo mode=export model=yolo26n.pt format=onnx
    $ yolo mode=export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml

Inference:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (macOS-only)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # IMX
                         yolo26n_rknn_model         # RKNN
                         yolo26n_executorch_model   # ExecuTorch
                         yolo26n_axelera_model      # Axelera AI
                         yolo26n_deepx_model        # DEEPX
                         yolo26n_qnn.onnx           # Qualcomm QNN
                         yolo26n.tflite             # LiteRT
"""

from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
>>>>>>> origin/main
from pathlib import Path

import numpy as np
import torch

from ultralytics import __version__
<<<<<<< HEAD
from ultralytics.cfg import TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder
=======
from ultralytics.cfg import QUANTIZE_DOCS_URL, TASK2CALIBRATIONDATA, TASK2DATA, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend, check_class_names, default_class_names
from ultralytics.nn.modules import (
    OBB,
    OBB26,
    C2f,
    Classify,
    Detect,
    Pose,
    Pose26,
    RTDETRDecoder,
    Segment,
    Segment26,
    SemanticSegment,
)
>>>>>>> origin/main
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
<<<<<<< HEAD
    IS_COLAB,
    IS_DEBIAN_BOOKWORM,
    IS_DEBIAN_TRIXIE,
    IS_JETSON,
    IS_RASPBERRYPI,
    IS_UBUNTU,
=======
    IS_DOCKER,
>>>>>>> origin/main
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
<<<<<<< HEAD
=======
    QNN_HTP_ARCHS,
>>>>>>> origin/main
    RKNN_CHIPS,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
<<<<<<< HEAD
)
from ultralytics.utils.checks import (
    IS_PYTHON_3_10,
    IS_PYTHON_MINIMUM_3_9,
    check_apt_requirements,
=======
    is_jetson,
)
from ultralytics.utils.checks import (
    IS_PYTHON_MINIMUM_3_9,
    IS_PYTHON_MINIMUM_3_13,
>>>>>>> origin/main
    check_imgsz,
    check_requirements,
    check_version,
    is_intel,
<<<<<<< HEAD
    is_sudo_available,
)
from ultralytics.utils.export import (
    keras2pb,
    onnx2engine,
    onnx2saved_model,
    pb2tfjs,
    tflite2edgetpu,
    torch2imx,
    torch2onnx,
=======
>>>>>>> origin/main
)
from ultralytics.utils.files import file_size
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import (
<<<<<<< HEAD
    TORCH_1_10,
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_4,
=======
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_3,
    TORCH_2_8,
>>>>>>> origin/main
    TORCH_2_9,
    select_device,
)


def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
<<<<<<< HEAD
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "half", "nms", "dynamic"]],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],
=======
    #          Format, Argument, Suffix, CPU, GPU, Arguments, Env
    x = [
        ["PyTorch", "-", ".pt", True, True, [], "base"],
        [
            "TorchScript",
            "torchscript",
            ".torchscript",
            True,
            True,
            ["batch", "quantize", "nms", "dynamic"],
            "base",
        ],
        [
            "ONNX",
            "onnx",
            ".onnx",
            True,
            True,
            ["batch", "data", "dynamic", "quantize", "opset", "simplify", "nms", "fraction"],
            "base",
        ],
>>>>>>> origin/main
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
<<<<<<< HEAD
            ["batch", "dynamic", "half", "int8", "nms", "fraction"],
=======
            ["batch", "data", "dynamic", "quantize", "nms", "fraction"],
            "base",
>>>>>>> origin/main
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
<<<<<<< HEAD
            ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "half", "int8", "nms"]],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras", "nms"]],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8", "nms", "fraction"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8", "nms"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["IMX", "imx", "_imx_model", True, True, ["int8", "fraction", "nms"]],
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"]],
        ["Axelera", "axelera", "_axelera_model", False, False, ["batch", "int8"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def best_onnx_opset(onnx, cuda=False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())
=======
            ["batch", "data", "dynamic", "quantize", "simplify", "nms", "fraction"],
            "base",
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "quantize", "nms"], "coreml"],
        [
            "TensorFlow SavedModel",
            "saved_model",
            "_saved_model",
            True,
            True,
            ["batch", "data", "fraction", "quantize", "keras", "nms"],
            "tensorflow",
        ],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"], "tensorflow"],
        [
            "TensorFlow Edge TPU",
            "edgetpu",
            "_edgetpu.tflite",
            True,
            False,
            ["data", "fraction", "quantize"],
            "tensorflow",
        ],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"], "base"],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "dynamic", "quantize", "nms"], "mnn"],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "quantize"], "ncnn"],
        ["IMX", "imx", "_imx_model", True, True, ["data", "quantize", "fraction", "nms"], "isolated-imx"],
        [
            "RKNN",
            "rknn",
            "_rknn_model",
            False,
            False,
            ["batch", "name", "quantize", "data", "fraction"],
            "isolated-rknn",
        ],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"], "executorch"],
        [
            "Axelera AI",
            "axelera",
            "_axelera_model",
            False,
            False,
            ["batch", "quantize", "fraction", "data"],
            "isolated-axelera",
        ],
        ["DEEPX", "deepx", "_deepx_model", False, False, ["data", "quantize", "optimize"], "isolated-deepx"],
        ["Qualcomm QNN", "qnn", "_qnn.onnx", False, False, ["batch", "name", "quantize", "fraction", "data"], "base"],
        ["LiteRT", "litert", ".tflite", True, False, ["batch", "quantize", "data", "fraction"], "litert"],
        [
            "Hailo",
            "hailo",
            "_hailo_model",
            False,
            False,
            ["name", "quantize", "data", "fraction", "opset", "simplify", "conf", "iou"],
            "base",
        ],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments", "Env"], zip(*x)))


EXPORT_ENVS = {
    "base": {
        "python": None,
        "extras": ["export-base"],
        "torch": None,
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": [],
    },
    "tensorflow": {
        "python": "3.12",
        "extras": ["export-base", "export-tensorflow"],
        "torch": None,
        "requirements": [
            "onnx2tf>=1.26.3,<1.29.0",
            "tf_keras<=2.19.0",
            "sng4onnx>=1.0.1",
            "onnx_graphsurgeon>=0.3.26",
            "ai-edge-litert>=1.2.0",
            "onnxruntime",
            "protobuf>=5",
        ],
        "indexes": [
            ("--extra-index-url", "https://pypi.ngc.nvidia.com"),
        ],
        "env": {},
        "smoke": ["yolo export format=saved_model model=yolo26n.pt imgsz=32"],
    },
    "coreml": {
        "python": "3.13",
        "extras": ["export-base", "export-coreml"],
        "torch": ">=2.12",
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=coreml model=yolo26n.pt imgsz=32"],
    },
    "mnn": {
        "python": "3.13",
        "extras": ["export-base"],
        "torch": None,
        "requirements": ["MNN>=2.9.6", "aliyun-log-python-sdk", "protobuf<6.0.0,>=3.20.3"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=mnn model=yolo26n.pt imgsz=32"],
    },
    "ncnn": {
        "python": "3.13",
        "extras": ["export-base"],
        "torch": None,
        "requirements": ["ncnn", "pnnx==20260526"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=ncnn model=yolo26n.pt imgsz=32"],
    },
    "executorch": {
        "python": "3.13",
        "extras": ["export-base", "export-executorch"],
        "torch": ">=2.12",
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=executorch model=yolo26n.pt imgsz=32"],
    },
    "isolated-imx": {
        "python": "3.11",
        "extras": ["export-base"],
        "torch": ">=2.9,<2.12",
        "requirements": [
            "model-compression-toolkit>=2.4.1",
            "edge-mdt-cl<1.1.0",
            "edge-mdt-tpc>=1.2.0",
            "pydantic<2.12",
            "imx500-converter[pt]>=3.17.3",
        ],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=imx model=yolo11n.pt imgsz=32 data=coco8.yaml"],
    },
    "isolated-rknn": {
        "python": "3.11",
        "extras": ["export-base"],
        "torch": "==2.4",
        "requirements": ["rknn-toolkit2>=2.3.2", "onnx>=1.16.1,<1.19.0", "setuptools<82"],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=rknn model=yolo26n.pt imgsz=32 quantize=16"],
    },
    "isolated-axelera": {
        # Axelera devkit 1.7.0 does not provide Python 3.13 wheels.
        "python": "3.12",
        "extras": ["export-base"],
        # Axelera export requires 2.8.0 <= torch < 2.12.0.
        "torch": ">=2.8,<2.12",
        "requirements": [
            "axelera-devkit==1.7.0",
            "omnimalloc==0.5.0",
            "numpy<=2.3.5",
            "onnx>=1.12.0,<2.0.0",
            "onnxslim>=0.1.71",
        ],
        "indexes": [
            ("--extra-index-url", "https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple"),
        ],
        # Use the Python protobuf runtime for Axelera compiler compatibility.
        "env": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
        "smoke": ["yolo export format=axelera model=yolo26n.pt imgsz=64 data=coco8.yaml"],
    },
    "isolated-deepx": {
        # dx-com 2.3.0 does not provide Python 3.13 wheels.
        "python": "3.12",
        "extras": ["export-base", "export-deepx"],
        "torch": ">=2.8,<2.12",
        "requirements": [],
        "indexes": [
            ("--find-links", "https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html"),
        ],
        # DeepX export is only supported on non-aarch64 Linux.
        "env": {},
        "smoke": ["yolo export format=deepx model=yolo26n.pt imgsz=32 data=coco8.yaml"],
    },
    "litert": {
        "python": "3.13",
        "extras": ["export-base", "export-litert"],
        "torch": None,
        "requirements": [],
        "indexes": [],
        "env": {},
        "smoke": ["yolo export format=litert model=yolo26n.pt imgsz=32"],
    },
}


# Export precision support per format. Unset/32 requests are FP32 except for formats listed in FP32_UNSUPPORTED_FORMATS.
FP16_FORMATS = frozenset({"torchscript", "onnx", "openvino", "engine", "coreml", "mnn", "ncnn", "rknn"})
INT8_FORMATS = frozenset(
    {
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "edgetpu",
        "mnn",
        "imx",
        "rknn",
        "axelera",
        "deepx",
        "hailo",
        "litert",
    }
)
W8A16_FORMATS = frozenset(
    {"coreml", "imx", "qnn", "litert"}
)  # INT8 weights + 16-bit activations (FP16; INT16 on LiteRT)
W8A32_FORMATS = frozenset({"litert"})  # INT8 weights + FP32 activations (dynamic/weight-only INT8, no calibration)
FP32_UNSUPPORTED_FORMATS = frozenset({"edgetpu", "imx", "rknn", "axelera", "deepx", "qnn", "hailo"})
# (label, supporting formats) per quantize precision, used to list valid options in errors. 32/None (FP32) is universal except FP32_UNSUPPORTED_FORMATS.
QUANTIZE_PRECISIONS = (
    ("16 (FP16)", FP16_FORMATS),
    ("8 (INT8)", INT8_FORMATS),
    ("'w8a16' (INT8 weights + INT16 activations)", W8A16_FORMATS),
    ("'w8a32' (dynamic INT8)", W8A32_FORMATS),
)
>>>>>>> origin/main


def validate_args(format, passed_args, valid_args):
    """Validate arguments based on the export format.

    Args:
        format (str): The export format.
<<<<<<< HEAD
        passed_args (Namespace): The arguments used during export.
=======
        passed_args (SimpleNamespace): The arguments used during export.
>>>>>>> origin/main
        valid_args (list): List of valid arguments for the format.

    Raises:
        AssertionError: If an unsupported argument is used, or if the format lacks supported argument listings.
    """
<<<<<<< HEAD
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch", "fraction"]
=======
    export_args = ["dynamic", "keras", "nms", "batch", "fraction", "data", "optimize"]
>>>>>>> origin/main

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # exporter defaults
    default_args = get_cfg(DEFAULT_CFG, custom)
<<<<<<< HEAD
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
=======
    if passed_args.quantize is not None:  # 32/None (FP32) is universal except FP32_UNSUPPORTED_FORMATS
        options = [label for label, formats in QUANTIZE_PRECISIONS if format in formats]
        if format not in FP32_UNSUPPORTED_FORMATS:
            options.append("32 (FP32)")
        hint = f"format='{format}' supports quantize={', '.join(options) or 'none'} (or None for FP32). See {QUANTIZE_DOCS_URL}"
        if passed_args.quantize == 16:  # FP16
            assert format in FP16_FORMATS, f"ERROR ❌️ quantize=16 (FP16) is not supported; {hint}"
        elif passed_args.quantize == 8:  # INT8
            assert format in INT8_FORMATS, f"ERROR ❌️ quantize=8 (INT8) is not supported; {hint}"
        elif passed_args.quantize == "w8a16":  # INT8 weights + 16-bit activations (FP16; INT16 on LiteRT)
            assert format in W8A16_FORMATS, f"ERROR ❌️ quantize='w8a16' is not supported; {hint}"
        elif passed_args.quantize == "w8a32":  # INT8 weights + FP32 activations (dynamic/weight-only INT8)
            assert format in W8A32_FORMATS, f"ERROR ❌️ quantize='w8a32' is not supported; {hint}"
        elif passed_args.quantize == 32:  # FP32
            assert format not in FP32_UNSUPPORTED_FORMATS, f"ERROR ❌️ quantize=32 (FP32) is not supported; {hint}"
    for arg in export_args:
        not_default = getattr(passed_args, arg, getattr(default_args, arg, None)) != getattr(default_args, arg, None)
>>>>>>> origin/main
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # exported file/dir or tuple of (file/dir, *)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
<<<<<<< HEAD
            assert mb > 0.0, "0.0 MB output model size"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
=======
            assert mb > 0.1, f"{mb:.3f} MB output model too small (likely corrupt or unsupported ops)"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            dependency_help = (
                " Ultralytics Platform runs exports in the cloud with no local dependencies required. "
                "Visit https://platform.ultralytics.com."
                if isinstance(e, ImportError)
                else ""
            )
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}{dependency_help}")
>>>>>>> origin/main
            raise e

    return outer_func


class Exporter:
    """A class for exporting YOLO models to various formats.

    This class provides functionality to export YOLO models to different formats including ONNX, TensorRT, CoreML,
    TensorFlow, and others. It handles format validation, device selection, model preparation, and the actual export
    process for each supported format.

    Attributes:
        args (SimpleNamespace): Configuration arguments for the exporter.
        callbacks (dict): Dictionary of callback functions for different export events.
        im (torch.Tensor): Input tensor for model inference during export.
        model (torch.nn.Module): The YOLO model to be exported.
        file (Path): Path to the model file being exported.
        output_shape (tuple): Shape of the model output tensor(s).
        pretty_name (str): Formatted model name for display purposes.
        metadata (dict): Model metadata including description, author, version, etc.
        device (torch.device): Device on which the model is loaded.
<<<<<<< HEAD
        imgsz (tuple): Input image size for the model.
=======
        imgsz (list): Input image size for the model.
>>>>>>> origin/main

    Methods:
        __call__: Main export method that handles the export process.
        get_int8_calibration_dataloader: Build dataloader for INT8 calibration.
        export_torchscript: Export model to TorchScript format.
        export_onnx: Export model to ONNX format.
        export_openvino: Export model to OpenVINO format.
        export_paddle: Export model to PaddlePaddle format.
        export_mnn: Export model to MNN format.
        export_ncnn: Export model to NCNN format.
        export_coreml: Export model to CoreML format.
        export_engine: Export model to TensorRT format.
        export_saved_model: Export model to TensorFlow SavedModel format.
        export_pb: Export model to TensorFlow GraphDef format.
<<<<<<< HEAD
        export_tflite: Export model to TensorFlow Lite format.
        export_edgetpu: Export model to Edge TPU format.
        export_tfjs: Export model to TensorFlow.js format.
        export_rknn: Export model to RKNN format.
        export_imx: Export model to IMX format.

    Examples:
        Export a YOLOv8 model to ONNX format
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolov8n.pt")  # exports to yolov8n.onnx

        Export with specific arguments
        >>> args = {"format": "onnx", "dynamic": True, "half": True}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolov8n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file.
=======
        export_edgetpu: Export model to Edge TPU format.
        export_rknn: Export model to RKNN format.
        export_imx: Export model to IMX format.
        export_executorch: Export model to ExecuTorch format.
        export_axelera: Export model to Axelera format.
        export_deepx: Export model to DEEPX format.

    Examples:
        Export a YOLO26 model to TorchScript format
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolo26n.pt")  # exports to yolo26n.torchscript

        Export with specific arguments
        >>> args = {"format": "onnx", "dynamic": True, "quantize": 8, "data": "coco8.yaml"}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolo26n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the Exporter class.

        Args:
            cfg (str | Path | dict | SimpleNamespace, optional): Configuration file path or configuration object.
>>>>>>> origin/main
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """Export a model and return the final exported path as a string.

        Returns:
            (str): Path to the exported file or directory (the last export artifact).
        """
        t = time.time()
        fmt = self.args.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
<<<<<<< HEAD
=======
        if fmt in {"tflite", "tfjs"}:  # deprecated formats, replaced by the unified Google LiteRT export
            LOGGER.warning(
                f"format='{fmt}' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT "
                f"format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert/"
            )
            fmt = self.args.format = "litert"
>>>>>>> origin/main
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # available export formats
        if fmt not in fmts:
            import difflib

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60% similarity required to match
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
<<<<<<< HEAD
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        (
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            executorch,
            axelera,
        ) = flags  # export booleans

        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # Device
        dla = None
        if engine and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if engine and "dla" in str(self.args.device):  # convert int/list to str first
            device_str = str(self.args.device)
            dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if imx and self.args.device is None and torch.cuda.is_available():
=======
        from ultralytics.utils.export_preflight import export_preflight

        molora_export_mode = str(getattr(self.args, "molora_export_mode", "dynamic")).lower()
        if molora_export_mode not in {"dynamic", "routing_preserved"}:
            raise ValueError("molora_export_mode must be 'dynamic' or 'routing_preserved'")
        if molora_export_mode == "routing_preserved" and fmt not in {"onnx", "torchscript"}:
            raise ValueError("molora_export_mode='routing_preserved' is only supported for ONNX/TorchScript export")
        preflight_kwargs = {}
        if molora_export_mode == "routing_preserved":
            preflight_kwargs["routing_preserved"] = True
        self.export_preflight_report = export_preflight(model, fmt, strict=True, **preflight_kwargs)
        decisions = self.export_preflight_report["decisions"]
        if decisions:
            LOGGER.info(
                f"Export preflight: format={fmt}, routed_modules={len(decisions)}, "
                f"strategies={sorted({item['strategy'] for item in decisions})}"
            )
        is_tf_format = fmt in {"saved_model", "pb", "edgetpu"}

        # Device
        self.dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # convert int/list to str first
            device_str = str(self.args.device)
            self.dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert self.dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if fmt == "imx" and self.args.device is None and torch.cuda.is_available():
>>>>>>> origin/main
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # update device to "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Argument compatibility checks
<<<<<<< HEAD
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if axelera:
            if not IS_PYTHON_3_10:
                raise SystemError("Axelera export only supported on Python 3.10.")
            if not self.args.int8:
                LOGGER.warning("Setting int8=True for Axelera mixed-precision export.")
                self.args.int8 = True
            if model.task not in {"detect"}:
                raise ValueError("Axelera export only supported for detection models.")
        if imx:
            if not self.args.int8:
                LOGGER.warning("IMX export requires int8=True, setting int8=True.")
                self.args.int8 = True
=======
        fmt_keys = dict(zip(fmts_dict["Argument"], fmts_dict["Arguments"]))[fmt]
        validate_args(fmt, self.args, fmt_keys)
        if fmt in {"deepx", "axelera", "imx", "edgetpu", "qnn", "hailo"} and self.args.quantize not in {8, "w8a16"}:
            if self.args.quantize == 32:
                raise ValueError(
                    f"{fmt} export only supports INT8, but got an explicit quantize=32 (FP32) request. "
                    f"See {QUANTIZE_DOCS_URL}"
                )
            LOGGER.warning(f"{fmt} export requires INT8 quantization, enabling it.")
            self.args.quantize = "w8a16" if fmt == "qnn" else 8
        if fmt in {"axelera", "hailo"} and not self.args.data:
            self.args.data = TASK2CALIBRATIONDATA.get(model.task)
        if fmt == "hailo":
            assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
            blocks = {str(x[2]) for x in model.yaml.get("backbone", []) + model.yaml.get("head", [])}
            family = Path(getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")).stem.lower() or (
                "yolov8" if "C2f" in blocks else "yolo11" if {"C3k2", "C2PSA"} <= blocks else ""
            )
            task26 = {Segment26: "segmentation", Pose26: "pose", OBB26: "OBB"}.get(type(model.model[-1]))
            if task26:
                raise ValueError(f"Hailo export does not currently support YOLO26 {task26} models.")
            if (
                model.task not in {"detect", "segment", "pose", "obb", "classify"}
                or type(model.model[-1]) not in {Detect, Segment, Pose, OBB, Classify}
                or not family.startswith(("yolov8", "yolo11", "yolo26"))
            ):
                raise ValueError(
                    "Hailo export currently supports YOLOv8/YOLO11/YOLO26 detection and classification models "
                    "and YOLOv8/YOLO11 segmentation, pose, and OBB models."
                )
            if self.args.end2end is not None:
                raise ValueError(
                    "Hailo export selects the model output path automatically; remove the end2end argument."
                )
            if self.args.opset not in {None, 11}:
                raise ValueError("Hailo export requires opset=11.")
            self.args.name = str(self.args.name or "hailo8l").lower()
            hailo_archs = ("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l")
            if self.args.name not in hailo_archs:
                raise ValueError(f"Invalid Hailo architecture '{self.args.name}'. Valid names are {hailo_archs}.")
        if fmt == "axelera":
            if model.task == "segment" and any(isinstance(m, Segment26) for m in model.modules()):
                raise ValueError("Axelera export does not currently support YOLO26 segmentation models.")
        if fmt == "imx":
>>>>>>> origin/main
            if not self.args.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
<<<<<<< HEAD
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and jit and self.device.type == "cpu":
            LOGGER.warning(
                "half=True only compatible with GPU export for TorchScript, i.e. use device=0, setting half=False."
            )
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if rknn:
=======
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu", "qnn"}:
                # Disable end2end branch for certain export formats as they does not support topk
                model.end2end = False
                LOGGER.warning(f"{fmt.upper()} export does not support end2end models, disabling end2end branch.")
            if fmt == "litert" and self.args.quantize in {8, "w8a16"}:
                # Static activation quantization collapses the end2end class-index output; export raw and run NMS later
                model.end2end = False
                LOGGER.warning("LiteRT INT8 export does not support end2end models, disabling end2end branch.")
            if fmt == "engine":
                try:
                    import tensorrt as trt

                    if check_version(trt.__version__, "<8.5.0"):
                        # https://github.com/ultralytics/ultralytics/issues/24607
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT versions earlier than 8.5.0 do not support the Mod operator in end-to-end models, disabling the end2end branch. "
                            "Please upgrade TensorRT to 8.5.0 or later to enable end2end export."
                        )

                    if (
                        self.args.quantize == 8
                        and check_version(trt.__version__, ">=10.3.0,<10.4.0")  # JetPack 6 builds report 10.3.0.x
                        and is_jetson(jetpack=6)
                    ):
                        # https://github.com/ultralytics/ultralytics/issues/23841
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT 10.3.0 on JetPack 6 with int8 has known end2end build issues, disabling end2end branch. "
                            "For a fix, see https://docs.ultralytics.com/guides/nvidia-jetson/#why-does-my-tensorrt-int8-export-disable-end2end-on-jetpack-6"
                            ""
                        )
                except ImportError:
                    pass
        if self.args.quantize == 16 and fmt == "torchscript" and self.device.type == "cpu":
            raise ValueError("FP16 TorchScript export is only supported on GPU, i.e. use device=0.")
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if fmt == "axelera" and min(self.imgsz) < 64:
            raise ValueError(f"Axelera export requires imgsz>=64, but got imgsz={self.imgsz}.")
        if fmt == "rknn":
>>>>>>> origin/main
            if not self.args.name:
                LOGGER.warning(
                    "Rockchip RKNN export requires a missing 'name' arg for processor type. "
                    "Using default name='rk3588'."
                )
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."
            )
<<<<<<< HEAD
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not tflite or not ARM64 or not LINUX, "TFLite export with NMS unsupported on ARM64 Linux"
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert not onnx or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
=======
            if self.args.name in {"rv1103", "rv1106", "rv1103b", "rv1106b"} and self.args.quantize != 8:
                if self.args.quantize not in {None, 8}:
                    raise ValueError(
                        f"Rockchip target '{self.args.name}' only supports INT8, but got quantize={self.args.quantize}. "
                        f"See {QUANTIZE_DOCS_URL}"
                    )
                LOGGER.warning(f"Rockchip target '{self.args.name}' requires INT8 quantization, enabling it.")
                self.args.quantize = 8
            elif self.args.quantize is None:
                self.args.quantize = 16
        if fmt == "qnn":
            if not self.args.name:
                LOGGER.warning(
                    "Qualcomm QNN export requires a missing 'name' arg for the target Hexagon HTP architecture. "
                    "Using default name='73' (Snapdragon 8 Gen 2)."
                )
                self.args.name = "73"
            self.args.name = str(self.args.name).lower().lstrip("v")  # accept '73' or 'v73'
            assert self.args.name in QNN_HTP_ARCHS, (
                f"Invalid HTP architecture '{self.args.name}' for Qualcomm QNN export. Valid archs are {QNN_HTP_ARCHS} "
                "(Snapdragon 888/8Gen1/8Gen2/8Gen3/8Elite/8EliteGen5 respectively)."
            )
        if self.args.nms and model.task == "semantic":
            LOGGER.warning("'nms=True' is not valid for semantic segmentation models. Forcing 'nms=False'.")
            self.args.nms = False
        if fmt == "coreml" and self.args.nms and model.task != "detect":
            LOGGER.warning("CoreML 'nms=True' is only supported for detect models. Forcing 'nms=False'.")
            self.args.nms = False
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert fmt != "onnx" or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
>>>>>>> origin/main
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # set conf default value for nms export
<<<<<<< HEAD
        if (engine or coreml or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if edgetpu:
=======
        if fmt == "mnn" and self.args.nms:
            if self.args.dynamic:
                raise ValueError("Alibaba MNN export does not support combining 'dynamic=True' with 'nms=True'.")
            if model.task not in {"detect", "pose"}:
                raise ValueError("Alibaba MNN export with 'nms=True' only supports detect and pose models.")
        if (fmt in {"engine", "coreml"} or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if fmt == "edgetpu":
>>>>>>> origin/main
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 export error: https://github.com/ultralytics/ultralytics/pull/18445
<<<<<<< HEAD
        if self.args.int8 and not self.args.data:
            if axelera:
                self.args.data = "coco128.yaml"  # Axelera default to coco128.yaml
            else:
                self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
        if tfjs and (ARM64 and LINUX):
            raise SystemError("TF.js exports are not currently supported on ARM64 Linux")
=======
        if self.args.quantize in {8, "w8a16"} and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
>>>>>>> origin/main
        # Recommend OpenVINO if export and Intel CPU
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino/"
                )
            SETTINGS["openvino_msg"] = False

        # Input
        im = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
<<<<<<< HEAD
        model = model.fuse()

        if imx:
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if tflite or edgetpu:
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
=======
        for module in model.modules():
            if type(module).__name__ in {"MoLoRALayer", "MoLoRAMoEAwareLayer"}:
                module._export_mode = molora_export_mode
        if getattr(self.args, "pre_export_prune", False):
            model = self._pre_export_prune(model, im, copy_model=False)
        model = model.fuse()

        if fmt == "imx":
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if fmt == "edgetpu":
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        if fmt == "executorch":
            from ultralytics.utils.export.executorch import executorch_wrapper

            model = executorch_wrapper(model)
        for m in model.modules():
            if isinstance(m, (Classify, SemanticSegment)):
                m.export = True
                m.format = self.args.format
                # Semantic argmax bake needs an integer graph output; TensorRT supports uint8 outputs only on TRT>=10
                # (Jetson TRT 8.x rejects them). Read the version from the package name to avoid importing tensorrt here.
                if isinstance(m, SemanticSegment) and fmt == "engine":
                    cuda_major = (torch.version.cuda or "12").split(".")[0]
                    m.bake_argmax = check_version(f"tensorrt-cu{cuda_major}", ">=10.0.0") or check_version(
                        "tensorrt", ">=10.0.0"
                    )
>>>>>>> origin/main
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
<<<<<<< HEAD
                m.max_det = self.args.max_det
                m.xyxy = self.args.nms and not coreml
                if hasattr(model, "pe") and hasattr(m, "fuse"):  # for YOLOE models
=======
                # Clamp max_det to available queries/anchors (required for TensorRT compatibility)
                available = (
                    m.num_queries
                    if isinstance(m, RTDETRDecoder)
                    else sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                )
                m.max_det = min(self.args.max_det, available)
                m.agnostic_nms = self.args.agnostic_nms
                m.xyxy = self.args.nms and fmt != "coreml"
                m.shape = None  # reset cached shape for new export input size
                if hasattr(model, "pe") and hasattr(m, "fuse") and not hasattr(m, "lrpc"):  # for YOLOE models
>>>>>>> origin/main
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

<<<<<<< HEAD
        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms and not coreml and not imx else model(im)
        if self.args.half and (onnx or jit) and self.device.type != "cpu":
=======
        if model.task == "semantic" and fmt in {"qnn", "coreml"}:
            # NPU-targeted semantic exports ship a compact uint8 class map instead of float logits: emitting logits
            # forces consumers to dequantize and argmax ~20M floats on the CPU every frame (measured erratic
            # 123-1065 ms on Hexagon). Not applied to LiteRT, where the GPU delegate cannot compile ArgMax (int64
            # indices) and a whole-graph CPU fallback is slower than GPU logits + consumer-side argmax. Python
            # predict/val accept both forms.
            model = ClassMapModel(model)

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms and fmt not in {"coreml", "imx"} else model(im)
        if self.args.quantize == 16 and fmt in {"onnx", "torchscript"} and self.device.type != "cpu":
>>>>>>> origin/main
            im, model = im.half(), model.half()  # to FP16

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
<<<<<<< HEAD
=======
            "head": type(model.model[-1]).__name__,
>>>>>>> origin/main
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
<<<<<<< HEAD
        }  # model metadata
        if dla is not None:
            self.metadata["dla"] = dla  # make sure `AutoBackend` uses correct dla device if it has one
=======
            "end2end": getattr(model, "end2end", False),
        }  # model metadata
        if self.export_preflight_report["decisions"]:
            self.metadata["mixture_export_preflight"] = self.export_preflight_report
        self.metadata["molora_export_mode"] = molora_export_mode
        if hasattr(self, "moe_prune_manifest"):
            self.metadata["moe_prune_manifest"] = self.moe_prune_manifest
        if self.dla is not None:
            self.metadata["dla"] = self.dla  # make sure `AutoBackend` uses correct dla device if it has one
>>>>>>> origin/main
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape
            if hasattr(model, "kpt_names"):
                self.metadata["kpt_names"] = model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")
<<<<<<< HEAD
        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit:  # TorchScript
            f[0] = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1] = self.export_engine(dla=dla)
        if onnx:  # ONNX
            f[2] = self.export_onnx()
        if xml:  # OpenVINO
            f[3] = self.export_openvino()
        if coreml:  # CoreML
            f[4] = self.export_coreml()
        if is_tf_format:  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6] = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7] = self.export_tflite()
            if edgetpu:
                f[8] = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9] = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10] = self.export_paddle()
        if mnn:  # MNN
            f[11] = self.export_mnn()
        if ncnn:  # NCNN
            f[12] = self.export_ncnn()
        if imx:
            f[13] = self.export_imx()
        if rknn:
            f[14] = self.export_rknn()
        if executorch:
            f[15] = self.export_executorch()
        if axelera:
            f[16] = self.export_axelera()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
=======

        # Export
        if is_tf_format:
            f, keras_model = self.export_saved_model()
            if fmt == "pb":
                f = self.export_pb(keras_model=keras_model)
            if fmt == "edgetpu":
                f = self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")
        else:
            f = getattr(self, f"export_{fmt}")()

        if hasattr(self, "moe_prune_manifest") and f:
            artifact = Path(f)
            manifest_path = (
                artifact / "moe-prune.json"
                if artifact.is_dir()
                else artifact.with_suffix(artifact.suffix + ".prune.json")
            )
            self.moe_prune_manifest["output_artifact"] = str(f)
            manifest_path.write_text(json.dumps(self.moe_prune_manifest, indent=2, sort_keys=True) + "\n")
            LOGGER.info("MoE pre-export pruning manifest saved to %s", manifest_path)

        # Finish
        if f:
>>>>>>> origin/main
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
<<<<<<< HEAD
            predict_data = f"data={data}" if model.task == "segment" and pb else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
=======
            q = "quantize=16" if self.args.quantize == 16 else ""  # FP16 inference flag for the val/predict hint
            inference_commands = (
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                if fmt in AutoBackend._BACKEND_MAP
                else ""
            )
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', Path(f).resolve())}"
                f"{inference_commands}"
>>>>>>> origin/main
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # path to final export artifact

<<<<<<< HEAD
    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader for calibration of INT8 models."""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=self.model.task,
            imgsz=self.imgsz[0],
            augment=False,
            batch_size=self.args.batch,
        )
        n = len(dataset)
        if n < self.args.batch:
            raise ValueError(
                f"The calibration dataset ({n} images) must have at least as many images as the batch size "
                f"('batch={self.args.batch}')."
            )
        elif self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=self.args.batch, workers=0, drop_last=True)  # required for batch loading
=======
    def _pre_export_prune(self, model, calibration_input, *, copy_model=True):
        """Calibrate and prune a copied model before graph export."""
        from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
        from ultralytics.nn.modules.moe.pruning import prune_moe_module

        # Keep this helper safe for direct callers as well as the exporter,
        # whose normal path has already created a deployment copy.
        if copy_model:
            model = deepcopy(model)
        steps = max(int(getattr(self.args, "moe_prune_calibration_steps", 8)), 1)
        model_root = getattr(model, "model", model)
        tracker = ExpertUsageTracker(model_root)
        try:
            with torch.no_grad():
                for _ in range(steps):
                    model(calibration_input)
        finally:
            tracker.remove_hooks()
        usage_stats = dict(tracker.usage_stats)
        manifest = {
            "schema_version": 1,
            "applied": bool(usage_stats),
            "calibration_steps": steps,
            "threshold": float(getattr(self.args, "moe_prune_threshold", 0.15)),
            "keep_top_m": getattr(self.args, "moe_prune_keep_top_m", None),
            "usage_layers": sorted(usage_stats),
        }
        if usage_stats:
            pruned_model, plan = prune_moe_module(
                model.model,
                usage_stats,
                threshold=manifest["threshold"],
                keep_top_m=manifest["keep_top_m"],
            )
            model.model = pruned_model
            manifest["pruning_plan"] = plan
            LOGGER.info("MoE pre-export pruning retained %d routed layers", len(plan))
        else:
            manifest["reason"] = "no routed MoE usage observed during calibration"
            LOGGER.warning("pre_export_prune requested but no MoE routing usage was observed")
        self.moe_prune_manifest = manifest
        return model

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader for calibration of INT8 models."""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        cfg = deepcopy(self.args)
        cfg.imgsz = max(self.imgsz)
        if self.model.task == "classify":
            import torchvision.transforms as T  # scope for faster 'import ultralytics'

            data = check_cls_dataset(self.args.data)
            dataset = ClassificationDataset(data[self.args.split or "val"], args=cfg, augment=False)
            # INT8 backends divide images by 255, so emit uint8 [0, 255] center-cropped like classify inference
            dataset.torch_transforms = T.Compose([T.Resize(cfg.imgsz), T.CenterCrop(cfg.imgsz), T.PILToTensor()])
        else:
            data = check_det_dataset(self.args.data, split=self.args.split)
            dataset = build_yolo_dataset(
                cfg,
                data[self.args.split or "val"],
                self.args.batch,
                data,
                mode="val",
                fraction=self.args.fraction,
            )
        if hasattr(dataset, "transforms") and hasattr(dataset.transforms.transforms[0], "new_shape"):
            dataset.transforms.transforms[0].new_shape = self.imgsz  # LetterBox with non-square imgsz
        n = len(dataset)
        if n < 1:
            raise ValueError(f"The calibration dataset must have at least 1 image, but found {n} images.")
        batch = min(self.args.batch, n)
        if n < self.args.batch:
            LOGGER.warning(
                f"{prefix} calibration dataset has only {n} images, reducing calibration batch size to {batch}."
            )
        if self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)  # required for batch loading
>>>>>>> origin/main

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """Export YOLO model to TorchScript format."""
<<<<<<< HEAD
        LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")
        f = self.file.with_suffix(".torchscript")

        ts = torch.jit.trace(NMSModel(self.model, self.args) if self.args.nms else self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f
=======
        from ultralytics.utils.export.torchscript import torch2torchscript

        return torch2torchscript(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            output_file=self.file.with_suffix(".torchscript"),
            metadata=self.metadata,
            prefix=prefix,
        )
>>>>>>> origin/main

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """Export YOLO model to ONNX format."""
        requirements = ["onnx>=1.12.0,<2.0.0"]
<<<<<<< HEAD
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
=======
        if self.args.simplify or (self.args.format == "onnx" and self.args.quantize == 8):
            # Pass onnxruntime variants as interchangeable candidates so AutoUpdate keeps an installed build
            # (e.g. onnxruntime-qnn for QNN export) instead of reinstalling stable onnxruntime and breaking its ABI.
            ort = "onnxruntime-gpu" if "cuda" in self.device.type else "onnxruntime"
            requirements += [(ort, "onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn")]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.82"]
        check_requirements(requirements)
        import onnx

        from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type, quantize=self.args.quantize)
>>>>>>> origin/main
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:  # only batch size is dynamic with NMS
                dynamic["output0"].pop(2)
        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset  # for NMSModel
<<<<<<< HEAD

        with arange_patch(self.args):
=======
            self.args.simplify = True  # fix OBB runtime error related to topk

        with arange_patch(dynamic=bool(dynamic), quantize=self.args.quantize, fmt=self.args.format):
>>>>>>> origin/main
            torch2onnx(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR version
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # FP16 conversion for CPU export (GPU exports are already FP16 from model.half() during tracing)
<<<<<<< HEAD
        if self.args.half and self.args.format == "onnx" and self.device.type == "cpu":
=======
        if self.args.quantize == 16 and self.args.format == "onnx" and self.device.type == "cpu":
>>>>>>> origin/main
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
<<<<<<< HEAD
=======
        if self.args.quantize == 8 and self.args.format == "onnx":
            from ultralytics.utils.export.onnx import onnx_int8_quantize

            source = Path(f)
            f_int8 = str(source.with_name(f"{source.stem}_int8{source.suffix}"))
            f = onnx_int8_quantize(
                source,
                f_int8,
                self.get_int8_calibration_dataloader(prefix),
                self._transform_fn,
                batch=0 if self.args.dynamic else self.args.batch,
                prefix=prefix,
            )
            source.unlink(missing_ok=True)
>>>>>>> origin/main
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """Export YOLO model to OpenVINO format."""
<<<<<<< HEAD
        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023"
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"
        ov_model = ov.convert_model(
            NMSModel(self.model, self.args) if self.args.nms else self.model,
            input=None if self.args.dynamic else [self.im.shape],
            example_input=self.im,
        )
=======
        from ultralytics.utils.export.openvino import torch2openvino

        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"
>>>>>>> origin/main

        def serialize(ov_model, file):
            """Set RT info, serialize, and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

<<<<<<< HEAD
            ov.save_model(ov_model, file, compress_to_fp16=self.args.half)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            # INT8 requires nncf, nncf requires packaging>=23.2 https://github.com/openvinotoolkit/nncf/issues/3463
            check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
            check_requirements("nncf>=2.14.0")
            import nncf

            def transform_fn(data_item) -> np.ndarray:
                """Quantization transform function."""
                data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
                assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
                im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0-255 to 0.0-1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im

            # Generate calibration data for integer quantization
            ignored_scope = None
=======
            ov.save_model(ov_model, file, compress_to_fp16=self.args.quantize == 16)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        calibration_dataset, ignored_scope = None, None
        if self.args.quantize == 8:
            check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
            check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
            import nncf

            calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)
>>>>>>> origin/main
            if isinstance(self.model.model[-1], Detect):
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # ignore operations
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
<<<<<<< HEAD
                        f".*{head_module_name}\\.dfl.*",
=======
>>>>>>> origin/main
                    ],
                    types=["Sigmoid"],
                )

<<<<<<< HEAD
            quantized_ov_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn),
                preset=nncf.QuantizationPreset.MIXED,
                ignored_scope=ignored_scope,
            )
            serialize(quantized_ov_model, fq_ov)
            return fq

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
=======
        ov_model = torch2openvino(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            dynamic=self.args.dynamic,
            quantize=self.args.quantize,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            prefix=prefix,
        )

        suffix = f"_{'int8_' if self.args.quantize == 8 else ''}openvino_model{os.sep}"
        f = str(self.file).replace(self.file.suffix, suffix)
>>>>>>> origin/main
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """Export YOLO model to PaddlePaddle format."""
<<<<<<< HEAD
        assert not IS_JETSON, "Jetson Paddle exports not supported yet"
        check_requirements(
            (
                "paddlepaddle-gpu"
                if torch.cuda.is_available()
                else "paddlepaddle==3.0.0"  # pin 3.0.0 for ARM64
                if ARM64
                else "paddlepaddle>=3.0.0",
                "x2paddle",
            )
        )
        import x2paddle
        from x2paddle.convert import pytorch2paddle

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # export
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f
=======
        from ultralytics.utils.export.paddle import torch2paddle

        return torch2paddle(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_litert(self, prefix=colorstr("LiteRT:")):
        """Export YOLO model to LiteRT format using litert_torch with optional INT8 quantization.

        Supports ``quantize=8`` (static INT8, int8 weights + int8 activations, requires calibration ``data``),
        ``quantize='w8a16'`` (static, int8 weights + int16 activations, requires calibration ``data``) and
        ``quantize='w8a32'`` (dynamic/weight-only INT8, int8 weights + FP32 activations, no calibration needed).
        """
        assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
        from ultralytics.utils.export.litert import torch2litert

        return torch2litert(
            self.model,
            self.im,
            self.file,
            quantize=self.args.quantize,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix)
            if self.args.quantize in {8, "w8a16"}
            else None,
            metadata=self.metadata,
            prefix=prefix,
        )
>>>>>>> origin/main

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
<<<<<<< HEAD
        assert TORCH_1_10, "MNN export requires torch>=1.10.0 to avoid segmentation faults"
        f_onnx = self.export_onnx()  # get onnx model first

        check_requirements("MNN>=2.9.6")
        import MNN
        from MNN.tools import mnnconvert

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = str(self.file.with_suffix(".mnn"))  # MNN model file
        args = ["", "-f", "ONNX", "--modelFile", f_onnx, "--MNNModel", f, "--bizCode", json.dumps(self.metadata)]
        if self.args.int8:
            args.extend(("--weightQuantBits", "8"))
        if self.args.half:
            args.append("--fp16")
        mnnconvert.convert(args)
        # remove scratch file for model convert optimize
        convert_scratch = Path(self.file.parent / ".__convert_external_data.bin")
        if convert_scratch.exists():
            convert_scratch.unlink()
        return f
=======
        from ultralytics.utils.export.mnn import onnx2mnn

        return onnx2mnn(
            onnx_file=self.export_onnx(),
            output_file=self.file.with_suffix(".mnn"),
            quantize=self.args.quantize,
            metadata=self.metadata,
            prefix=prefix,
        )
>>>>>>> origin/main

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
<<<<<<< HEAD
        check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
        check_requirements("pnnx")
        import ncnn
        import pnnx

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))

        ncnn_args = dict(
            ncnnparam=(f / "model.ncnn.param").as_posix(),
            ncnnbin=(f / "model.ncnn.bin").as_posix(),
            ncnnpy=(f / "model_ncnn.py").as_posix(),
        )

        pnnx_args = dict(
            ptpath=(f / "model.pt").as_posix(),
            pnnxparam=(f / "model.pnnx.param").as_posix(),
            pnnxbin=(f / "model.pnnx.bin").as_posix(),
            pnnxpy=(f / "model_pnnx.py").as_posix(),
            pnnxonnx=(f / "model.pnnx.onnx").as_posix(),
        )

        f.mkdir(exist_ok=True)  # make ncnn_model directory
        pnnx.export(self.model, inputs=self.im, **ncnn_args, **pnnx_args, fp16=self.args.half, device=self.device.type)

        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
            Path(f_debug).unlink(missing_ok=True)

        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        return str(f)

=======
        from ultralytics.utils.export.ncnn import torch2ncnn

        return torch2ncnn(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_ncnn_model/"),
            quantize=self.args.quantize,
            metadata=self.metadata,
            device=self.device,
            prefix=prefix,
        )

>>>>>>> origin/main
    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """Export YOLO model to CoreML format."""
        mlmodel = self.args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
<<<<<<< HEAD
        check_requirements(
            ["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"]
        )  # latest numpy 2.4.0rc1 breaks coremltools exports
        import coremltools as ct

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
=======
        from ultralytics.utils.export.coreml import IOSDetectModel, pipeline_coreml, torch2coreml

        # numpy 2.4.x breaks coremltools CoreML export https://github.com/apple/coremltools/issues/2633
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

>>>>>>> origin/main
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        if self.args.batch > 1:
            assert self.args.dynamic, (
                "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
            )
        if self.args.dynamic:
            assert not self.args.nms, (
                "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
            )
            assert self.model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

<<<<<<< HEAD
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values()))
            model = self.model
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo11n.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model
        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript model
=======
        # TODO CoreML Segment and Pose model pipelining; 'nms=True' is forced off for non-detect tasks upstream
        model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.args.nms else self.model
>>>>>>> origin/main

        if self.args.dynamic:
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[0] * 2, default=self.imgsz[0]),
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[1] * 2, default=self.imgsz[1]),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

<<<<<<< HEAD
        # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
        # Internally based on the model conversion and output type.
        # Setting minimum_deployment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
        # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
        ct_model = ct.convert(
            ts,
            inputs=inputs,
            classifier_config=classifier_config,
            convert_to="neuralnetwork" if mlmodel else "mlprogram",
        )
        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # scikit-learn package required for k-means quantization
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            elif bits == 8:  # mlprogram already quantized to FP16
                import coremltools.optimize.coreml as cto

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                ct_model = cto.palettize_weights(ct_model, config=config)
        if self.args.nms and self.model.task == "detect":
            ct_model = self._pipeline_coreml(ct_model, weights_dir=None if mlmodel else ct_model.weights_dir)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
=======
        ct_model = torch2coreml(
            model=model,
            inputs=inputs,
            im=self.im,
            classifier_names=list(self.model.names.values()) if self.model.task == "classify" else None,
            mlmodel=mlmodel,
            quantize=16 if self.args.nms and not mlmodel and self.args.quantize is None else self.args.quantize,
            metadata=self.metadata,
            prefix=prefix,
        )

        if self.args.nms:
            ct_model = pipeline_coreml(
                ct_model,
                self.output_shape,
                weights_dir=None if mlmodel else ct_model.weights_dir,
                metadata=self.metadata,
                mlmodel=mlmodel,
                iou=self.args.iou,
                conf=self.args.conf,
                agnostic_nms=self.args.agnostic_nms,
                prefix=prefix,
            )

>>>>>>> origin/main
        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
<<<<<<< HEAD
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt
        except ImportError:
            if LINUX:
                cuda_version = torch.version.cuda.split(".")[0]
                check_requirements(f"tensorrt-cu{cuda_version}>7.0.0,!=10.1.0")
            import tensorrt as trt
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
=======
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016
        from ultralytics.utils.export.engine import onnx2engine

>>>>>>> origin/main
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
<<<<<<< HEAD
            self.args.half,
            self.args.int8,
            self.args.dynamic,
            self.im.shape,
            dla=dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.int8 else None,
=======
            self.args.quantize,
            self.args.dynamic,
            self.im.shape,
            dla=self.dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.quantize == 8 else None,
>>>>>>> origin/main
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """Export YOLO model to TensorFlow SavedModel format."""
<<<<<<< HEAD
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf
        except ImportError:
            check_requirements("tensorflow>=2.0.0,<=2.19.0")
            import tensorflow as tf
        check_requirements(
            (
                "tf_keras<=2.19.0",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),  # required by 'onnx2tf' package
                "onnx>=1.12.0,<2.0.0",
                "onnx2tf>=1.26.3",
                "onnxslim>=0.1.71",
                "onnxruntime-gpu" if cuda else "onnxruntime",
                "protobuf>=5",
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # onnx_graphsurgeon only on NVIDIA
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )
=======
        assert not (MACOS and IS_PYTHON_MINIMUM_3_13), (
            "TensorFlow exports not supported on macOS with Python>=3.13: the ai-edge-litert macOS wheel fails to load "
            "(missing libpywrap_litert_common.dylib). TensorFlow export works on Linux Python 3.13."
        )
        from ultralytics.utils.export.tensorflow import onnx2saved_model

>>>>>>> origin/main
        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Export to TF
        images = None
<<<<<<< HEAD
        if self.args.int8 and self.args.data:
=======
        if self.args.quantize == 8 and self.args.data:
>>>>>>> origin/main
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.float32)
            )

        # Export to ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.args.opset = self.args.opset or 19
            assert 16 <= self.args.opset <= 19, "RTDETR export requires opset>=16;<=19"
        self.args.simplify = True
        f_onnx = self.export_onnx()  # ensure ONNX is available
        keras_model = onnx2saved_model(
            f_onnx,
            f,
<<<<<<< HEAD
            int8=self.args.int8,
            images=images,
            disable_group_convolution=self.args.format in {"tfjs", "edgetpu"},
=======
            quantize=self.args.quantize,
            images=images,
            disable_group_convolution=self.args.format == "edgetpu",
            cuda=self.device.type == "cuda",
>>>>>>> origin/main
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # or keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
<<<<<<< HEAD
        f = self.file.with_suffix(".pb")
        keras2pb(keras_model, f, prefix)
        return f

    @try_export
    def export_tflite(self, prefix=colorstr("TensorFlow Lite:")):
        """Export YOLO model to TensorFlow Lite format."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):
        """YOLO Axelera export."""
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        try:
            from axelera import compiler
        except ImportError:
            check_apt_requirements(
                ["libllvm14", "libgirepository1.0-dev", "pkg-config", "libcairo2-dev", "build-essential", "cmake"]
            )

            check_requirements(
                "axelera-voyager-sdk==1.5.2",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi "
                "--extra-index-url https://software.axelera.ai/artifactory/axelera-dev-pypi",
            )

        from axelera import compiler
        from axelera.compiler import CompilerConfig

        self.args.opset = 17
        onnx_path = self.export_onnx()
        model_name = f"{Path(onnx_path).stem}"
        export_path = Path(f"{model_name}_axelera_model")
        export_path.mkdir(exist_ok=True)

        def transform_fn(data_item) -> np.ndarray:
            data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
            assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
            im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
            return np.expand_dims(im, 0) if im.ndim == 3 else im

        if "C2PSA" in self.model.__str__():  # YOLO11
            config = CompilerConfig(
                quantization_scheme="per_tensor_min_max",
                ignore_weight_buffers=False,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
                model_name=model_name,
            )
        else:  # YOLOv8
            config = CompilerConfig(
                tiling_depth=6,
                split_buffer_promotion=True,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
                model_name=model_name,
            )

        qmodel = compiler.quantize(
            model=onnx_path,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            config=config,
            transform_fn=transform_fn,
        )

        compiler.compile(model=qmodel, config=config, output_dir=export_path)

        axm_name = f"{model_name}.axm"
        axm_src = Path(axm_name)
        axm_dst = export_path / axm_name

        if axm_src.exists():
            axm_src.replace(axm_dst)

        YAML.save(export_path / "metadata.yaml", self.metadata)

        return export_path

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        """Exports a model to ExecuTorch (.pte) format into a dedicated directory and saves the required metadata,
        following Ultralytics conventions.
        """
        LOGGER.info(f"\n{prefix} starting export with ExecuTorch...")
        assert TORCH_2_9, f"ExecuTorch export requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        # TorchAO release compatibility table bug https://github.com/pytorch/ao/issues/2919
        # Setuptools bug: https://github.com/pypa/setuptools/issues/4483
        check_requirements("setuptools<71.0.0")  # Setuptools bug: https://github.com/pypa/setuptools/issues/4483
        check_requirements(("executorch==1.0.1", "flatbuffers"))

        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        file_directory = Path(str(self.file).replace(self.file.suffix, "_executorch_model"))
        file_directory.mkdir(parents=True, exist_ok=True)

        file_pte = file_directory / self.file.with_suffix(".pte").name
        sample_inputs = (self.im,)

        et_program = to_edge_transform_and_lower(
            torch.export.export(self.model, sample_inputs), partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        with open(file_pte, "wb") as file:
            file.write(et_program.buffer)

        YAML.save(file_directory / "metadata.yaml", self.metadata)

        return str(file_directory)
=======
        from ultralytics.utils.export.tensorflow import keras2pb

        return keras2pb(keras_model, output_file=self.file.with_suffix(".pb"), prefix=prefix)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):
        """Export YOLO model to Axelera format."""
        assert LINUX and not (ARM64 and IS_DOCKER), (
            "export is only supported on Linux and is not supported on ARM64 Docker."
        )
        assert TORCH_2_8, "export requires torch>=2.8.0."

        from ultralytics.utils.export.axelera import torch2axelera

        output_dir = self.file.parent / f"{self.file.stem}_axelera_model"
        return torch2axelera(
            model=self.model,
            output_dir=output_dir,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            model_name=self.file.stem,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        """Export YOLO model to ExecuTorch *.pte format."""
        assert TORCH_2_9, f"ExecuTorch requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        from ultralytics.utils.export.executorch import torch2executorch

        return torch2executorch(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_executorch_model/"),
            metadata=self.metadata,
            prefix=prefix,
        )
>>>>>>> origin/main

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
<<<<<<< HEAD
        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            sudo = "sudo " if is_sudo_available() else ""
            for c in (
                f"{sudo}mkdir -p /etc/apt/keyrings",
                f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | {sudo}gpg --dearmor -o /etc/apt/keyrings/google.gpg",
                f'echo "deb [signed-by=/etc/apt/keyrings/google.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | {sudo}tee /etc/apt/sources.list.d/coral-edgetpu.list',
            ):
                subprocess.run(c, shell=True, check=True)
            check_apt_requirements(["edgetpu-compiler"])

        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().rsplit(maxsplit=1)[-1]
        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model
        self._add_tflite_metadata(f)
        return f

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """Export YOLO model to TensorFlow.js format."""
        check_requirements("tensorflowjs")

        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path
        pb2tfjs(pb_file=f_pb, output_dir=f, half=self.args.half, int8=self.args.int8, prefix=prefix)
        # Add metadata
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """Export YOLO model to RKNN format."""
        LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")

        check_requirements("rknn-toolkit2")
        if IS_COLAB:
            # Prevent 'exit' from closing the notebook https://github.com/airockchip/rknn-toolkit2/issues/259
            import builtins

            builtins.exit = lambda: None

        from rknn.api import RKNN

        f = self.export_onnx()
        export_path = Path(f"{Path(f).stem}_rknn_model")
        export_path.mkdir(exist_ok=True)

        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=self.args.name)
        rknn.load_onnx(model=f)
        rknn.build(do_quantization=False)  # TODO: Add quantization support
        f = f.replace(".onnx", f"-{self.args.name}.rknn")
        rknn.export_rknn(f"{export_path / f}")
        YAML.save(export_path / "metadata.yaml", self.metadata)
        return export_path
=======
        from ultralytics.utils.export.tensorflow import tflite2edgetpu

        output_file = tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        self._add_tflite_metadata(output_file)
        return output_file

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """Export YOLO model to RKNN format with optional INT8 quantization."""
        from ultralytics.utils.export.rknn import onnx2rknn

        self.args.opset = min(self.args.opset or 19, 19)  # rknn-toolkit expects opset<=19
        self.im = self.im[:1]  # RKNN Toolkit expands the batch after calibrating the batch-1 ONNX model
        f_onnx = self.export_onnx()
        output_dir = Path(str(self.file).replace(self.file.suffix, f"_rknn_model{os.sep}"))
        rknn_dataset = None
        if self.args.quantize == 8:
            dataloader = self.get_int8_calibration_dataloader(prefix)
            image_paths = getattr(dataloader.dataset, "im_files", None)
            if image_paths is None and hasattr(dataloader.dataset, "samples"):
                image_paths = [x[0] for x in dataloader.dataset.samples]
            if not image_paths:
                raise ValueError("RKNN INT8 export requires a calibration dataset with image file paths.")
            output_dir.mkdir(parents=True, exist_ok=True)
            rknn_dataset = output_dir / "dataset.txt"
            rknn_dataset.write_text("\n".join(str(Path(x).resolve()) for x in image_paths) + "\n")
        return onnx2rknn(
            onnx_file=f_onnx,
            output_dir=output_dir,
            name=self.args.name,
            quantize=self.args.quantize,
            batch=self.args.batch,
            dataset=rknn_dataset,
            metadata=self.metadata,
            prefix=prefix,
        )
>>>>>>> origin/main

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """Export YOLO model to IMX format."""
        assert LINUX, (
            "Export only supported on Linux."
            "See https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter?version=3.17.3&progLang="
        )
<<<<<<< HEAD
        assert not ARM64, "IMX export is not supported on ARM64 architectures."
=======
>>>>>>> origin/main
        assert IS_PYTHON_MINIMUM_3_9, "IMX export is only supported on Python 3.9 or above."

        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
<<<<<<< HEAD
        check_requirements(
            (
                "model-compression-toolkit>=2.4.1",
                "edge-mdt-cl<1.1.0",
                "edge-mdt-tpc>=1.2.0",
                "pydantic<=2.11.7",
            )
        )

        check_requirements("imx500-converter[pt]>=3.17.3")

        # Install Java>=17
        try:
            java_output = subprocess.run(["java", "--version"], check=True, capture_output=True).stdout.decode()
            version_match = re.search(r"(?:openjdk|java) (\d+)", java_output)
            java_version = int(version_match.group(1)) if version_match else 0
            assert java_version >= 17, "Java version too old"
        except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
            if IS_UBUNTU or IS_DEBIAN_TRIXIE:
                LOGGER.info(f"\n{prefix} installing Java 21 for Ubuntu...")
                check_apt_requirements(["openjdk-21-jre"])
            elif IS_RASPBERRYPI or IS_DEBIAN_BOOKWORM:
                LOGGER.info(f"\n{prefix} installing Java 17 for Raspberry Pi or Debian ...")
                check_apt_requirements(["openjdk-17-jre"])

        return torch2imx(
            self.model,
            self.file,
            self.args.conf,
            self.args.iou,
            self.args.max_det,
            metadata=self.metadata,
            dataset=self.get_int8_calibration_dataloader(prefix),
            prefix=prefix,
        )

=======
        from ultralytics.utils.export.imx import torch2imx

        return torch2imx(
            model=self.model,
            output_dir=str(self.file).replace(self.file.suffix, "_imx_model/"),
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            metadata=self.metadata,
            dataset=partial(self.get_int8_calibration_dataloader, prefix),
            prefix=prefix,
        )

    @try_export
    def export_deepx(self, prefix=colorstr("DEEPX:")):
        """Export YOLO model to DEEPX format."""
        assert LINUX and not ARM64, "DEEPX export only supported on non-aarch64 Linux"
        from ultralytics.utils.export.deepx import onnx2deepx

        f = self.export_onnx()
        return onnx2deepx(
            onnx_file=f,
            imgsz=self.imgsz,
            dataset=self.get_int8_calibration_dataloader(prefix),
            metadata=self.metadata,
            optimize=self.args.optimize,
            prefix=prefix,
        )

    @try_export
    def export_qnn(self, prefix=colorstr("Qualcomm QNN:")):
        """Export YOLO model to a Qualcomm QNN context binary using ONNX Runtime QNN."""
        from ultralytics.utils.export.qnn import onnx2qnn

        # Wrap for Hexagon-friendly I/O: channel-last input (the class-map wrap for semantic is format-agnostic)
        model, im = self.model, self.im
        try:
            self.model, self.im = QNNModel(model), im.permute(0, 2, 3, 1)
            f_onnx = self.export_onnx()
        finally:
            self.model, self.im = model, im
        return onnx2qnn(
            onnx_file=f_onnx,
            output_file=str(self.file.with_name(f"{self.file.stem}_qnn.onnx")),
            dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            name=self.args.name,
            metadata=self.metadata,
            batch=0 if self.args.dynamic else self.args.batch,
            prefix=prefix,
        )

    @try_export
    def export_hailo(self, prefix=colorstr("Hailo:")):
        """Export a YOLO model to Hailo Executable Format (HEF)."""
        try:
            import tensorflow as tf
            from hailo_sdk_client import ClientRunner
        except ImportError as e:
            raise ImportError("Hailo export requires the Hailo Dataflow Compiler.") from e

        calibration_dataloader = self.get_int8_calibration_dataloader(prefix)
        calibration_size = len(calibration_dataloader.dataset)
        LOGGER.warning(
            f"\nHailo level-2 optimization will use {calibration_size} calibration images. "
            "Hailo recommends at least 1,024 representative images for best accuracy. "
            'Pass data="path/to/dataset.yaml". '
            "See https://docs.ultralytics.com/integrations/hailo/#export-a-hailo-hef-model"
        )
        head_index = len(self.model.model) - 1
        head = self.model.model[head_index]
        one2one = getattr(self.model, "end2end", False)
        task = self.model.task
        if task == "classify":
            # The Classify head ends in Gemm -> Softmax; cut at the Softmax so the HEF returns the same
            # (1, nc) probabilities as the PyTorch model. The DFC translates the softmax to a native layer.
            end_nodes = [f"/model.{head_index}/Softmax"]
        else:
            scales = range(len(head.one2one_cv2 if one2one else head.cv2))
            if one2one:
                end_nodes = [
                    f"/model.{head_index}/one2one_cv{branch}.{i}/one2one_cv{branch}.{i}.2/Conv"
                    for branch in (2, 3)
                    for i in scales
                ]
            elif task in {"segment", "pose", "obb"}:
                # reg/cls/extra triple per scale (extra = mask coeffs, keypoints, or angle); segment adds prototypes.
                end_nodes = [
                    f"/model.{head_index}/cv{branch}.{i}/cv{branch}.{i}.2/Conv" for i in scales for branch in (2, 3, 4)
                ]
                if task == "segment":
                    end_nodes.append(f"/model.{head_index}/proto/cv3/act/Mul")
            else:
                end_nodes = [
                    f"/model.{head_index}/cv{branch}.{i}/cv{branch}.{i}.2/Conv" for i in scales for branch in (2, 3)
                ]
        self.args.opset = 11
        f_onnx = Path(self.export_onnx())
        output_dir = self.file.parent / f"{self.file.stem}_hailo_model"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            runner = ClientRunner(hw_arch=self.args.name)
            runner.translate_onnx_model(str(f_onnx), self.file.stem, end_node_names=end_nodes)
            model_script = [
                "normalization1 = normalization([0, 0, 0], [255, 255, 255])",
                "model_optimization_flavor(optimization_level=2)",
                f"post_quantization_optimization(finetune, policy=enabled, dataset_size={calibration_size})",
            ]
            if one2one:
                outputs = ", ".join(f"output_layer{i + 1}" for i in range(len(end_nodes)))
                model_script.append(f"quantization_param([{outputs}], precision_mode=a16_w16)")
            elif task == "classify":
                pass  # softmax is already the graph output; no NMS or activation changes needed
            else:
                outputs = [layer.inputs[0].rsplit("/", 1)[-1] for layer in runner.get_hn_model().get_output_layers()]
                if task in {"segment", "pose", "obb"}:
                    # Bake sigmoid into the class convs only (position 1 of each per-scale reg/cls/extra triple).
                    # Mask coeffs, prototypes, keypoints and angles stay raw and are decoded on the host.
                    model_script.extend(
                        f"change_output_activation({outputs[i]}, sigmoid)" for i in range(1, 3 * len(scales), 3)
                    )
                else:
                    nms_config = output_dir / "nms_config.json"
                    nms_config.write_text(
                        json.dumps(
                            {
                                "nms_scores_th": self.args.conf if self.args.conf is not None else 0.25,
                                "nms_iou_th": self.args.iou,
                                "image_dims": self.imgsz,
                                "max_proposals_per_class": 100,
                                "classes": len(self.model.names),
                                "regression_length": 16,
                                "background_removal": False,
                                "background_removal_index": 0,
                                "bbox_decoders": [
                                    {
                                        "name": f"bbox_decoder_{stride}",
                                        "stride": stride,
                                        "reg_layer": outputs[i * 2],
                                        "cls_layer": outputs[i * 2 + 1],
                                    }
                                    for i, stride in enumerate(int(x) for x in head.stride)
                                ],
                            },
                            indent=2,
                        )
                    )
                    model_script.extend(
                        f"change_output_activation({outputs[i]}, sigmoid)" for i in range(1, len(outputs), 2)
                    )
                    model_script.append(f'nms_postprocess("{nms_config}", meta_arch=yolov8, engine=cpu)')
                    model_script.append("allocator_param(width_splitter_defuse=disabled)")
            runner.load_model_script("\n".join(model_script))

            def calibration_dataset():
                for batch in calibration_dataloader:
                    for image in batch["img"].permute(0, 2, 3, 1).numpy().astype(np.float32):
                        yield image, {}

            runner.optimize(
                lambda: tf.data.Dataset.from_generator(
                    calibration_dataset,
                    output_signature=(tf.TensorSpec(shape=(*self.imgsz, 3), dtype=tf.float32), {}),
                )
            )
            (output_dir / f"{self.file.stem}.hef").write_bytes(runner.compile())
            YAML.save(
                output_dir / "metadata.yaml",
                {**self.metadata, "hailo_arch": self.args.name, "nms": task == "detect" and not one2one},
            )
            return str(output_dir)
        finally:
            f_onnx.unlink(missing_ok=True)

>>>>>>> origin/main
    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

<<<<<<< HEAD
    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        """Create CoreML pipeline with NMS for YOLO detection models."""
        import coremltools as ct

        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")

        # Output shapes
        spec = model.get_spec()
        outs = list(iter(spec.description.output))
        if self.args.format == "mlmodel":  # mlmodel doesn't infer shapes automatically
            outs[0].type.multiArrayType.shape[:] = self.output_shape[2], self.output_shape[1] - 4
            outs[1].type.multiArrayType.shape[:] = self.output_shape[2], 4

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        nc = outs[0].type.multiArrayType.shape[-1]
        if len(names) != nc:  # Hack fix for MLProgram NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            names = {**names, **{i: str(i) for i in range(len(names), nc)}}

        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = spec.specificationVersion
        for i in range(len(outs)):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        output_names = ["confidence", "coordinates"]
        for i, name in enumerate(output_names):
            nms_spec.description.output[i].name = name

        for i, out in enumerate(outs):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = out.type.multiArrayType.shape[-1]
            ma_type.shapeRange.sizeRanges[1].upperBound = out.type.multiArrayType.shape[-1]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
        nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
        nms.confidenceOutputFeatureName = output_names[0]
        nms.coordinatesOutputFeatureName = output_names[1]
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = self.args.iou
        nms.confidenceThreshold = self.args.conf
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=output_names,
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = spec.specificationVersion
        pipeline.spec.description.metadata.userDefined.update(
            {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
        )

        # Save the model
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
        model.input_description["image"] = "Input image"
        model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
        model.input_description["confidenceThreshold"] = (
            f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
        )
        model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
        LOGGER.info(f"{prefix} pipeline success")
        return model
=======
    @staticmethod
    def _transform_fn(data_item) -> np.ndarray:
        """Quantization preprocessing transform for INT8 calibration (Axelera, OpenVINO, ONNX, QNN)."""
        data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
        assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
        im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
        return im[None] if im.ndim == 3 else im
>>>>>>> origin/main

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


<<<<<<< HEAD
class IOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model, im, mlprogram=True):
        """Initialize the IOSDetectModel class with a YOLO model and example image.

        Args:
            model (torch.nn.Module): The YOLO model to wrap.
            im (torch.Tensor): Example input tensor with shape (B, C, H, W).
            mlprogram (bool): Whether exporting to MLProgram format to fix NMS bug.
        """
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        self.mlprogram = mlprogram
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],  # broadcast (slower, smaller)
                device=next(model.parameters()).device,
            )

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # pad class length to multiple of 80
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)

        return cls, xywh * self.normalize
=======
class ExportWrapper(torch.nn.Module):
    """Base for export-time model wrappers: stores the wrapped model and forwards attribute lookups.

    Subclasses adapt a fused YOLO model's inference I/O for a specific deployment contract (layout, output
    reduction) while the exporter keeps interacting with the wrapper as if it were the model itself.
    """

    def __init__(self, model):
        """Wrap a fused YOLO `model` prepared for export."""
        super().__init__()
        # Stored under a private name so attribute forwarding resolves `wrapper.model` to the wrapped model's own
        # `model` (its nn.Sequential), keeping exporter code like `self.model.model[-1]` working unchanged.
        self._model = model
        self.task = model.task

    def __getattr__(self, name):
        """Forward attribute lookups (model, names, stride, yaml, args, ...) to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


class QNNModel(ExportWrapper):
    """Wraps a YOLO model with channel-last inference input for Qualcomm QNN export.

    Traced by the standard ONNX export (`export_qnn` swaps it in with a channel-last dummy input). The graph takes `[N,
    H, W, C]` images - the Hexagon HTP's native layout and what camera pipelines produce - so ONNX Runtime's layout
    transformer folds the wrapper's Transpose into the NPU partition during context generation, and neither the NPU
    (boundary transpose) nor the consuming app (CPU-side permute) pays a per-inference layout cost.

    Attributes:
        task (str): The wrapped model's task, forwarded for the ONNX export plumbing.
    """

    def forward(self, x):
        """Run inference on channel-last `[N, H, W, C]` input normalized to [0, 1]."""
        return self._model(x.permute(0, 3, 1, 2))  # the wrapped model is NCHW; the transpose folds into the NPU graph


class ClassMapModel(ExportWrapper):
    """Reduces semantic-segmentation logits to a compact integer class map for export.

    Applied to QNN and Core ML semantic exports, where the argmax runs on the NPU: deployment consumers want per-pixel
    class indices, and shipping float logits instead forces a dequantize + argmax over large tensors (~20M values at
    1024px) on the consumer's CPU every frame - measured as both slow and highly variable on mobile
    NPUs. The argmax cannot live in the model's own forward because it is non-differentiable (training needs
    logits), so it is attached here at export time, mirroring how `NMSModel` adds suppression only for export.

    Attributes:
        task (str): The wrapped model's task ("semantic").
        dtype (torch.dtype): Class-index dtype; uint8 unless the model has more than 256 classes.
    """

    def __init__(self, model):
        """Wrap a fused semantic `model` so export emits class indices instead of logits."""
        super().__init__(model)
        # uint8 quarters the NPU->CPU output transfer vs int32 and Core ML promotes it to int32 in-spec;
        # int32 only when more than 256 classes make uint8 indices ambiguous.
        self.dtype = torch.uint8 if len(model.names) <= 256 else torch.int32

    def forward(self, x):
        """Run the wrapped model and return a `[N, H, W]` integer class map instead of float logits."""
        y = self._model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        # Single-channel (binary) models threshold the logit, matching predict/val semantics for nc == 1
        return (y.argmax(1) if y.shape[1] > 1 else y[:, 0].gt(0)).to(self.dtype)
>>>>>>> origin/main


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model, args):
        """Initialize the NMSModel.

        Args:
            model (torch.nn.Module): The model to wrap with NMS postprocessing.
<<<<<<< HEAD
            args (Namespace): The export arguments.
=======
            args (SimpleNamespace): The export arguments.
>>>>>>> origin/main
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
<<<<<<< HEAD
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})
=======
        self.is_tf = self.args.format == "saved_model"
>>>>>>> origin/main

    def forward(self, x):
        """Perform inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
<<<<<<< HEAD
            x (torch.Tensor): The preprocessed tensor with shape (N, 3, H, W).

        Returns:
            (torch.Tensor): List of detections, each an (N, max_det, 4 + 2 + extra_shape) Tensor where N is the number
                of detections after NMS.
        """
        from functools import partial

=======
            x (torch.Tensor): The preprocessed tensor with shape (B, C, H, W).

        Returns:
            (torch.Tensor | tuple): Tensor of shape (B, max_det, 4 + 2 + extra_shape) where B is the batch size, or a
                tuple of (detections, proto) for segmentation models.
        """
>>>>>>> origin/main
        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # extras from Segment, OBB, Pose
        if self.args.dynamic and self.args.batch > 1:  # batch size needs to always be same due to loop unroll
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # in case num_anchors < max_det
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
<<<<<<< HEAD
            if self.args.format == "tflite":  # TFLite is already normalized
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
=======
            nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
>>>>>>> origin/main
            if not self.args.agnostic_nms:  # class-wise NMS
                end = 2 if self.obb else 4
                # fully explicit expansion otherwise reshape error
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
<<<<<<< HEAD
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 error with triu
=======
                        or (self.args.format == "openvino" and self.args.quantize == 8)  # OpenVINO INT8 error with triu
>>>>>>> origin/main
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]

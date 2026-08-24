---
comments: true
<<<<<<< HEAD
description: Learn to export YOLO11 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLO11, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
=======
description: Learn to export YOLO26 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLO26, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
>>>>>>> origin/main
---

# Intel OpenVINO Export

<<<<<<< HEAD
<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

In this guide, we cover exporting YOLO11 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.
=======
<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ecosystem.avif" alt="OpenVINO Intel AI inference toolkit">

In this guide, we cover exporting YOLO26 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.
>>>>>>> origin/main

OpenVINO, short for Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AvFh-oTGDaw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Export Ultralytics YOLO11 to Intel OpenVINO Format for Faster Inference 🚀
=======
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to Intel OpenVINO Format for Faster Inference 🚀
>>>>>>> origin/main
</p>

## Usage Examples

<<<<<<< HEAD
Export a YOLO11n model to OpenVINO format and run inference with the exported model.

!!! example
=======
The OpenVINO format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy on Intel CPU, integrated/discrete GPU, or NPU.

!!! example "Export"
>>>>>>> origin/main

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo11n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
=======
        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to OpenVINO format
        model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'

        # Run inference with the exported model
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
=======
        # Export a YOLO26n PyTorch model to OpenVINO format
        yolo export model=yolo26n.pt format=openvino # creates 'yolo26n_openvino_model/'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Run inference on a specific device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported OpenVINO model
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference on a specific device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported OpenVINO model
        yolo val model=yolo26n_openvino_model data=coco8.yaml
>>>>>>> origin/main
        ```

## Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'openvino'`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
<<<<<<< HEAD
| `half`     | `bool`           | `False`        | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                                                                                     |
| `int8`     | `bool`           | `False`        | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices.                                                                    |
| `dynamic`  | `bool`           | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`           | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
=======
| `quantize` | `int` or `str`   | `None`         | Quantization precision: `16` (FP16) or `8` (INT8/PTQ; needs calibration `data`/`fraction`); `32`/unset is FP32. Replaces the deprecated `half`/`int8` flags.                                                                                                     |
| `dynamic`  | `bool`           | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`           | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](../datasets/index.md) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                                              |
>>>>>>> origin/main
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

!!! warning

    OpenVINO™ is compatible with most Intel® processors but to ensure optimal performance:

    1. Verify OpenVINO™ support
        Check whether your Intel® chip is officially supported by OpenVINO™ using [Intel's compatibility list](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).

    2. Identify your accelerator
        Determine if your processor includes an integrated NPU (Neural Processing Unit) or GPU (integrated GPU) by consulting [Intel's hardware guide](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html).

    3. Install the latest drivers
        If your chip supports an NPU or GPU but OpenVINO™ isn't detecting it, you may need to install or update the associated drivers. Follow the [driver‑installation instructions](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) to enable full acceleration.

    By following these three steps, you can ensure OpenVINO™ runs optimally on your Intel® hardware.

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) frameworks such as PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
<<<<<<< HEAD
4. **Ease of Use**: The toolkit comes with more than [80 tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLOv8 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)) teaching different aspects of the toolkit.
=======
4. **Ease of Use**: The toolkit comes with a large collection of [tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLO26 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization)) teaching different aspects of the toolkit.
>>>>>>> origin/main

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once your model is successfully exported to the OpenVINO format, you have two primary options for running inference:

1. Use the `ultralytics` package, which provides a high-level API and wraps the OpenVINO Runtime.

2. Use the native `openvino` package for more advanced or customized control over inference behavior.

### Inference with Ultralytics

The ultralytics package allows you to easily run inference using the exported OpenVINO model via the predict method. You can also specify the target device (e.g., `intel:gpu`, `intel:npu`, `intel:cpu`) using the device argument.

```python
from ultralytics import YOLO

# Load the exported OpenVINO model
<<<<<<< HEAD
ov_model = YOLO("yolo11n_openvino_model/")  # the path of your exported OpenVINO model
=======
ov_model = YOLO("yolo26n_openvino_model/")  # the path of your exported OpenVINO model
>>>>>>> origin/main
# Run inference with the exported model
ov_model.predict(device="intel:gpu")  # specify the device you want to run inference on
```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

<<<<<<< HEAD
The OpenVINO Runtime provides a unified API for inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running inference, refer to the [YOLO11 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization).
=======
The OpenVINO Runtime provides a unified API for inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running inference, refer to the [YOLO26 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization).
>>>>>>> origin/main

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

<<<<<<< HEAD
## OpenVINO YOLO11 Benchmarks

The Ultralytics team benchmarked YOLO11 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

!!! note

    The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    All benchmarks run with `openvino` Python package version [2025.1.0](https://pypi.org/project/openvino/2025.1.0/).

### Intel Core CPU

The Intel® Core® series is a range of high-performance processors by Intel. The lineup includes Core i3 (entry-level), Core i5 (mid-range), Core i7 (high-end), and Core i9 (extreme performance). Each series caters to different computing needs and budgets, from everyday tasks to demanding professional workloads. With each new generation, improvements are made to performance, energy efficiency, and features.

Benchmarks below run on 12th Gen Intel® Core® i9-12900KS CPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-corei9.avif" alt="Core CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch     | ✅     | 5.4       | 0.5071              | 21.00                  |
    | YOLO11n | TorchScript | ✅     | 10.5      | 0.5077              | 21.39                  |
    | YOLO11n | ONNX        | ✅     | 10.2      | 0.5077              | 15.55                  |
    | YOLO11n | OpenVINO    | ✅     | 10.4      | 0.5077              | 11.49                  |
    | YOLO11s | PyTorch     | ✅     | 18.4      | 0.5770              | 43.16                  |
    | YOLO11s | TorchScript | ✅     | 36.6      | 0.5781              | 50.06                  |
    | YOLO11s | ONNX        | ✅     | 36.3      | 0.5781              | 31.53                  |
    | YOLO11s | OpenVINO    | ✅     | 36.4      | 0.5781              | 30.82                  |
    | YOLO11m | PyTorch     | ✅     | 38.8      | 0.6257              | 110.60                 |
    | YOLO11m | TorchScript | ✅     | 77.3      | 0.6306              | 128.09                 |
    | YOLO11m | ONNX        | ✅     | 76.9      | 0.6306              | 76.06                  |
    | YOLO11m | OpenVINO    | ✅     | 77.1      | 0.6306              | 79.38                  |
    | YOLO11l | PyTorch     | ✅     | 49.0      | 0.6367              | 150.38                 |
    | YOLO11l | TorchScript | ✅     | 97.7      | 0.6408              | 172.57                 |
    | YOLO11l | ONNX        | ✅     | 97.0      | 0.6408              | 108.91                 |
    | YOLO11l | OpenVINO    | ✅     | 97.3      | 0.6408              | 102.30                 |
    | YOLO11x | PyTorch     | ✅     | 109.3     | 0.6989              | 272.72                 |
    | YOLO11x | TorchScript | ✅     | 218.1     | 0.6900              | 320.86                 |
    | YOLO11x | ONNX        | ✅     | 217.5     | 0.6900              | 196.20                 |
    | YOLO11x | OpenVINO    | ✅     | 217.8     | 0.6900              | 195.32                 |
=======
## OpenVINO YOLO26 Benchmarks

The Ultralytics team benchmarked YOLO26 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

!!! note

    - The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    - All benchmarks were run with `openvino` Python package version [2026.2.1](https://pypi.org/project/openvino/2026.2.1).

    - YOLO26 models on NPU are only supported on Intel® Core™ Ultra™ systems with 2xxV series and 3xx series and above.
>>>>>>> origin/main

### Intel® Core™ Ultra

The Intel® Core™ Ultra™ series represents a new benchmark in high-performance computing, engineered to meet the evolving demands of modern users—from gamers and creators to professionals leveraging AI. This next-generation lineup is more than a traditional CPU series; it combines powerful CPU cores, integrated high-performance GPU capabilities, and a dedicated Neural Processing Unit (NPU) within a single chip, offering a unified solution for diverse and intensive computing workloads.

At the heart of the Intel® Core Ultra™ architecture is a hybrid design that enables exceptional performance across traditional processing tasks, GPU-accelerated workloads, and AI-driven operations. The inclusion of the NPU enhances on-device AI inference, enabling faster, more efficient machine learning and data processing across a wide range of applications.

The Core Ultra™ family includes various models tailored for different performance needs, with options ranging from energy-efficient designs to high-power variants marked by the "H" designation—ideal for laptops and compact form factors that demand serious computing power. Across the lineup, users benefit from the synergy of CPU, GPU, and NPU integration, delivering remarkable efficiency, responsiveness, and multitasking capabilities.

As part of Intel's ongoing innovation, the Core Ultra™ series sets a new standard for future-ready computing. With multiple models available and more on the horizon, this series underscores Intel's commitment to delivering cutting-edge solutions for the next generation of intelligent, AI-enhanced devices.

<<<<<<< HEAD
Benchmarks below run on Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 265K at FP32 and INT8 precision.
=======
Benchmarks below run on Intel® Core™ Ultra™ X7 358H, Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 155H at FP32, FP16 and INT8 precision.

#### Intel® Core™ Ultra™ X7 358H

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultraX7-358H-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 25.18                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4763              | 2.67                   |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4763              | 2.64                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4625              | 2.73                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 50.09                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5615              | 3.57                   |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.5615              | 3.55                   |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.547               | 3.09                   |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 135.1                  |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.3      | 0.6168              | 5.64                   |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6168              | 5.85                   |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.5      | 0.5994              | 4.14                   |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 169.75                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6206              | 8.14                   |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6206              | 8.18                   |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5999              | 4.67                   |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 407.56                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6569              | 13.11                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6569              | 13.15                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6374              | 9.24                   |

    === "Intel® Panther Lake CPU"

        <div align="center">
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultraX7-358H-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch       | FP32      | ✅     | 5.3       | 0.4765              | 25.18                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4734              | 19.66                  |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4771              | 19.6                   |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4652              | 8.55                   |
            | YOLO26s | PyTorch       | FP32      | ✅     | 19.5      | 0.5703              | 50.09                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5632              | 56.99                  |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.563               | 56.75                  |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5491              | 27.26                  |
            | YOLO26m | PyTorch       | FP32      | ✅     | 42.2      | 0.6196              | 135.1                  |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.3      | 0.6191              | 169.83                 |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.618               | 168.95                 |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.5      | 0.6038              | 54.98                  |
            | YOLO26l | PyTorch       | FP32      | ✅     | 50.7      | 0.6215              | 169.75                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6206              | 213.85                 |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.621               | 213.05                 |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.6028              | 70.71                  |
            | YOLO26x | PyTorch       | FP32      | ✅     | 113.2     | 0.6512              | 407.56                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6552              | 499.71                 |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6552              | 498.27                 |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6445              | 140.67                 |

    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultraX7-358H-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅     | 5.3       | 0.4765              | 25.18                  |
            | YOLO26n | OpenVINO      | FP32      | ✅     | 9.6       | 0.4751              | 10.98                  |
            | YOLO26n | OpenVINO      | FP16      | ✅     | 5.1       | 0.4763              | 10.45                  |
            | YOLO26n | OpenVINO      | INT8      | ✅     | 3.2       | 0.4724              | 11.23                  |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅     | 19.5      | 0.5703              | 50.09                  |
            | YOLO26s | OpenVINO      | FP32      | ✅     | 36.7      | 0.5616              | 11.39                  |
            | YOLO26s | OpenVINO      | FP16      | ✅     | 18.6      | 0.5617              | 11.23                  |
            | YOLO26s | OpenVINO      | INT8      | ✅     | 10.0      | 0.5498              | 12.15                  |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅     | 42.2      | 0.6196              | 135.1                  |
            | YOLO26m | OpenVINO      | FP32      | ✅     | 78.3      | 0.6171              | 14.42                  |
            | YOLO26m | OpenVINO      | FP16      | ✅     | 39.5      | 0.6167              | 14.42                  |
            | YOLO26m | OpenVINO      | INT8      | ✅     | 20.5      | 0.6026              | 15.53                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅     | 50.7      | 0.6215              | 169.75                 |
            | YOLO26l | OpenVINO      | FP32      | ✅     | 95.3      | 0.6201              | 16.18                  |
            | YOLO26l | OpenVINO      | FP16      | ✅     | 48.1      | 0.6201              | 16.38                  |
            | YOLO26l | OpenVINO      | INT8      | ✅     | 25.2      | 0.5997              | 16.46                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅     | 113.2     | 0.6512              | 407.56                 |
            | YOLO26x | OpenVINO      | FP32      | ✅     | 213.2     | 0.6563              | 25.49                  |
            | YOLO26x | OpenVINO      | FP16      | ✅     | 107.1     | 0.6563              | 25.53                  |
            | YOLO26x | OpenVINO      | INT8      | ✅     | 54.8      | 0.6409              | 23.23                  |
>>>>>>> origin/main

#### Intel® Core™ Ultra™ 7 258V

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
<<<<<<< HEAD
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
=======
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-258V-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
>>>>>>> origin/main
        </div>

        ??? abstract "Detailed Benchmark Results"

<<<<<<< HEAD
            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5068              | 11.84                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4969              | 11.24                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5797              | 14.82                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 12.88                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 22.94                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 17.85                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6365              | 27.34                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6242              | 20.83                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6890              | 39.09                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6856              | 30.60                  |
=======
            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅      | 5.3       | 0.4765              | 31.43                  |
            | YOLO26n | OpenVINO      | FP32      | ✅      | 9.6       | 0.4762              | 3.57                   |
            | YOLO26n | OpenVINO      | FP16      | ✅      | 5.1       | 0.4762              | 3.53                   |
            | YOLO26n | OpenVINO      | INT8      | ✅      | 3.2       | 0.4625              | 3.65                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅      | 19.5      | 0.5703              | 60.4                   |
            | YOLO26s | OpenVINO      | FP32      | ✅      | 36.7      | 0.5616              | 5.02                   |
            | YOLO26s | OpenVINO      | FP16      | ✅      | 18.6      | 0.5616              | 5.01                   |
            | YOLO26s | OpenVINO      | INT8      | ✅      | 10.0      | 0.547               | 4.31                   |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅      | 42.2      | 0.6196              | 173.31                 |
            | YOLO26m | OpenVINO      | FP32      | ✅      | 78.3      | 0.6191              | 9.48                   |
            | YOLO26m | OpenVINO      | FP16      | ✅      | 39.5      | 0.6168              | 9.6                    |
            | YOLO26m | OpenVINO      | INT8      | ✅      | 20.5      | 0.5994              | 6.03                   |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅      | 50.7      | 0.6173              | 224.52                 |
            | YOLO26l | OpenVINO      | FP32      | ✅      | 95.3      | 0.3725              | 11.88                  |
            | YOLO26l | OpenVINO      | FP16      | ✅      | 48.1      | 0.6201              | 12.0                   |
            | YOLO26l | OpenVINO      | INT8      | ✅      | 25.2      | 0.5999              | 8.47                   |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅      | 113.2     | 0.6512              | 595.72                 |
            | YOLO26x | OpenVINO      | FP32      | ✅      | 213.2     | 0.6567              | 20.26                  |
            | YOLO26x | OpenVINO      | FP16      | ✅      | 107.1     | 0.6454              | 20.25                  |
            | YOLO26x | OpenVINO      | INT8      | ✅      | 54.8      | 0.6374              | 14.77                  |
>>>>>>> origin/main

    === "Intel® Lunar Lake CPU"

        <div align="center">
<<<<<<< HEAD
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
=======
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-258V-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
>>>>>>> origin/main
        </div>

        ??? abstract "Detailed Benchmark Results"

<<<<<<< HEAD
            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 32.55                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 22.98                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 98.38                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 52.84                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 275.74                 |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 132.63                 |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 171.36                 |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 783.16                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 346.82                 |

=======
            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch       | FP32      | ✅      | 5.3       | 0.4765              | 31.43                  |
            | YOLO26n | OpenVINO      | FP32      | ✅      | 9.6       | 0.4734              | 17.04                  |
            | YOLO26n | OpenVINO      | FP16      | ✅      | 5.1       | 0.4771              | 17.0                   |
            | YOLO26n | OpenVINO      | INT8      | ✅      | 3.2       | 0.4652              | 8.84                   |
            | YOLO26s | PyTorch       | FP32      | ✅      | 19.5      | 0.5703              | 60.4                   |
            | YOLO26s | OpenVINO      | FP32      | ✅      | 36.7      | 0.5632              | 54.25                  |
            | YOLO26s | OpenVINO      | FP16      | ✅      | 18.6      | 0.563               | 54.24                  |
            | YOLO26s | OpenVINO      | INT8      | ✅      | 10.0      | 0.5491              | 20.47                  |
            | YOLO26m | PyTorch       | FP32      | ✅      | 42.2      | 0.6196              | 173.31                 |
            | YOLO26m | OpenVINO      | FP32      | ✅      | 78.3      | 0.6191              | 166.36                 |
            | YOLO26m | OpenVINO      | FP16      | ✅      | 39.5      | 0.618               | 167.2                  |
            | YOLO26m | OpenVINO      | INT8      | ✅      | 20.5      | 0.6038              | 52.89                  |
            | YOLO26l | PyTorch       | FP32      | ✅      | 50.7      | 0.6173              | 224.52                 |
            | YOLO26l | OpenVINO      | FP32      | ✅      | 95.3      | 0.6206              | 216.26                 |
            | YOLO26l | OpenVINO      | FP16      | ✅      | 48.1      | 0.621               | 217.08                 |
            | YOLO26l | OpenVINO      | INT8      | ✅      | 25.2      | 0.6028              | 66.98                  |
            | YOLO26x | PyTorch       | FP32      | ✅      | 113.2     | 0.6512              | 595.72                 |
            | YOLO26x | OpenVINO      | FP32      | ✅      | 213.2     | 0.6552              | 541.35                 |
            | YOLO26x | OpenVINO      | FP16      | ✅      | 107.1     | 0.6552              | 537.96                 |
            | YOLO26x | OpenVINO      | INT8      | ✅      | 54.8      | 0.6445              | 138.26                 |
>>>>>>> origin/main

    === "Integrated Intel® AI Boost NPU"

        <div align="center">
<<<<<<< HEAD
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-npu.avif" alt="Intel Core Ultra NPU benchmarks">
=======
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-258V-npu.avif" alt="Intel Core Ultra NPU benchmarks">
>>>>>>> origin/main
        </div>

        ??? abstract "Detailed Benchmark Results"

<<<<<<< HEAD
            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5085              | 8.33                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.5019              | 8.91                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5788              | 9.72                   |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5710              | 10.58                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6301              | 19.41                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6124              | 18.26                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6362              | 23.70                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 21.40                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6892              | 43.91                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 34.04                  |

#### Intel® Core™ Ultra™ 7 265K
=======
            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅      | 5.3       | 0.4765              | 31.43                  |
            | YOLO26n | OpenVINO      | FP32      | ✅      | 9.6       | 0.4729              | 8.82                   |
            | YOLO26n | OpenVINO      | FP16      | ✅      | 5.1       | 0.477               | 7.98                   |
            | YOLO26n | OpenVINO      | INT8      | ✅      | 3.2       | 0.4591              | 8.17                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅      | 19.5      | 0.5703              | 60.4                   |
            | YOLO26s | OpenVINO      | FP32      | ✅      | 36.7      | 0.5617              | 9.79                   |
            | YOLO26s | OpenVINO      | FP16      | ✅      | 18.6      | 0.5618              | 9.74                   |
            | YOLO26s | OpenVINO      | INT8      | ✅      | 10.0      | 0.5487              | 11.88                  |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅      | 42.2      | 0.6196              | 173.31                 |
            | YOLO26m | OpenVINO      | FP32      | ✅      | 78.3      | 0.6166              | 16.74                  |
            | YOLO26m | OpenVINO      | FP16      | ✅      | 39.5      | 0.6166              | 16.8                   |
            | YOLO26m | OpenVINO      | INT8      | ✅      | 20.5      | 0.6041              | 15.68                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅      | 50.7      | 0.6173              | 224.52                 |
            | YOLO26l | OpenVINO      | FP32      | ✅      | 95.3      | 0.6199              | 19.33                  |
            | YOLO26l | OpenVINO      | FP16      | ✅      | 48.1      | 0.6201              | 19.21                  |
            | YOLO26l | OpenVINO      | INT8      | ✅      | 25.2      | 0.6003              | 17.25                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅      | 113.2     | 0.6512              | 595.72                 |
            | YOLO26x | OpenVINO      | FP32      | ✅      | 213.2     | 0.5764              | 32.84                  |
            | YOLO26x | OpenVINO      | FP16      | ✅      | 107.1     | 0.6563              | 32.87                  |
            | YOLO26x | OpenVINO      | INT8      | ✅      | 54.8      | 0.6463              | 25.88                  |

#### Intel® Core™ Ultra™ 7 155H
>>>>>>> origin/main

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
<<<<<<< HEAD
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
=======
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-155H-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
>>>>>>> origin/main
        </div>

        ??? abstract "Detailed Benchmark Results"

<<<<<<< HEAD
            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5079              | 13.13                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4976              | 8.86                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5808              | 18.26                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5726              | 13.24                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 43.50                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6137              | 20.90                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6371              | 54.52                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6226              | 27.36                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6884              | 112.76                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6900              | 52.06                  |


    === "Intel® Arrow Lake CPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
=======
            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch (CPU) | FP32      | ✅      | 5.3       | 0.4765              | 38.77                  |
            | YOLO26n | OpenVINO      | FP32      | ✅      | 9.6       | 0.4774              | 9.87                   |
            | YOLO26n | OpenVINO      | FP16      | ✅      | 5.1       | 0.4774              | 9.84                   |
            | YOLO26n | OpenVINO      | INT8      | ✅      | 3.2       | 0.4705              | 5.86                   |
            | YOLO26s | PyTorch (CPU) | FP32      | ✅      | 19.5      | 0.5703              | 69.54                  |
            | YOLO26s | OpenVINO      | FP32      | ✅      | 36.7      | 0.5616              | 17.29                  |
            | YOLO26s | OpenVINO      | FP16      | ✅      | 18.6      | 0.5616              | 17.06                  |
            | YOLO26s | OpenVINO      | INT8      | ✅      | 10.0      | 0.5452              | 10.33                  |
            | YOLO26m | PyTorch (CPU) | FP32      | ✅      | 42.2      | 0.6196              | 192.22                 |
            | YOLO26m | OpenVINO      | FP32      | ✅      | 78.3      | 0.6187              | 34.64                  |
            | YOLO26m | OpenVINO      | FP16      | ✅      | 39.5      | 0.6187              | 34.75                  |
            | YOLO26m | OpenVINO      | INT8      | ✅      | 20.5      | 0.6073              | 15.99                  |
            | YOLO26l | PyTorch (CPU) | FP32      | ✅      | 50.7      | 0.6215              | 245.62                 |
            | YOLO26l | OpenVINO      | FP32      | ✅      | 95.3      | 0.6202              | 43.7                   |
            | YOLO26l | OpenVINO      | FP16      | ✅      | 48.1      | 0.6202              | 44.65                  |
            | YOLO26l | OpenVINO      | INT8      | ✅      | 25.2      | 0.6048              | 20.31                  |
            | YOLO26x | PyTorch (CPU) | FP32      | ✅      | 113.2     | 0.6512              | 513.06                 |
            | YOLO26x | OpenVINO      | FP32      | ✅      | 213.2     | 0.6544              | 80.19                  |
            | YOLO26x | OpenVINO      | FP16      | ✅      | 107.1     | 0.6544              | 79.83                  |
            | YOLO26x | OpenVINO      | INT8      | ✅      | 54.8      | 0.6393              | 35.16                  |

    === "Intel® Meteor Lake CPU"

        <div align="center">
        <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-155H-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
>>>>>>> origin/main
        </div>

        ??? abstract "Detailed Benchmark Results"

<<<<<<< HEAD
            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 15.04                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 11.60                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 33.45                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 20.64                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 81.15                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 44.63                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6409              | 103.77                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 58.00                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 208.37                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6897              | 113.04                 |


    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5075              | 8.02                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.3656              | 9.28                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5801              | 13.12                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5686              | 13.12                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 29.88                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6111              | 26.32                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6356              | 37.08                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6245              | 30.81                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6894              | 68.48                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6417              | 49.76                  |

## Intel® Arc GPU

Intel® Arc™ is Intel's line of discrete graphics cards designed for high-performance gaming, content creation, and AI workloads. The Arc series features advanced GPU architectures that support real-time ray tracing, AI-enhanced graphics, and high-resolution gaming. With a focus on performance and efficiency, Intel® Arc™ aims to compete with other leading GPU brands while providing unique features like hardware-accelerated AV1 encoding and support for the latest graphics APIs.

Benchmarks below run on Intel Arc A770 and Intel Arc B580 at FP32 and INT8 precision.

### Intel Arc A770

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-a770-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5073              | 6.98                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4978              | 7.24                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5798              | 9.41                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 8.72                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6311              | 14.88                  |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 11.97                  |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6364              | 19.17                  |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6241              | 15.75                  |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6888              | 18.13                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6930              | 18.91                  |

### Intel Arc B580

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-b580-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5072              | 4.27                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4981              | 4.33                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5789              | 5.04                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5746              | 4.97                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 6.45                   |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6125              | 6.28                   |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6360              | 8.23                   |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6236              | 8.49                   |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6889              | 11.10                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6924              | 10.30                  |
=======
            | Model   | Format        | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | ------------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO26n | PyTorch       | FP32      | ✅      | 5.3       | 0.4765              | 33.54                  |
            | YOLO26n | OpenVINO      | FP32      | ✅      | 9.6       | 0.4734              | 13.47                  |
            | YOLO26n | OpenVINO      | FP16      | ✅      | 5.1       | 0.4771              | 13.45                  |
            | YOLO26n | OpenVINO      | INT8      | ✅      | 3.2       | 0.4597              | 11.79                  |
            | YOLO26s | PyTorch       | FP32      | ✅      | 19.5      | 0.5703              | 75.81                  |
            | YOLO26s | OpenVINO      | FP32      | ✅      | 36.7      | 0.5632              | 36.4                   |
            | YOLO26s | OpenVINO      | FP16      | ✅      | 18.6      | 0.563               | 36.12                  |
            | YOLO26s | OpenVINO      | INT8      | ✅      | 10.0      | 0.5469              | 15.75                  |
            | YOLO26m | PyTorch       | FP32      | ✅      | 42.2      | 0.6196              | 212.82                 |
            | YOLO26m | OpenVINO      | FP32      | ✅      | 78.3      | 0.6191              | 102.35                 |
            | YOLO26m | OpenVINO      | FP16      | ✅      | 39.5      | 0.618               | 101.87                 |
            | YOLO26m | OpenVINO      | INT8      | ✅      | 20.5      | 0.6038              | 35.97                  |
            | YOLO26l | PyTorch       | FP32      | ✅      | 50.7      | 0.6215              | 273.41                 |
            | YOLO26l | OpenVINO      | FP32      | ✅      | 95.3      | 0.6206              | 129.12                 |
            | YOLO26l | OpenVINO      | FP16      | ✅      | 48.1      | 0.621               | 128.31                 |
            | YOLO26l | OpenVINO      | INT8      | ✅      | 25.2      | 0.5997              | 45.55                  |
            | YOLO26x | PyTorch       | FP32      | ✅      | 113.2     | 0.6512              | 560.87                 |
            | YOLO26x | OpenVINO      | FP32      | ✅      | 213.2     | 0.6552              | 287.93                 |
            | YOLO26x | OpenVINO      | FP16      | ✅      | 107.1     | 0.6552              | 285.98                 |
            | YOLO26x | OpenVINO      | INT8      | ✅      | 54.8      | 0.6455              | 85.32                  |
>>>>>>> origin/main

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
=======
        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
>>>>>>> origin/main
        results = model.benchmark(data="coco128.yaml")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml
=======
        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo26n.pt data=coco128.yaml
>>>>>>> origin/main
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Conclusion

<<<<<<< HEAD
The benchmarking results clearly demonstrate the benefits of exporting the YOLO11 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.
=======
The benchmarking results clearly demonstrate the benefits of exporting the YOLO26 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.
>>>>>>> origin/main

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

## FAQ

<<<<<<< HEAD
### How do I export YOLO11 models to OpenVINO format?

Exporting YOLO11 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:
=======
### How do I export YOLO26 models to OpenVINO format?

Exporting YOLO26 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'
=======
        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'
=======
        # Export a YOLO26n PyTorch model to OpenVINO format
        yolo export model=yolo26n.pt format=openvino # creates 'yolo26n_openvino_model/'
>>>>>>> origin/main
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

<<<<<<< HEAD
### What are the benefits of using OpenVINO with YOLO11 models?

Using Intel's OpenVINO toolkit with YOLO11 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: Over 80 tutorial notebooks are available to help users get started, including ones for YOLO11.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolo11-benchmarks).

### How can I run inference using a YOLO11 model exported to OpenVINO?

After exporting a YOLO11n model to OpenVINO format, you can run inference using Python or CLI:
=======
### What are the benefits of using OpenVINO with YOLO26 models?

Using Intel's OpenVINO toolkit with YOLO26 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: A large collection of tutorial notebooks is available to help users get started, including ones for YOLO26.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolo26-benchmarks).

### How can I run inference using a YOLO26 model exported to OpenVINO?

After exporting a YOLO26n model to OpenVINO format, you can run inference using Python or CLI:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
<<<<<<< HEAD
        ov_model = YOLO("yolo11n_openvino_model/")
=======
        ov_model = YOLO("yolo26n_openvino_model/")
>>>>>>> origin/main

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
<<<<<<< HEAD
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'
=======
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg'
>>>>>>> origin/main
        ```

Refer to our [predict mode documentation](../modes/predict.md) for more details.

<<<<<<< HEAD
### Why should I choose Ultralytics YOLO11 over other models for OpenVINO export?

Ultralytics YOLO11 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLO11 provides:
=======
### Why should I choose Ultralytics YOLO26 over other models for OpenVINO export?

Ultralytics YOLO26 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLO26 provides:
>>>>>>> origin/main

- Up to 3x speedup on Intel CPUs
- Seamless deployment on Intel GPUs and NPUs
- Consistent and comparable accuracy across various export formats

<<<<<<< HEAD
For in-depth performance analysis, check our detailed [YOLO11 benchmarks](#openvino-yolo11-benchmarks) on different hardware.

### Can I benchmark YOLO11 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLO11 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:
=======
For in-depth performance analysis, check our detailed [YOLO26 benchmarks](#openvino-yolo26-benchmarks) on different hardware.

### Can I benchmark YOLO26 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLO26 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
=======
        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Benchmark YOLO26n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
>>>>>>> origin/main
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolo11-benchmarks) and [export formats](../modes/export.md) documentation.
=======
        # Benchmark YOLO26n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolo26n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolo26-benchmarks) and [export formats](../modes/export.md) documentation.
>>>>>>> origin/main

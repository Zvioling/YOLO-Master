---
<<<<<<< HEAD
comments: true
description: Export YOLO11 models to ExecuTorch format for efficient on-device inference on mobile and edge devices. Optimize your AI models for iOS, Android, and embedded systems.
keywords: Ultralytics, YOLO11, ExecuTorch, model export, PyTorch, edge AI, mobile deployment, on-device inference, XNNPACK, embedded systems
---

# Deploy YOLO11 on Mobile & Edge with ExecuTorch
=======
title: YOLO26 ExecuTorch for Mobile & Edge
comments: true
description: Export YOLO26 models to ExecuTorch format for efficient on-device inference on mobile and edge devices. Optimize your AI models for iOS, Android, and embedded systems.
keywords: Ultralytics, YOLO26, ExecuTorch, model export, PyTorch, edge AI, mobile deployment, on-device inference, XNNPACK, embedded systems
---

# Deploy YOLO26 on Mobile & Edge with ExecuTorch
>>>>>>> origin/main

Deploying computer vision models on edge devices like smartphones, tablets, and embedded systems requires an optimized runtime that balances performance with resource constraints. ExecuTorch, PyTorch's solution for edge computing, enables efficient on-device inference for [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models.

This guide outlines how to export Ultralytics YOLO models to ExecuTorch format, enabling you to deploy your models on mobile and edge devices with optimized performance.

## Why export to ExecuTorch?

<p align="center">
<<<<<<< HEAD
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/executorch-pipeline.avif" alt="PyTorch ExecuTorch overview">
=======
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/executorch-pipeline.avif" alt="PyTorch ExecuTorch mobile inference framework">
>>>>>>> origin/main
</p>

[ExecuTorch](https://docs.pytorch.org/executorch/) is PyTorch's end-to-end solution for enabling on-device inference capabilities across mobile and edge devices. Built with the goal of being portable and efficient, ExecuTorch can be used to run PyTorch programs on a wide variety of computing platforms.

## Key features of ExecuTorch

ExecuTorch provides several powerful features for deploying Ultralytics YOLO models on edge devices:

- **Portable Model Format**: ExecuTorch uses the `.pte` (PyTorch ExecuTorch) format, which is optimized for size and loading speed on resource-constrained devices.

- **XNNPACK Backend**: Default integration with XNNPACK provides highly optimized inference on mobile CPUs, delivering excellent performance without requiring specialized hardware.

<<<<<<< HEAD
- **Quantization Support**: Built-in support for quantization techniques to reduce model size and improve inference speed while maintaining accuracy.
=======
- **Quantization Ready**: The ExecuTorch ecosystem supports quantization techniques to reduce model size and improve inference speed; Ultralytics currently exports FP32 models via the XNNPACK backend.
>>>>>>> origin/main

- **Memory Efficiency**: Optimized memory management reduces runtime memory footprint, making it suitable for devices with limited RAM.

- **Model Metadata**: Exported models include metadata (image size, class names, etc.) in a separate YAML file for easy integration.

## Deployment Options with ExecuTorch

ExecuTorch models can be deployed across various edge and mobile platforms:

- **Mobile Applications**: Deploy on iOS and Android applications with native performance, enabling real-time object detection in mobile apps.

- **Embedded Systems**: Run on embedded Linux devices like Raspberry Pi, NVIDIA Jetson, and other ARM-based systems with optimized performance.

- **Edge AI Devices**: Deploy on specialized edge AI hardware with custom delegates for accelerated inference.

- **IoT Devices**: Integrate into IoT devices for on-device inference without cloud connectivity requirements.

<<<<<<< HEAD
## Exporting Ultralytics YOLO11 Models to ExecuTorch

Converting Ultralytics YOLO11 models to ExecuTorch format enables efficient deployment on mobile and edge devices.

### Installation

ExecuTorch export requires Python 3.10 or higher and specific dependencies:
=======
## Exporting Ultralytics YOLO26 Models to ExecuTorch

Converting Ultralytics YOLO26 models to ExecuTorch format enables efficient deployment on mobile and edge devices.

### Installation

ExecuTorch export requires Python 3.10-3.13 and PyTorch >= 2.9.0 along with the `executorch` package:
>>>>>>> origin/main

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install Ultralytics package
        pip install ultralytics
        ```

<<<<<<< HEAD
For detailed instructions and best practices related to the installation process, check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Exporting YOLO11 models to ExecuTorch is straightforward:

!!! example "Usage"
=======
For detailed instructions and best practices related to the installation process, check our [YOLO26 Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Exporting YOLO26 models to ExecuTorch is straightforward:

The ExecuTorch format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"
>>>>>>> origin/main

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to ExecuTorch format
        model.export(format="executorch")  # creates 'yolo11n_executorch_model' directory

        executorch_model = YOLO("yolo11n_executorch_model")

        results = executorch_model.predict("https://ultralytics.com/images/bus.jpg")
=======
        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ExecuTorch format
        model.export(format="executorch")  # creates 'yolo26n_executorch_model'
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Export a YOLO11n PyTorch model to ExecuTorch format
        yolo export model=yolo11n.pt format=executorch # creates 'yolo11n_executorch_model' directory

        # Run inference with the exported model
        yolo predict model=yolo11n_executorch_model source=https://ultralytics.com/images/bus.jpg
=======
        # Export a YOLO26n PyTorch model to ExecuTorch format
        yolo export model=yolo26n.pt format=executorch # creates 'yolo26n_executorch_model'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ExecuTorch model
        yolo predict model=yolo26n_executorch_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ExecuTorch model
        yolo val model=yolo26n_executorch_model data=coco8.yaml
>>>>>>> origin/main
        ```

    ExecuTorch exports generate a directory that includes a `.pte` file and metadata. Use the ExecuTorch runtime in your mobile or embedded application to load the `.pte` model and perform inference.

### Export Arguments

When exporting to ExecuTorch format, you can specify the following arguments:

<<<<<<< HEAD
| Argument | Type            | Default | Description                                |
| -------- | --------------- | ------- | ------------------------------------------ |
| `imgsz`  | `int` or `list` | `640`   | Image size for model input (height, width) |
| `device` | `str`           | `'cpu'` | Device to use for export (`'cpu'`)         |
=======
| Argument   | Type             | Default        | Description                                                                                                                             |
| ---------- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'executorch'` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `quantize` | `int` or `str`   | `None`         | Fixed FP32 export. ExecuTorch export does not support export-time FP16, INT8, or W8A16 precision conversion.                            |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device`   | `str`            | `None`         | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                         |
>>>>>>> origin/main

### Output Structure

The ExecuTorch export creates a directory containing the model and metadata:

```text
<<<<<<< HEAD
yolo11n_executorch_model/
├── yolo11n.pte              # ExecuTorch model file
└── metadata.yaml            # Model metadata (classes, image size, etc.)
=======
yolo26n_executorch_model/
├── model.pte               # ExecuTorch model file
└── metadata.yaml           # Model metadata (classes, image size, etc.)
>>>>>>> origin/main
```

## Using Exported ExecuTorch Models

After exporting your model, you'll need to integrate it into your target application using the ExecuTorch runtime.

### Mobile Integration

For mobile applications (iOS/Android), you'll need to:

1. **Add ExecuTorch Runtime**: Include the ExecuTorch runtime library in your mobile project
2. **Load Model**: Load the `.pte` file in your application
3. **Run Inference**: Process images and get predictions

Example iOS integration (Objective-C/C++):

```objc
// iOS uses C++ APIs for model loading and inference
// See https://pytorch.org/executorch/stable/using-executorch-ios.html for complete examples

#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

// Load the model
<<<<<<< HEAD
Module module("/path/to/yolo11n.pte");
=======
Module module("/path/to/model.pte");
>>>>>>> origin/main

// Create input tensor
float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

// Run inference
const auto result = module.forward(tensor);
```

Example Android integration (Kotlin):

```kotlin
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

// Load the model
<<<<<<< HEAD
val module = Module.load("/path/to/yolo11n.pte")
=======
val module = Module.load("/path/to/model.pte")
>>>>>>> origin/main

// Prepare input tensor
val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 640, 640))
val inputEValue = EValue.from(inputTensor)

// Run inference
val outputs = module.forward(inputEValue)
val scores = outputs[0].toTensor().dataAsFloatArray
```

### Embedded Linux

For embedded Linux systems, use the ExecuTorch C++ API:

```cpp
#include <executorch/extension/module/module.h>
<<<<<<< HEAD

// Load model
auto module = torch::executor::Module("yolo11n.pte");

// Prepare input
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = torch::executor::Tensor(input_data, {1, 3, 640, 640});

// Run inference
auto outputs = module.forward({input_tensor});
=======
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Load model
Module module("model.pte");

// Prepare input
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = from_blob(input_data.data(), {1, 3, 640, 640});

// Run inference
const auto outputs = module.forward(input_tensor);
>>>>>>> origin/main
```

For more details on integrating ExecuTorch into your applications, visit the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/).

## Performance Optimization

### Model Size Optimization

To reduce model size for deployment:

<<<<<<< HEAD
- **Use Smaller Models**: Start with YOLO11n (nano) for the smallest footprint
=======
- **Use Smaller Models**: Start with YOLO26n (nano) for the smallest footprint
>>>>>>> origin/main
- **Lower Input Resolution**: Use smaller image sizes (e.g., `imgsz=320` or `imgsz=416`)
- **Quantization**: Apply quantization techniques (supported in future ExecuTorch versions)

### Inference Speed Optimization

For faster inference:

- **XNNPACK Backend**: The default XNNPACK backend provides optimized CPU inference
- **Hardware Acceleration**: Use platform-specific delegates (e.g., CoreML for iOS)
- **Batch Processing**: Process multiple images when possible

## Benchmarks

<<<<<<< HEAD
The Ultralytics team benchmarked YOLO11 models, comparing speed and accuracy between PyTorch and ExecuTorch.
=======
The Ultralytics team benchmarked YOLO26 models, comparing speed and accuracy between PyTorch and ExecuTorch.
>>>>>>> origin/main

!!! tip "Performance"

    === "Raspberry Pi 5"

        | Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
<<<<<<< HEAD
        | YOLO11n | PyTorch     | ✅     | 5.4       | 0.5060              | 337.67                 |
        | YOLO11n | ExecuTorch  | ✅     | 11        | 0.5080              | 167.28                 |
        | YOLO11s | PyTorch     | ✅     | 19        | 0.5770              |  928.80                |
        | YOLO11s | ExecuTorch  | ✅     | 37        | 0.5780              | 388.31                 |

    === "More devices coming soon!"

=======
        | YOLO26n | PyTorch     | ✅     | 5.3       | 0.4790              | 314.80                  |
        | YOLO26n | ExecuTorch  | ✅     | 9.4        | 0.4800              | 142                    |
        | YOLO26s | PyTorch     | ✅     | 19.5       | 0.5730             | 930.90                 |
        | YOLO26s | ExecuTorch  | ✅     | 36.5        | 0.5780              | 376.1                 |

    === "More devices coming soon!"

    Benchmarked with Ultralytics 8.4.9

>>>>>>> origin/main
    !!! note

        Inference time does not include pre/ post-processing.

## Troubleshooting

### Common Issues

**Issue**: `Python version error`

**Solution**: ExecuTorch requires Python 3.10 or higher. Upgrade your Python installation:

```bash
# Using conda
conda create -n executorch python=3.10
conda activate executorch
```

**Issue**: `Export fails during first run`

<<<<<<< HEAD
**Solution**: ExecuTorch may need to download and compile components on first use. Ensure you have:
=======
**Solution**: Ensure you have the latest prebuilt `executorch` wheel installed:
>>>>>>> origin/main

```bash
pip install --upgrade executorch
```

**Issue**: `Import errors for ExecuTorch modules`

**Solution**: Ensure ExecuTorch is properly installed:

```bash
pip install executorch --force-reinstall
```

For more troubleshooting help, visit the [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/getting-started-setup.html).

## Summary

<<<<<<< HEAD
Exporting YOLO11 models to ExecuTorch format enables efficient deployment on mobile and edge devices. With PyTorch-native integration, cross-platform support, and optimized performance, ExecuTorch is an excellent choice for edge AI applications.
=======
Exporting YOLO26 models to ExecuTorch format enables efficient deployment on mobile and edge devices. With PyTorch-native integration, cross-platform support, and optimized performance, ExecuTorch is an excellent choice for edge AI applications.
>>>>>>> origin/main

Key takeaways:

- ExecuTorch provides PyTorch-native edge deployment with excellent performance
- Export is simple with `format='executorch'` parameter
- Models are optimized for mobile CPUs via XNNPACK backend
- Supports iOS, Android, and embedded Linux platforms
<<<<<<< HEAD
- Requires Python 3.10+ and FlatBuffers compiler

## FAQ

### How do I export a YOLO11 model to ExecuTorch format?

Export a YOLO11 model to ExecuTorch using either Python or CLI:
=======
- Requires Python 3.10-3.13 and PyTorch >= 2.9.0

## FAQ

### How do I export a YOLO26 model to ExecuTorch format?

Export a YOLO26 model to ExecuTorch using either Python or CLI:
>>>>>>> origin/main

```python
from ultralytics import YOLO

<<<<<<< HEAD
model = YOLO("yolo11n.pt")
=======
model = YOLO("yolo26n.pt")
>>>>>>> origin/main
model.export(format="executorch")
```

or

```bash
<<<<<<< HEAD
yolo export model=yolo11n.pt format=executorch
=======
yolo export model=yolo26n.pt format=executorch
>>>>>>> origin/main
```

### What are the system requirements for ExecuTorch export?

ExecuTorch export requires:

- Python 3.10 or higher
- `executorch` package (install via `pip install executorch`)
- PyTorch (installed automatically with ultralytics)

<<<<<<< HEAD
Note: During the first export, ExecuTorch will download and compile necessary components including the FlatBuffers compiler automatically.

### Can I run inference with ExecuTorch models directly in Python?

ExecuTorch models (`.pte` files) are designed for deployment on mobile and edge devices using the ExecuTorch runtime. They cannot be directly loaded with `YOLO()` for inference in Python. You need to integrate them into your target application using the ExecuTorch runtime libraries.
=======
Note: The `executorch` package ships prebuilt wheels (with the XNNPACK backend), so no extra compilation step is required during export.

### Can I run inference with ExecuTorch models directly in Python?

ExecuTorch models can be loaded directly with `YOLO()` for inference and validation in Python (see the Predict/Validate examples above), and they can also be deployed on mobile and edge devices using the ExecuTorch runtime libraries.
>>>>>>> origin/main

### What platforms are supported by ExecuTorch?

ExecuTorch supports:

- **Mobile**: iOS and Android
- **Embedded Linux**: Raspberry Pi, NVIDIA Jetson, and other ARM devices
- **Desktop**: Linux, macOS, and Windows (for development)

<<<<<<< HEAD
### How does ExecuTorch compare to TFLite for mobile deployment?

Both ExecuTorch and TFLite are excellent for mobile deployment:

- **ExecuTorch**: Better PyTorch integration, native PyTorch workflow, growing ecosystem
- **TFLite**: More mature, wider hardware support, more deployment examples

Choose ExecuTorch if you're already using PyTorch and want a native deployment path. Choose TFLite for maximum compatibility and mature tooling.
=======
### How does ExecuTorch compare to LiteRT for mobile deployment?

Both ExecuTorch and [LiteRT](litert.md) are excellent for mobile deployment:

- **ExecuTorch**: Better PyTorch integration, native PyTorch workflow, growing ecosystem
- **LiteRT**: More mature, wider hardware support, more deployment examples, and runs the same model on Android, iOS, and the browser

Choose ExecuTorch if you're already using PyTorch and want a native deployment path. Choose LiteRT for maximum compatibility and mature tooling.
>>>>>>> origin/main

### Can I use ExecuTorch models with GPU acceleration?

Yes! ExecuTorch supports hardware acceleration through various backends:

- **Mobile GPU**: Via Vulkan, Metal, or OpenCL delegates
- **NPU/DSP**: Via platform-specific delegates
- **Default**: XNNPACK for optimized CPU inference

Refer to the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html) for backend-specific setup.

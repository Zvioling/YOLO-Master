---
comments: true
<<<<<<< HEAD
description: Learn how to export YOLO11 models to CoreML for optimized, on-device machine learning on iOS and macOS. Follow step-by-step instructions.
keywords: CoreML export, YOLO11 models, CoreML conversion, Ultralytics, iOS object detection, macOS machine learning, AI deployment, machine learning integration
---

# CoreML Export for YOLO11 Models

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on Apple devices like iPhones and Macs requires a format that ensures seamless performance.
=======
description: Export Ultralytics YOLO26 models to CoreML for fast on-device inference on the Apple Neural Engine across iPhone, iPad, and Mac. Step-by-step export, quantization, and deployment guide.
keywords: CoreML export, Core ML, YOLO26 CoreML, Apple Neural Engine, ANE, mlpackage, ML Program, Vision framework, iOS object detection, macOS machine learning, Swift, quantization, INT8, FP16, on-device AI, Ultralytics
---

# CoreML Export for YOLO26 Models

Apple ships dedicated AI silicon — the Neural Engine — in every modern iPhone, iPad, and Mac, and [CoreML](https://developer.apple.com/documentation/coreml) is Ultralytics' supported path for deploying models to it today. Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to CoreML turns a trained `.pt` checkpoint into a native `.mlpackage` that runs all seven YOLO tasks on-device at low latency, with no network connection and no data leaving the device.

!!! tip "Run YOLO on the Apple Neural Engine today with the official mobile apps"

    The official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) run CoreML exports on the Apple Neural Engine out of the box — real-time camera inference, single-image prediction, and automatic model download for all seven YOLO26 tasks, including Depth. For Android NPU deployment, see the [Qualcomm QNN integration](qnn.md).

!!! note "Apple's future Core AI format"

    Apple has introduced the new [Core AI framework and `.aimodel` format](coreai.md) for the iOS 27 and macOS 27 generation. Ultralytics Core AI export is planned for Q4 2026 but is not available yet. CoreML remains the supported format for current Ultralytics releases and broader Apple device compatibility.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hfSK3Mk5P0I"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Export Ultralytics YOLO11 to CoreML for 2x Fast Inference on Apple Devices 🚀
</p>

The CoreML export format allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for efficient [object detection](https://www.ultralytics.com/glossary/object-detection) in iOS and macOS applications. In this guide, we'll walk you through the steps for converting your models to the CoreML format, making it easier for your models to perform well on Apple devices.

## CoreML

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-overview.avif" alt="CoreML Overview">
</p>

[CoreML](https://developer.apple.com/documentation/coreml) is Apple's foundational machine learning framework that builds upon Accelerate, BNNS, and Metal Performance Shaders. It provides a machine-learning model format that seamlessly integrates into iOS applications and supports tasks such as image analysis, [natural language processing](https://www.ultralytics.com/glossary/natural-language-processing-nlp), audio-to-text conversion, and sound analysis.

Applications can take advantage of Core ML without the need to have a network connection or API calls because the Core ML framework works using on-device computing. This means model inference can be performed locally on the user's device.

## Key Features of CoreML Models

Apple's CoreML framework offers robust features for on-device machine learning. Here are the key features that make CoreML a powerful tool for developers:

- **Comprehensive Model Support**: Converts and runs models from popular frameworks like TensorFlow, [PyTorch](https://www.ultralytics.com/glossary/pytorch), scikit-learn, XGBoost, and LibSVM.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-supported-models.avif" alt="CoreML Supported Models">
</p>

- **On-device [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml)**: Ensures data privacy and swift processing by executing models directly on the user's device, eliminating the need for network connectivity.

- **Performance and Optimization**: Uses the device's CPU, GPU, and Neural Engine for optimal performance with minimal power and memory usage. Offers tools for model compression and optimization while maintaining [accuracy](https://www.ultralytics.com/glossary/accuracy).

- **Ease of Integration**: Provides a unified format for various model types and a user-friendly API for seamless integration into apps. Supports domain-specific tasks through frameworks like Vision and Natural Language.

- **Advanced Features**: Includes on-device training capabilities for personalized experiences, asynchronous predictions for interactive ML experiences, and model inspection and validation tools.

## CoreML Deployment Options

Before we look at the code for exporting YOLO11 models to the CoreML format, let's understand where CoreML models are usually used.

CoreML offers various deployment options for machine learning models, including:

- **On-Device Deployment**: This method directly integrates CoreML models into your iOS app. It's particularly advantageous for ensuring low latency, enhanced privacy (since data remains on the device), and offline functionality. This approach, however, may be limited by the device's hardware capabilities, especially for larger and more complex models, and it can be executed in the following two ways:
    - **Embedded Models**: These models are included in the app bundle and are immediately accessible. They are ideal for small models that do not require frequent updates.

    - **Downloaded Models**: These models are fetched from a server as needed. This approach is suitable for larger models or those needing regular updates. It helps keep the app bundle size smaller.

- **Cloud-Based Deployment**: CoreML models are hosted on servers and accessed by the iOS app through API requests. This scalable and flexible option enables easy model updates without app revisions. It's ideal for complex models or large-scale apps requiring regular updates. However, it does require an internet connection and may pose latency and security issues.

## Exporting YOLO11 Models to CoreML

Exporting YOLO11 to CoreML enables optimized, on-device machine learning performance within Apple's ecosystem, offering benefits in terms of efficiency, security, and seamless integration with iOS, macOS, watchOS, and tvOS platforms.
=======
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to CoreML for 2x Fast Inference on Apple Devices 🚀
</p>

## What is CoreML?

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/coreml-overview.avif" alt="Apple CoreML deployment pipeline">
</p>

CoreML (styled "Core ML" by Apple) is Apple's on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) framework. It loads models in the modern **ML Program** format — the `.mlpackage` bundle the Ultralytics exporter produces — and schedules them across the device's CPU, GPU, and **Apple Neural Engine (ANE)**, the dedicated NPU in every Apple-silicon chip. Because everything runs locally, inference works offline, adds no network latency, and keeps user data on the device.

CoreML integrates directly with Apple's [Vision framework](https://developer.apple.com/documentation/vision), which handles image scaling and orientation on the way into the model — this is how the Ultralytics iOS SDK feeds camera frames to YOLO with effectively zero preprocessing cost.

## Why Export YOLO26 to CoreML?

- **Neural Engine speed**: YOLO26n detection runs end-to-end in **3.8 ms** on an iPhone 17 Pro for single images and 11.3 ms/frame in sustained real-time camera use; YOLO26n Depth takes **5.5 ms** for a single image and 16.5 ms/frame in the live camera (see the table and notes below).
- **NMS-free by design**: YOLO26 is [end-to-end](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), so the exported graph needs no NMS pipeline and decode is sub-millisecond. Older detection models like YOLO11 can embed a CoreML NMS pipeline with `nms=True`.
- **Private and offline**: All computation stays on the device — no cloud round-trips, no API keys, full [data privacy](https://www.ultralytics.com/glossary/data-privacy).
- **One export, the whole ecosystem**: The same `.mlpackage` runs on iOS, iPadOS, macOS, watchOS, tvOS, and visionOS, and powers the official Ultralytics [iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app).

## Measured Performance

End-to-end single-image inference for the official YOLO26n INT8 CoreML models on an iPhone 17 Pro (Apple A19, iOS 26.5.2). Each cell shows the **total time** (preprocessing + inference + postprocessing, excluding annotation) with the per-stage split beneath it. On iOS, Vision performs input scaling inside the inference request, so preprocessing is reported as 0 and its cost is included in inference.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>`.cpuOnly`<br>(ms)</sup> | Neural Engine<br><sup>`.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | ------------------------------------ | --------------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 9.1<br><sup>0.0 / 9.1 / 0.0</sup>    | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-seg   | Segment  | 640                         | 12.3<br><sup>0.0 / 12.1 / 0.2</sup>  | **4.8**<br><sup>0.0 / 4.5 / 0.3</sup>                     |
| YOLO26n-sem   | Semantic | 1024<sup>1</sup>            | 21.8<br><sup>0.0 / 21.0 / 0.8</sup>  | **12.1**<br><sup>0.0 / 11.3 / 0.8</sup>                   |
| YOLO26n-depth | Depth    | 640                         | 24.8<br><sup>0.0 / 23.9 / 0.9</sup>  | **5.5**<br><sup>0.0 / 4.7 / 0.9</sup>                     |
| YOLO26n-cls   | Classify | 224                         | 2.2<br><sup>0.0 / 2.2 / 0.0</sup>    | **2.0**<br><sup>0.0 / 2.0 / 0.0</sup>                     |
| YOLO26n-pose  | Pose     | 640                         | 12.0<br><sup>0.0 / 11.9 / 0.0</sup>  | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-obb   | OBB      | 1024                        | 21.7<br><sup>0.0 / 21.7 / 0.0</sup>  | **7.2**<br><sup>0.0 / 7.2 / 0.0</sup>                     |

- <sup>1</sup> Semantic CoreML exports embed the ArgMax in the graph and return a compact full-resolution class map (`[1, 1024, 1024]`) instead of float logits, so the postprocess is a sub-millisecond color sweep and masks render pixel-sharp.
- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on `bus.jpg`, measured through the [iOS SDK's](https://github.com/ultralytics/yolo-ios-app) per-stage timing via the [Flutter plugin's](https://github.com/ultralytics/yolo-flutter-app) benchmark harness in profile mode (optimized native code). Sustained real-time camera operation runs higher because it includes the capture and scaling pipeline plus thermal settling: YOLO26n detect measures 11.3 ms/frame and YOLO26n Depth 16.5 ms/frame in the live camera app on the same device — see the [iOS SDK performance doc](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md) for steady-state profiling.
- The matching Snapdragon CPU/GPU/NPU table is in the [Qualcomm QNN integration](qnn.md).

## Exporting YOLO26 Models to CoreML
>>>>>>> origin/main

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage"
=======
        # Install the required package for YOLO26
        pip install ultralytics
        ```

The `coremltools` converter is installed automatically on first export. Export runs on macOS or x86 Linux; for detailed instructions and best practices, check our [installation guide](../quickstart.md) and the [Common Issues guide](../guides/yolo-common-issues.md).

### Usage

The CoreML format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation with CoreML run on macOS only. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"
>>>>>>> origin/main

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to CoreML format
        model.export(format="coreml")  # creates 'yolo11n.mlpackage'

        # Load the exported CoreML model
        coreml_model = YOLO("yolo11n.mlpackage")

        # Run inference
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
=======
        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export to CoreML with INT8 weight quantization, matching the official app models
        model.export(format="coreml", quantize=8)  # creates 'yolo26n.mlpackage'
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Export a YOLO11n PyTorch model to CoreML format
        yolo export model=yolo11n.pt format=coreml # creates 'yolo11n.mlpackage'

        # Run inference with the exported model
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
=======
        # Export a YOLO26n PyTorch model to CoreML format with INT8 weight quantization
        yolo export model=yolo26n.pt format=coreml quantize=8 # creates 'yolo26n.mlpackage'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported CoreML model (macOS)
        model = YOLO("yolo26n.mlpackage")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported CoreML model
        yolo predict model=yolo26n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported CoreML model (macOS)
        model = YOLO("yolo26n.mlpackage")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported CoreML model
        yolo val model=yolo26n.mlpackage data=coco8.yaml
>>>>>>> origin/main
        ```

### Export Arguments

<<<<<<< HEAD
| Argument | Type             | Default    | Description                                                                                                                                                                                   |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `imgsz`  | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`   | `bool`           | `False`    | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`   | `bool`           | `False`    | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`    | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                               |

!!! tip

    Please make sure to use a macOS or x86 Linux machine when exporting to CoreML.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 CoreML Models

Having successfully exported your Ultralytics YOLO11 models to CoreML, the next critical phase is deploying these models effectively. For detailed guidance on deploying CoreML models in various environments, check out these resources:

- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: This guide includes instructions and examples to convert models from [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), PyTorch, and other libraries to Core ML.

- **[ML and Vision](https://developer.apple.com/videos/)**: A collection of comprehensive videos that cover various aspects of using and implementing CoreML models.

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: A comprehensive guide on integrating a CoreML model into an iOS application, detailing steps from preparing the model to implementing it in the app for various functionalities.

## Summary

In this guide, we went over how to export Ultralytics YOLO11 models to CoreML format. By following the steps outlined in this guide, you can ensure maximum compatibility and performance when exporting YOLO11 models to CoreML.

For further details on usage, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

Also, if you'd like to know more about other Ultralytics YOLO11 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of valuable resources and insights there.

## FAQ

### How do I export YOLO11 models to CoreML format?

To export your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models to CoreML format, you'll first need to ensure you have the `ultralytics` package installed. You can install it using:

!!! example "Installation"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

Next, you can export the model using the following Python or CLI commands:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="coreml")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=coreml
        ```

For further details, refer to the [Exporting YOLO11 Models to CoreML](../modes/export.md) section of our documentation.

### What are the benefits of using CoreML for deploying YOLO11 models?

CoreML provides numerous advantages for deploying [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models on Apple devices:

- **On-device Processing**: Enables local model inference on devices, ensuring [data privacy](https://www.ultralytics.com/glossary/data-privacy) and minimizing latency.
- **Performance Optimization**: Leverages the full potential of the device's CPU, GPU, and Neural Engine, optimizing both speed and efficiency.
- **Ease of Integration**: Offers a seamless integration experience with Apple's ecosystems, including iOS, macOS, watchOS, and tvOS.
- **Versatility**: Supports a wide range of machine learning tasks such as image analysis, audio processing, and natural language processing using the CoreML framework.

For more details on integrating your CoreML model into an iOS app, check out the guide on [Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app).

### What are the deployment options for YOLO11 models exported to CoreML?

Once you export your YOLO11 model to CoreML format, you have multiple deployment options:

1. **On-Device Deployment**: Directly integrate CoreML models into your app for enhanced privacy and offline functionality. This can be done as:
    - **Embedded Models**: Included in the app bundle, accessible immediately.
    - **Downloaded Models**: Fetched from a server as needed, keeping the app bundle size smaller.

2. **Cloud-Based Deployment**: Host CoreML models on servers and access them via API requests. This approach supports easier updates and can handle more complex models.

For detailed guidance on deploying CoreML models, refer to [CoreML Deployment Options](#coreml-deployment-options).

### How does CoreML ensure optimized performance for YOLO11 models?

CoreML ensures optimized performance for [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models by utilizing various optimization techniques:

- **Hardware Acceleration**: Uses the device's CPU, GPU, and Neural Engine for efficient computation.
- **Model Compression**: Provides tools for compressing models to reduce their footprint without compromising accuracy.
- **Adaptive Inference**: Adjusts inference based on the device's capabilities to maintain a balance between speed and performance.

For more information on performance optimization, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

### Can I run inference directly with the exported CoreML model?

Yes, you can run inference directly using the exported CoreML model. Below are the commands for Python and CLI:

!!! example "Running Inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        coreml_model = YOLO("yolo11n.mlpackage")
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

For additional information, refer to the [Usage section](#usage) of the CoreML export guide.
=======
| Argument   | Type             | Default    | Description                                                                                                                                                                                                                                                           |
| ---------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                    |
| `imgsz`    | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                     |
| `quantize` | `int` or `str`   | `None`     | Quantization precision (weight-only for CoreML): `16` (FP16), `8` (INT8), `"w8a16"` (INT8 weights with FP16 activations), or `32`/unset (FP32). Unset NMS ML Programs use FP16 for Xcode preview; pass `32` to override. Replaces the deprecated `half`/`int8` flags. |
| `nms`      | `bool`           | `False`    | Embeds a CoreML NMS pipeline. Detection models only (ignored with a warning for other tasks); not needed for NMS-free YOLO26, use for earlier models like YOLO11.                                                                                                     |
| `dynamic`  | `bool`           | `False`    | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                               |
| `batch`    | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                               |
| `device`   | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                                                                       |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Targeting the Neural Engine

CoreML chooses hardware via `MLModelConfiguration.computeUnits`. The Ultralytics iOS SDK defaults to `.cpuAndNeuralEngine` on iOS 16+ rather than `.all`: in a real-time camera app the GPU is already busy compositing the preview and overlays, so excluding it avoids contention and frame-time jitter while the ANE does the heavy lifting. Pin `.cpuOnly` only for compatibility testing — the table above shows what it costs.

Running a CoreML model from Python on a **Mac host** (via Ultralytics or `coremltools`) follows the same rule: Ultralytics loads with `ComputeUnit.CPU_AND_NE` (macOS 13+, falling back to `CPU_ONLY` on older macOS), keeping inference on the Neural Engine (~3× faster than CPU). This also avoids a current macOS host limitation where the default `ComputeUnit.ALL` / `CPU_AND_GPU` — which add the GPU/MPSGraph compile path — **abort the process** with an `Error: MLIR pass manager failed` assertion on `coremltools` 9.x.

## Deploying Exported YOLO26 CoreML Models

The fastest path is the official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app), the same Swift package that powers the Ultralytics iOS app and the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app). It resolves official model names automatically, downloads and caches the `.mlpackage`, and returns fully decoded results:

```swift
import UltralyticsYOLO

// Loads the official INT8 model (downloaded and cached on first use), then runs inference
let yolo = YOLO("yolo26n", task: .detect) { result in
    if case .success(let model) = result {
        let results = model(uiImage)  // boxes, labels, confidences, timing
    }
}
```

For camera apps, drop in the SDK's `YOLOView` for real-time inference with native overlays, or use the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) for cross-platform apps that share one codebase with Android.

Integrating a raw `.mlpackage` yourself is also straightforward with Apple's stack — load it with `MLModel`, wrap it in a `VNCoreMLRequest`, and feed images through `VNImageRequestHandler`. These resources cover the details:

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: Apple's guide to bundling and calling a CoreML model.
- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: Conversion, quantization, and optimization reference for the `coremltools` toolchain that powers this export.
- **[Xcode Core ML Performance Reports](https://developer.apple.com/videos/)**: Per-layer device placement and latency profiling for your exact model and device.

Ship the model either embedded in the app bundle (instant availability, ideal for nano/small models) or downloaded on first run and cached (smaller binary, easy model updates) — the official apps use the second approach with the [GitHub release assets](https://github.com/ultralytics/yolo-ios-app/releases).

## Recommended Workflow

1. **Train** your model with Ultralytics [Train mode](../modes/train.md), or start from the official YOLO26 weights
2. **Export** with `model.export(format="coreml", quantize=8)` on macOS or x86 Linux
3. **Verify** accuracy with `model.val()` on a Mac, and profile with an Xcode Core ML Performance Report on your target device
4. **Deploy** with the iOS SDK, the Flutter plugin, or your own Vision integration, targeting `.cpuAndNeuralEngine`

## Summary

In this guide, you learned how to export Ultralytics YOLO26 models to CoreML's `.mlpackage` format, quantize them for the Apple Neural Engine, and deploy them at single-digit-millisecond latencies — either through the official iOS SDK and Flutter plugin or your own Vision integration. For other deployment targets, browse the [integration guide page](../integrations/index.md), and compare formats with [Benchmark mode](../modes/benchmark.md).

## FAQ

### How do I export YOLO26 models to CoreML format?

Run `model.export(format="coreml")` in Python or `yolo export model=yolo26n.pt format=coreml` from the CLI on macOS or x86 Linux. Add `quantize=8` to match the official app models. The export produces a `yolo26n.mlpackage` ML Program ready for Xcode, the iOS SDK, or the Flutter plugin.

### Do I need `nms=True` when exporting YOLO26?

No. YOLO26 is NMS-free end-to-end, so the exported graph already emits final detections and decode costs well under a millisecond. The `nms=True` option exists for earlier detection models such as YOLO11, where it embeds a CoreML NMS pipeline so your app does not have to implement suppression. CoreML NMS pipelines only support object detection, so `nms=True` is ignored with a warning for other tasks like segmentation and pose.

### Which precision should I use — FP16 or INT8?

The official Ultralytics app models ship as INT8, which minimizes download size and runs at the speeds in the table above. `quantize=16` (FP16) is a conservative alternative with essentially no accuracy loss. Validate your exact export with `model.val()` on a Mac before shipping.

### How do I make sure inference runs on the Neural Engine?

Set `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` (the iOS SDK default on iOS 16+). Avoid `.all` in camera apps — the GPU is busy compositing the preview, and scheduling inference there causes frame-time jitter. Confirm placement with an Xcode Core ML Performance Report.

### Can I run and validate CoreML models with the Ultralytics CLI?

Yes, on macOS: `yolo predict model=yolo26n.mlpackage source=image.jpg` and `yolo val model=yolo26n.mlpackage data=coco8.yaml` work like any other format. CoreML execution requires Apple hardware, so these modes are unavailable on Linux and Windows.

### What is the fastest way to get YOLO26 running in an iOS or Flutter app?

Use the official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) (Swift Package) or the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app). Both load official models by name with automatic download and caching, run them on the Neural Engine, and include complete real-time camera UIs — the measured performance table above was produced with exactly this stack.
>>>>>>> origin/main

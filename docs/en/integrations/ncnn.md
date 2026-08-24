---
comments: true
<<<<<<< HEAD
description: Optimize YOLO11 models for mobile and embedded devices by exporting to NCNN format. Enhance performance in resource-constrained environments.
keywords: Ultralytics, YOLO11, NCNN, model export, machine learning, deployment, mobile, embedded systems, deep learning, AI models
---

# How to Export to NCNN from YOLO11 for Smooth Deployment

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on devices with limited computational power, such as mobile or embedded systems, can be tricky. You need to make sure you use a format optimized for optimal performance. This makes sure that even devices with limited processing power can handle advanced computer vision tasks well.

The export to NCNN format feature allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for lightweight device-based applications. In this guide, we'll walk you through how to convert your models to the NCNN format, making it easier for your models to perform well on various mobile and embedded devices.

## Why should you export to NCNN?

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ncnn-overview.avif" alt="NCNN overview">
=======
description: Optimize YOLO26 models for mobile and embedded devices by exporting to NCNN format. Enhance performance in resource-constrained environments.
keywords: Ultralytics, YOLO26, NCNN, model export, machine learning, deployment, mobile, embedded systems, deep learning, AI models, Vulkan, GPU acceleration
---

# Ultralytics YOLO NCNN Export

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on devices with limited computational power, such as mobile or embedded systems, requires careful format selection. Using an optimized format ensures that even resource-constrained devices can handle advanced computer vision tasks efficiently.

Exporting to NCNN format allows you to optimize your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for lightweight device-based applications. This guide covers how to convert your models to NCNN format for improved performance on mobile and embedded devices.

## Why Export to NCNN?

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ncnn-overview.avif" alt="NCNN high-performance neural network inference framework">
>>>>>>> origin/main
</p>

The [NCNN](https://github.com/Tencent/ncnn) framework, developed by Tencent, is a high-performance [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference computing framework optimized specifically for mobile platforms, including mobile phones, embedded devices, and IoT devices. NCNN is compatible with a wide range of platforms, including Linux, Android, iOS, and macOS.

<<<<<<< HEAD
NCNN is known for its fast processing speed on mobile CPUs and enables rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models to mobile platforms. This makes it easier to build smart apps, putting the power of AI right at your fingertips.

## Key Features of NCNN Models

NCNN models offer a wide range of key features that enable on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) by helping developers run their models on mobile, embedded, and edge devices:

- **Efficient and High-Performance**: NCNN models are made to be efficient and lightweight, optimized for running on mobile and embedded devices like Raspberry Pi with limited resources. They can also achieve high performance with high [accuracy](https://www.ultralytics.com/glossary/accuracy) on various computer vision-based tasks.

- **Quantization**: NCNN models often support quantization which is a technique that reduces the [precision](https://www.ultralytics.com/glossary/precision) of the model's weights and activations. This leads to further improvements in performance and reduces memory footprint.

- **Compatibility**: NCNN models are compatible with popular deep learning frameworks like [TensorFlow](https://www.tensorflow.org/), [Caffe](https://caffe.berkeleyvision.org/), and [ONNX](https://onnx.ai/). This compatibility allows developers to use existing models and workflows easily.

- **Easy to Use**: NCNN models are designed for easy integration into various applications, thanks to their compatibility with popular deep learning frameworks. Additionally, NCNN offers user-friendly tools for converting models between different formats, ensuring smooth interoperability across the development landscape.

## Deployment Options with NCNN

Before we look at the code for exporting YOLO11 models to the NCNN format, let's understand how NCNN models are normally used.

NCNN models, designed for efficiency and performance, are compatible with a variety of deployment platforms:

- **Mobile Deployment**: Specifically optimized for Android and iOS, allowing for seamless integration into mobile applications for efficient on-device inference.

- **Embedded Systems and IoT Devices**: If you find that running inference on a Raspberry Pi with the [Ultralytics Guide](../guides/raspberry-pi.md) isn't fast enough, switching to an NCNN exported model could help speed things up. NCNN is great for devices like Raspberry Pi and NVIDIA Jetson, especially in situations where you need quick processing right on the device.

- **Desktop and Server Deployment**: Capable of being deployed in desktop and server environments across Linux, Windows, and macOS, supporting development, training, and evaluation with higher computational capacities.

## Export to NCNN: Converting Your YOLO11 Model

You can expand model compatibility and deployment flexibility by converting YOLO11 models to NCNN format.
=======
NCNN is known for its fast processing speed on mobile CPUs and enables rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models to mobile platforms, making it an excellent choice for building AI-powered applications.

## Key Features of NCNN Models

NCNN models provide several key features that enable on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml), helping developers deploy models on mobile, embedded, and edge devices:

- **Efficient and High-Performance**: NCNN models are lightweight and optimized for mobile and embedded devices like Raspberry Pi with limited resources, while maintaining high [accuracy](https://www.ultralytics.com/glossary/accuracy) on computer vision tasks.

- **Quantization**: NCNN supports quantization, a technique that reduces the [precision](https://www.ultralytics.com/glossary/precision) of model weights and activations to improve performance and reduce memory footprint.

- **Compatibility**: NCNN models are compatible with popular deep learning frameworks including [TensorFlow](https://www.tensorflow.org/), [Caffe](https://caffe.berkeleyvision.org/), and [ONNX](https://onnx.ai/), allowing developers to leverage existing models and workflows.

- **Ease of Use**: NCNN provides user-friendly tools for converting models between formats, ensuring smooth interoperability across different development environments.

- **Vulkan GPU Acceleration**: NCNN supports Vulkan for GPU-accelerated inference across multiple vendors including AMD, Intel, and other non-NVIDIA GPUs, enabling high-performance deployment on a wider range of hardware.

## Deployment Options with NCNN

NCNN models are compatible with a variety of deployment platforms:

- **Mobile Deployment**: Optimized for Android and iOS, enabling seamless integration into mobile applications for efficient on-device inference.

- **Embedded Systems and IoT Devices**: Ideal for resource-constrained devices like Raspberry Pi and NVIDIA Jetson. If standard inference on a Raspberry Pi with the [Ultralytics Guide](../guides/raspberry-pi.md) is insufficient, NCNN can provide significant performance improvements.

- **Desktop and Server Deployment**: Supports deployment across Linux, Windows, and macOS for development, training, and evaluation workflows.

## Vulkan GPU Acceleration

NCNN supports GPU acceleration through Vulkan, enabling high-performance inference on a wide range of GPUs including AMD, Intel, and other non-NVIDIA graphics cards. This is particularly useful for:

- **Cross-Vendor GPU Support**: Unlike CUDA, which is limited to NVIDIA GPUs, Vulkan works across multiple GPU vendors.
- **Multi-GPU Systems**: Select a specific Vulkan device in systems with multiple GPUs using `device="vulkan:0"`, `device="vulkan:1"`, etc.
- **Edge and Desktop Deployments**: Leverage GPU acceleration on devices where CUDA is not available.

To use Vulkan acceleration, specify the Vulkan device when running inference:

!!! example "Vulkan Inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        ncnn_model = YOLO("./yolo26n_ncnn_model")

        # Run inference with Vulkan GPU acceleration (first Vulkan device)
        results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:0")

        # Use second Vulkan device in multi-GPU systems
        results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:1")
        ```

    === "CLI"

        ```bash
        # Run inference with Vulkan GPU acceleration
        yolo predict model='./yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg' device=vulkan:0
        ```

!!! tip "Vulkan Requirements"

    Ensure you have Vulkan drivers installed for your GPU. Most modern GPU drivers include Vulkan support by default. You can verify Vulkan availability using tools like `vulkaninfo` on Linux or the Vulkan SDK on Windows.

## Export to NCNN: Converting Your YOLO26 Model

You can expand model compatibility and deployment flexibility by converting YOLO26 models to NCNN format.

## Supported Tasks

NCNN export supports the standard Ultralytics YOLO26 task set.

| Task                                          | Supported |
| :-------------------------------------------- | :-------- |
| [Object Detection](../tasks/detect.md)        | ✅        |
| [Instance Segmentation](../tasks/segment.md)  | ✅        |
| [Semantic Segmentation](../tasks/semantic.md) | ✅        |
| [Pose Estimation](../tasks/pose.md)           | ✅        |
| [OBB Detection](../tasks/obb.md)              | ✅        |
| [Classification](../tasks/classify.md)        | ✅        |
>>>>>>> origin/main

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
<<<<<<< HEAD
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

All [Ultralytics YOLO11 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

!!! example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLO11 model
          model = YOLO("yolo11n.pt")

          # Export the model to NCNN format
          model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

          # Load the exported NCNN model
          ncnn_model = YOLO("./yolo11n_ncnn_model")

          # Run inference
          results = ncnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLO11n PyTorch model to NCNN format
          yolo export model=yolo11n.pt format=ncnn # creates '/yolo11n_ncnn_model'

          # Run inference with the exported model
          yolo predict model='./yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
          ```

### Export Arguments

| Argument | Type             | Default  | Description                                                                                                                             |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'ncnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`  | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `half`   | `bool`           | `False`  | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.            |
| `batch`  | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str`            | `None`   | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                         |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 NCNN Models

After successfully exporting your Ultralytics YOLO11 models to NCNN format, you can now deploy them. The primary and recommended first step for running an NCNN model is to use the `YOLO("yolo11n_ncnn_model/")` method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your NCNN models in various other settings, take a look at the following resources:

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**: This blog explains how to use NCNN models for performing tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) through Android applications.

- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**: Understand how to use NCNN models for performing tasks through macOS.

- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**: Explore this page to learn how to deploy NCNN models on limited resource devices like Raspberry Pi and other similar devices.

- **[Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**: Explore this blog to learn how to deploy NCNN models on Windows x64 using Visual Studio Community 2017.

## Summary

In this guide, we've gone over exporting Ultralytics YOLO11 models to the NCNN format. This conversion step is crucial for improving the efficiency and speed of YOLO11 models, making them more effective and suitable for limited-resource computing environments.

For detailed instructions on usage, please refer to the [official NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html).

Also, if you're interested in exploring other integration options for Ultralytics YOLO11, be sure to visit our [integration guide page](index.md) for further insights and information.

## FAQ

### How do I export Ultralytics YOLO11 models to NCNN format?

To export your Ultralytics YOLO11 model to NCNN format, follow these steps:

- **Python**: Use the `export` function from the YOLO class.
=======
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices, see the [Ultralytics Installation guide](../quickstart.md). If you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions.

### Usage

All [Ultralytics YOLO26 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

The NCNN format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to NCNN format
        yolo export model=yolo26n.pt format=ncnn # creates '/yolo26n_ncnn_model'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        model = YOLO("./yolo26n_ncnn_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported NCNN model
        yolo predict model=./yolo26n_ncnn_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        model = YOLO("./yolo26n_ncnn_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported NCNN model
        yolo val model=./yolo26n_ncnn_model data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default  | Description                                                                                                                                 |
| ---------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'ncnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                          |
| `imgsz`    | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.           |
| `quantize` | `int` or `str`   | `None`   | Quantization precision: `16` (FP16) reduces model size and can speed up inference; `32`/unset is FP32. Replaces the deprecated `half` flag. |
| `batch`    | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.     |
| `device`   | `str`            | `None`   | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                             |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 NCNN Models

After exporting your Ultralytics YOLO26 models to NCNN format, you can deploy them using the `YOLO("yolo26n_ncnn_model/")` method as shown in the usage example above. For platform-specific deployment instructions, see the following resources:

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**: Build and integrate NCNN models for [object detection](https://www.ultralytics.com/glossary/object-detection) in Android applications.

- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**: Deploy NCNN models on macOS systems.

- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**: Deploy NCNN models on Linux devices including Raspberry Pi and similar embedded systems.

- **[Windows x64](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**: Deploy NCNN models on Windows x64 using Visual Studio.

## Summary

This guide covered exporting Ultralytics YOLO26 models to NCNN format for improved efficiency and speed on resource-constrained devices.

For additional details, refer to the [official NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html). For other export options, visit our [integration guide page](index.md).

## FAQ

### How do I export Ultralytics YOLO26 models to NCNN format?

To export your Ultralytics YOLO26 model to NCNN format:

- **Python**: Use the `export` method from the YOLO class.
>>>>>>> origin/main

    ```python
    from ultralytics import YOLO

<<<<<<< HEAD
    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

    # Export to NCNN format
    model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'
    ```

- **CLI**: Use the `yolo` command with the `export` argument.
    ```bash
    yolo export model=yolo11n.pt format=ncnn # creates '/yolo11n_ncnn_model'
    ```

For detailed export options, check the [Export](../modes/export.md) page in the documentation.

### What are the advantages of exporting YOLO11 models to NCNN?

Exporting your Ultralytics YOLO11 models to NCNN offers several benefits:
=======
    # Load a YOLO26 model
    model = YOLO("yolo26n.pt")

    # Export to NCNN format
    model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'
    ```

- **CLI**: Use the `yolo export` command.

    ```bash
    yolo export model=yolo26n.pt format=ncnn # creates '/yolo26n_ncnn_model'
    ```

For detailed export options, see the [Export](../modes/export.md) documentation.

### What are the advantages of exporting YOLO26 models to NCNN?

Exporting your Ultralytics YOLO26 models to NCNN offers several benefits:
>>>>>>> origin/main

- **Efficiency**: NCNN models are optimized for mobile and embedded devices, ensuring high performance even with limited computational resources.
- **Quantization**: NCNN supports techniques like quantization that improve model speed and reduce memory usage.
- **Broad Compatibility**: You can deploy NCNN models on multiple platforms, including Android, iOS, Linux, and macOS.
<<<<<<< HEAD

For more details, see the [Export to NCNN](#why-should-you-export-to-ncnn) section in the documentation.
=======
- **Vulkan GPU Acceleration**: Leverage GPU acceleration on AMD, Intel, and other non-NVIDIA GPUs via Vulkan for faster inference.

For more details, see the [Why Export to NCNN?](#why-export-to-ncnn) section.
>>>>>>> origin/main

### Why should I use NCNN for my mobile AI applications?

NCNN, developed by Tencent, is specifically optimized for mobile platforms. Key reasons to use NCNN include:

- **High Performance**: Designed for efficient and fast processing on mobile CPUs.
- **Cross-Platform**: Compatible with popular frameworks such as [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and ONNX, making it easier to convert and deploy models across different platforms.
- **Community Support**: Active community support ensures continual improvements and updates.

<<<<<<< HEAD
To understand more, visit the [NCNN overview](#key-features-of-ncnn-models) in the documentation.
=======
For more information, see the [Key Features of NCNN Models](#key-features-of-ncnn-models) section.
>>>>>>> origin/main

### What platforms are supported for NCNN [model deployment](https://www.ultralytics.com/glossary/model-deployment)?

NCNN is versatile and supports various platforms:

- **Mobile**: Android, iOS.
- **Embedded Systems and IoT Devices**: Devices like Raspberry Pi and NVIDIA Jetson.
- **Desktop and Servers**: Linux, Windows, and macOS.

<<<<<<< HEAD
If running models on a Raspberry Pi isn't fast enough, converting to the NCNN format could speed things up as detailed in our [Raspberry Pi Guide](../guides/raspberry-pi.md).

### How can I deploy Ultralytics YOLO11 NCNN models on Android?

To deploy your YOLO11 models on Android:
=======
For improved performance on Raspberry Pi, consider using NCNN format as detailed in our [Raspberry Pi Guide](../guides/raspberry-pi.md).

### How can I deploy Ultralytics YOLO26 NCNN models on Android?

To deploy your YOLO26 models on Android:
>>>>>>> origin/main

1. **Build for Android**: Follow the [NCNN Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android) guide.
2. **Integrate with Your App**: Use the NCNN Android SDK to integrate the exported model into your application for efficient on-device inference.

<<<<<<< HEAD
For step-by-step instructions, refer to our guide on [Deploying YOLO11 NCNN Models](#deploying-exported-yolo11-ncnn-models).

For more advanced guides and use cases, visit the [Ultralytics documentation page](../guides/model-deployment-options.md).
=======
For detailed instructions, see [Deploying Exported YOLO26 NCNN Models](#deploying-exported-yolo26-ncnn-models).

For more advanced guides and use cases, visit the [Ultralytics deployment guide](../guides/model-deployment-options.md).

### How do I use Vulkan GPU acceleration with NCNN models?

NCNN supports Vulkan for GPU acceleration on AMD, Intel, and other non-NVIDIA GPUs. To use Vulkan:

```python
from ultralytics import YOLO

# Load NCNN model and run with Vulkan GPU
model = YOLO("yolo26n_ncnn_model")
results = model("image.jpg", device="vulkan:0")  # Use first Vulkan device
```

For multi-GPU systems, specify the device index (e.g., `vulkan:1` for the second GPU). Ensure Vulkan drivers are installed for your GPU. See the [Vulkan GPU Acceleration](#vulkan-gpu-acceleration) section for more details.
>>>>>>> origin/main

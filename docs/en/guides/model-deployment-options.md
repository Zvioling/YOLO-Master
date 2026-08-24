---
<<<<<<< HEAD
comments: true
description: Learn about YOLO11's diverse deployment options to maximize your model's performance. Explore PyTorch, TensorRT, OpenVINO, TF Lite, and more!
keywords: YOLO11, deployment options, export formats, PyTorch, TensorRT, OpenVINO, TF Lite, machine learning, model deployment
---

# Comparative Analysis of YOLO11 Deployment Options

## Introduction

You've come a long way on your journey with YOLO11. You've diligently collected data, meticulously annotated it, and put in the hours to train and rigorously evaluate your custom YOLO11 model. Now, it's time to put your model to work for your specific application, use case, or project. But there's a critical decision that stands before you: how to export and deploy your model effectively.
=======
title: YOLO26 Deployment Options Compared
comments: true
description: Learn about YOLO26's diverse deployment options to maximize your model's performance. Explore PyTorch, TensorRT, OpenVINO, LiteRT, and more!
keywords: YOLO26, deployment options, export formats, PyTorch, TensorRT, OpenVINO, LiteRT, ONNX, CoreML, edge AI, NPU, model deployment
---

# Comparative Analysis of YOLO26 Deployment Options

YOLO26 supports more than 20 deployment options, each tuned for a different runtime, hardware target, or platform — from PyTorch and [ONNX](../integrations/onnx.md) to [TensorRT](../integrations/tensorrt.md), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), and dedicated edge-NPU formats. Picking the right one balances inference speed, hardware constraints, and ease of integration. This guide compares every option so you can choose the best fit for your application, then move on to [model deployment best practices](./model-deployment-practices.md) to deploy it reliably.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/QkCsj2SvZc4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Choose the Best Ultralytics YOLO11 Deployment Format for Your Project | TensorRT | OpenVINO 🚀
</p>

This guide walks you through YOLO11's deployment options and the essential factors to consider to choose the right option for your project.

## How to Select the Right Deployment Option for Your YOLO11 Model

When it's time to deploy your YOLO11 model, selecting a suitable export format is very important. As outlined in the [Ultralytics YOLO11 Modes documentation](../modes/export.md#usage-examples), the `model.export()` function allows you to convert your trained model into a variety of formats tailored to diverse environments and performance requirements.

The ideal format depends on your model's intended operational context, balancing speed, hardware constraints, and ease of integration. In the following section, we'll take a closer look at each export option, understanding when to choose each one.

## YOLO11's Deployment Options

Let's walk through the different YOLO11 deployment options. For a detailed walkthrough of the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### PyTorch

PyTorch is an open-source machine learning library widely used for applications in [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) and [artificial intelligence](https://www.ultralytics.com/glossary/artificial-intelligence-ai). It provides a high level of flexibility and speed, which has made it a favorite among researchers and developers.

- **Performance Benchmarks**: PyTorch is known for its ease of use and flexibility, which may result in a slight trade-off in raw performance when compared to other frameworks that are more specialized and optimized.
- **Compatibility and Integration**: Offers excellent compatibility with various data science and machine learning libraries in Python.
- **Community Support and Ecosystem**: One of the most vibrant communities, with extensive resources for learning and troubleshooting.
- **Case Studies**: Commonly used in research prototypes, many academic papers reference models deployed in PyTorch.
- **Maintenance and Updates**: Regular updates with active development and support for new features.
- **Security Considerations**: Regular patches for security issues, but security is largely dependent on the overall environment it's deployed in.
- **Hardware Acceleration**: Supports CUDA for GPU acceleration, essential for speeding up model training and inference.

### TorchScript

TorchScript extends PyTorch's capabilities by allowing the exportation of models to be run in a C++ runtime environment. This makes it suitable for production environments where Python is unavailable.

- **Performance Benchmarks**: Can offer improved performance over native PyTorch, especially in production environments.
- **Compatibility and Integration**: Designed for seamless transition from PyTorch to C++ production environments, though some advanced features might not translate perfectly.
- **Community Support and Ecosystem**: Benefits from PyTorch's large community but has a narrower scope of specialized developers.
- **Case Studies**: Widely used in industry settings where Python's performance overhead is a bottleneck.
- **Maintenance and Updates**: Maintained alongside PyTorch with consistent updates.
- **Security Considerations**: Offers improved security by enabling the running of models in environments without full Python installations.
- **Hardware Acceleration**: Inherits PyTorch's CUDA support, ensuring efficient GPU utilization.

### ONNX

The Open [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Exchange (ONNX) is a format that allows for model interoperability across different frameworks, which can be critical when deploying to various platforms.

- **Performance Benchmarks**: ONNX models may experience a variable performance depending on the specific runtime they are deployed on.
- **Compatibility and Integration**: High interoperability across multiple platforms and hardware due to its framework-agnostic nature.
- **Community Support and Ecosystem**: Supported by many organizations, leading to a broad ecosystem and a variety of tools for optimization.
- **Case Studies**: Frequently used to move models between different machine learning frameworks, demonstrating its flexibility.
- **Maintenance and Updates**: As an open standard, ONNX is regularly updated to support new operations and models.
- **Security Considerations**: As with any cross-platform tool, it's essential to ensure secure practices in the conversion and deployment pipeline.
- **Hardware Acceleration**: With ONNX Runtime, models can leverage various hardware optimizations.

### OpenVINO

OpenVINO is an Intel toolkit designed to facilitate the deployment of deep learning models across Intel hardware, enhancing performance and speed.

- **Performance Benchmarks**: Specifically optimized for Intel CPUs, GPUs, and VPUs, offering significant performance boosts on compatible hardware.
- **Compatibility and Integration**: Works best within the Intel ecosystem but also supports a range of other platforms.
- **Community Support and Ecosystem**: Backed by Intel, with a solid user base especially in the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) domain.
- **Case Studies**: Often utilized in IoT and [edge computing](https://www.ultralytics.com/glossary/edge-computing) scenarios where Intel hardware is prevalent.
- **Maintenance and Updates**: Intel regularly updates OpenVINO to support the latest deep learning models and Intel hardware.
- **Security Considerations**: Provides robust security features suitable for deployment in sensitive applications.
- **Hardware Acceleration**: Tailored for acceleration on Intel hardware, leveraging dedicated instruction sets and hardware features.

For more details on deployment using OpenVINO, refer to the Ultralytics Integration documentation: [Intel OpenVINO Export](../integrations/openvino.md).

### TensorRT

TensorRT is a high-performance deep learning inference optimizer and runtime from NVIDIA, ideal for applications needing speed and efficiency.

- **Performance Benchmarks**: Delivers top-tier performance on NVIDIA GPUs with support for high-speed inference.
- **Compatibility and Integration**: Best suited for NVIDIA hardware, with limited support outside this environment.
- **Community Support and Ecosystem**: Strong support network through NVIDIA's developer forums and documentation.
- **Case Studies**: Widely adopted in industries requiring real-time inference on video and image data.
- **Maintenance and Updates**: NVIDIA maintains TensorRT with frequent updates to enhance performance and support new GPU architectures.
- **Security Considerations**: Like many NVIDIA products, it has a strong emphasis on security, but specifics depend on the deployment environment.
- **Hardware Acceleration**: Exclusively designed for NVIDIA GPUs, providing deep optimization and acceleration.

For more information on TensorRT deployment, check out the [TensorRT integration guide](../integrations/tensorrt.md).

### CoreML

CoreML is Apple's machine learning framework, optimized for on-device performance in the Apple ecosystem, including iOS, macOS, watchOS, and tvOS.

- **Performance Benchmarks**: Optimized for on-device performance on Apple hardware with minimal battery usage.
- **Compatibility and Integration**: Exclusively for Apple's ecosystem, providing a streamlined workflow for iOS and macOS applications.
- **Community Support and Ecosystem**: Strong support from Apple and a dedicated developer community, with extensive documentation and tools.
- **Case Studies**: Commonly used in applications that require on-device machine learning capabilities on Apple products.
- **Maintenance and Updates**: Regularly updated by Apple to support the latest machine learning advancements and Apple hardware.
- **Security Considerations**: Benefits from Apple's focus on user privacy and [data security](https://www.ultralytics.com/glossary/data-security).
- **Hardware Acceleration**: Takes full advantage of Apple's neural engine and GPU for accelerated machine learning tasks.

### TF SavedModel

TF SavedModel is TensorFlow's format for saving and serving machine learning models, particularly suited for scalable server environments.

- **Performance Benchmarks**: Offers scalable performance in server environments, especially when used with TensorFlow Serving.
- **Compatibility and Integration**: Wide compatibility across TensorFlow's ecosystem, including cloud and enterprise server deployments.
- **Community Support and Ecosystem**: Large community support due to TensorFlow's popularity, with a vast array of tools for deployment and optimization.
- **Case Studies**: Extensively used in production environments for serving deep learning models at scale.
- **Maintenance and Updates**: Supported by Google and the TensorFlow community, ensuring regular updates and new features.
- **Security Considerations**: Deployment using TensorFlow Serving includes robust security features for enterprise-grade applications.
- **Hardware Acceleration**: Supports various hardware accelerations through TensorFlow's backends.

### TF GraphDef

TF GraphDef is a TensorFlow format that represents the model as a graph, which is beneficial for environments where a static computation graph is required.

- **Performance Benchmarks**: Provides stable performance for static computation graphs, with a focus on consistency and reliability.
- **Compatibility and Integration**: Easily integrates within TensorFlow's infrastructure but less flexible compared to SavedModel.
- **Community Support and Ecosystem**: Good support from TensorFlow's ecosystem, with many resources available for optimizing static graphs.
- **Case Studies**: Useful in scenarios where a static graph is necessary, such as in certain embedded systems.
- **Maintenance and Updates**: Regular updates alongside TensorFlow's core updates.
- **Security Considerations**: Ensures safe deployment with TensorFlow's established security practices.
- **Hardware Acceleration**: Can utilize TensorFlow's hardware acceleration options, though not as flexible as SavedModel.

Learn more about TF GraphDef in our [TF GraphDef integration guide](../integrations/tf-graphdef.md).

### TF Lite

TF Lite is TensorFlow's solution for mobile and embedded device machine learning, providing a lightweight library for on-device inference.

- **Performance Benchmarks**: Designed for speed and efficiency on mobile and embedded devices.
- **Compatibility and Integration**: Can be used on a wide range of devices due to its lightweight nature.
- **Community Support and Ecosystem**: Backed by Google, it has a robust community and a growing number of resources for developers.
- **Case Studies**: Popular in mobile applications that require on-device inference with minimal footprint.
- **Maintenance and Updates**: Regularly updated to include the latest features and optimizations for mobile devices.
- **Security Considerations**: Provides a secure environment for running models on end-user devices.
- **Hardware Acceleration**: Supports a variety of hardware acceleration options, including GPU and DSP.

### TF Edge TPU

TF Edge TPU is designed for high-speed, efficient computing on Google's Edge TPU hardware, perfect for IoT devices requiring real-time processing.

- **Performance Benchmarks**: Specifically optimized for high-speed, efficient computing on Google's Edge TPU hardware.
- **Compatibility and Integration**: Works exclusively with TensorFlow Lite models on Edge TPU devices.
- **Community Support and Ecosystem**: Growing support with resources provided by Google and third-party developers.
- **Case Studies**: Used in IoT devices and applications that require real-time processing with low latency.
- **Maintenance and Updates**: Continually improved upon to leverage the capabilities of new Edge TPU hardware releases.
- **Security Considerations**: Integrates with Google's robust security for IoT and edge devices.
- **Hardware Acceleration**: Custom-designed to take full advantage of Google Coral devices.

### TF.js

TensorFlow.js (TF.js) is a library that brings machine learning capabilities directly to the browser, offering a new realm of possibilities for web developers and users alike. It allows for the integration of machine learning models in web applications without the need for back-end infrastructure.

- **Performance Benchmarks**: Enables machine learning directly in the browser with reasonable performance, depending on the client device.
- **Compatibility and Integration**: High compatibility with web technologies, allowing for easy integration into web applications.
- **Community Support and Ecosystem**: Support from a community of web and Node.js developers, with a variety of tools for deploying ML models in browsers.
- **Case Studies**: Ideal for interactive web applications that benefit from client-side machine learning without the need for server-side processing.
- **Maintenance and Updates**: Maintained by the TensorFlow team with contributions from the open-source community.
- **Security Considerations**: Runs within the browser's secure context, utilizing the security model of the web platform.
- **Hardware Acceleration**: Performance can be enhanced with web-based APIs that access hardware acceleration like WebGL.

### PaddlePaddle

PaddlePaddle is an open-source deep learning framework developed by Baidu. It is designed to be both efficient for researchers and easy to use for developers. It's particularly popular in China and offers specialized support for Chinese language processing.

- **Performance Benchmarks**: Offers competitive performance with a focus on ease of use and scalability.
- **Compatibility and Integration**: Well-integrated within Baidu's ecosystem and supports a wide range of applications.
- **Community Support and Ecosystem**: While the community is smaller globally, it's rapidly growing, especially in China.
- **Case Studies**: Commonly used in Chinese markets and by developers looking for alternatives to other major frameworks.
- **Maintenance and Updates**: Regularly updated with a focus on serving Chinese language AI applications and services.
- **Security Considerations**: Emphasizes [data privacy](https://www.ultralytics.com/glossary/data-privacy) and security, catering to Chinese data governance standards.
- **Hardware Acceleration**: Supports various hardware accelerations, including Baidu's own Kunlun chips.

### MNN

MNN is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models and has industry-leading performance for inference and training on-device. In addition, MNN is also used on embedded devices, such as IoT.

- **Performance Benchmarks**: High-performance for mobile devices with excellent optimization for ARM systems.
- **Compatibility and Integration**: Works well with mobile and embedded ARM systems and X86-64 CPU architectures.
- **Community Support and Ecosystem**: Supported by the mobile and embedded machine learning community.
- **Case Studies**: Ideal for applications requiring efficient performance on mobile systems.
- **Maintenance and Updates**: Regularly maintained to ensure high performance on mobile devices.
- **Security Considerations**: Provides on-device security advantages by keeping data local.
- **Hardware Acceleration**: Optimized for ARM CPUs and GPUs for maximum efficiency.

### NCNN

NCNN is a high-performance neural network inference framework optimized for the mobile platform. It stands out for its lightweight nature and efficiency, making it particularly well-suited for mobile and embedded devices where resources are limited.

- **Performance Benchmarks**: Highly optimized for mobile platforms, offering efficient inference on ARM-based devices.
- **Compatibility and Integration**: Suitable for applications on mobile phones and embedded systems with ARM architecture.
- **Community Support and Ecosystem**: Supported by a niche but active community focused on mobile and embedded ML applications.
- **Case Studies**: Favored for mobile applications where efficiency and speed are critical on Android and other ARM-based systems.
- **Maintenance and Updates**: Continuously improved to maintain high performance on a range of ARM devices.
- **Security Considerations**: Focuses on running locally on the device, leveraging the inherent security of on-device processing.
- **Hardware Acceleration**: Tailored for ARM CPUs and GPUs, with specific optimizations for these architectures.

## Comparative Analysis of YOLO11 Deployment Options

The following table provides a snapshot of the various deployment options available for YOLO11 models, helping you to assess which may best fit your project needs based on several critical criteria. For an in-depth look at each deployment option's format, please see the [Ultralytics documentation page on export formats](../modes/export.md#export-formats).

| Deployment Option | Performance Benchmarks                          | Compatibility and Integration                  | Community Support and Ecosystem               | Case Studies                               | Maintenance and Updates                        | Security Considerations                           | Hardware Acceleration              |
| ----------------- | ----------------------------------------------- | ---------------------------------------------- | --------------------------------------------- | ------------------------------------------ | ---------------------------------------------- | ------------------------------------------------- | ---------------------------------- |
| PyTorch           | Good flexibility; may trade off raw performance | Excellent with Python libraries                | Extensive resources and community             | Research and prototypes                    | Regular, active development                    | Dependent on deployment environment               | CUDA support for GPU acceleration  |
| TorchScript       | Better for production than PyTorch              | Smooth transition from PyTorch to C++          | Specialized but narrower than PyTorch         | Industry where Python is a bottleneck      | Consistent updates with PyTorch                | Improved security without full Python             | Inherits CUDA support from PyTorch |
| ONNX              | Variable depending on runtime                   | High across different frameworks               | Broad ecosystem, supported by many orgs       | Flexibility across ML frameworks           | Regular updates for new operations             | Ensure secure conversion and deployment practices | Various hardware optimizations     |
| OpenVINO          | Optimized for Intel hardware                    | Best within Intel ecosystem                    | Solid in computer vision domain               | IoT and edge with Intel hardware           | Regular updates for Intel hardware             | Robust features for sensitive applications        | Tailored for Intel hardware        |
| TensorRT          | Top-tier on NVIDIA GPUs                         | Best for NVIDIA hardware                       | Strong network through NVIDIA                 | Real-time video and image inference        | Frequent updates for new GPUs                  | Emphasis on security                              | Designed for NVIDIA GPUs           |
| CoreML            | Optimized for on-device Apple hardware          | Exclusive to Apple ecosystem                   | Strong Apple and developer support            | On-device ML on Apple products             | Regular Apple updates                          | Focus on privacy and security                     | Apple neural engine and GPU        |
| TF SavedModel     | Scalable in server environments                 | Wide compatibility in TensorFlow ecosystem     | Large support due to TensorFlow popularity    | Serving models at scale                    | Regular updates by Google and community        | Robust features for enterprise                    | Various hardware accelerations     |
| TF GraphDef       | Stable for static computation graphs            | Integrates well with TensorFlow infrastructure | Resources for optimizing static graphs        | Scenarios requiring static graphs          | Updates alongside TensorFlow core              | Established TensorFlow security practices         | TensorFlow acceleration options    |
| TF Lite           | Speed and efficiency on mobile/embedded         | Wide range of device support                   | Robust community, Google backed               | Mobile applications with minimal footprint | Latest features for mobile                     | Secure environment on end-user devices            | GPU and DSP among others           |
| TF Edge TPU       | Optimized for Google's Edge TPU hardware        | Exclusive to Edge TPU devices                  | Growing with Google and third-party resources | IoT devices requiring real-time processing | Improvements for new Edge TPU hardware         | Google's robust IoT security                      | Custom-designed for Google Coral   |
| TF.js             | Reasonable in-browser performance               | High with web technologies                     | Web and Node.js developers support            | Interactive web applications               | TensorFlow team and community contributions    | Web platform security model                       | Enhanced with WebGL and other APIs |
| PaddlePaddle      | Competitive, easy to use and scalable           | Baidu ecosystem, wide application support      | Rapidly growing, especially in China          | Chinese market and language processing     | Focus on Chinese AI applications               | Emphasizes data privacy and security              | Including Baidu's Kunlun chips     |
| MNN               | High-performance for mobile devices.            | Mobile and embedded ARM systems and X86-64 CPU | Mobile/embedded ML community                  | Mobile systems efficiency                  | High performance maintenance on Mobile Devices | On-device security advantages                     | ARM CPUs and GPUs optimizations    |
| NCNN              | Optimized for mobile ARM-based devices          | Mobile and embedded ARM systems                | Niche but active mobile/embedded ML community | Android and ARM systems efficiency         | High performance maintenance on ARM            | On-device security advantages                     | ARM CPUs and GPUs optimizations    |

This comparative analysis gives you a high-level overview. For deployment, it's essential to consider the specific requirements and constraints of your project, and consult the detailed documentation and resources available for each option.

## Community and Support

When you're getting started with YOLO11, having a helpful community and support can make a significant impact. Here's how to connect with others who share your interests and get the assistance you need.

### Engage with the Broader Community

- **GitHub Discussions:** The [YOLO11 repository on GitHub](https://github.com/ultralytics/ultralytics) has a "Discussions" section where you can ask questions, report issues, and suggest improvements.
- **Ultralytics Discord Server:** Ultralytics has a [Discord server](https://discord.com/invite/ultralytics) where you can interact with other users and developers.

### Official Documentation and Resources

- **Ultralytics YOLO11 Docs:** The [official documentation](../index.md) provides a comprehensive overview of YOLO11, along with guides on installation, usage, and troubleshooting.

These resources will help you tackle challenges and stay updated on the latest trends and best practices in the YOLO11 community.

## Conclusion

In this guide, we've explored the different deployment options for YOLO11. We've also discussed the important factors to consider when making your choice. These options allow you to customize your model for various environments and performance requirements, making it suitable for real-world applications.

Don't forget that the YOLO11 and [Ultralytics community](https://github.com/orgs/ultralytics/discussions) is a valuable source of help. Connect with other developers and experts to learn unique tips and solutions you might not find in regular documentation. Keep seeking knowledge, exploring new ideas, and sharing your experiences.

## FAQ

### What are the deployment options available for YOLO11 on different hardware platforms?

Ultralytics YOLO11 supports various deployment formats, each designed for specific environments and hardware platforms. Key formats include:
=======
  <strong>Watch:</strong> How to Choose the Best Ultralytics YOLO26 Deployment Format for Your Project | TensorRT | OpenVINO 🚀
</p>

Deployment is the stage in the [computer vision project workflow](./steps-of-a-cv-project.md) where a trained model starts doing real work, so the format you export to has a direct impact on speed, cost, and portability.

## How to Select the Right Deployment Option for Your YOLO26 Model

When it's time to deploy your YOLO26 model, selecting a suitable export format is very important. As outlined in the [Ultralytics YOLO26 export documentation](../modes/export.md#usage-examples), the `model.export()` function converts your trained model into a variety of formats tailored to diverse environments and performance requirements.

The ideal format depends on your model's intended operational context and hardware.

!!! tip "Skip the manual export"

    For managed deployment without manual export, [Ultralytics Platform](https://platform.ultralytics.com) provides ready-to-use [inference endpoints](../platform/deploy/endpoints.md) with auto-scaling across 43 global regions.

## YOLO26's Deployment Options

Here is a short description of each format and when to reach for it. For the full export walkthrough, see the [export documentation](../modes/export.md); for the side-by-side criteria, jump to the [comparison table](#deployment-options-compared).

- **PyTorch** (`.pt`): The native training and inference format, offering maximum flexibility and CUDA GPU acceleration — ideal for research and prototyping with no export step required.
- **[TorchScript](../integrations/torchscript.md)** (`torchscript`): Serializes the model for a Python-free C++ runtime, suited to production systems where Python is unavailable.
- **[ONNX](../integrations/onnx.md)** (`onnx`): A framework-agnostic interchange format with broad cross-platform and hardware support through ONNX Runtime.
- **[OpenVINO](../integrations/openvino.md)** (`openvino`): Intel's toolkit for optimized inference on Intel CPUs, integrated GPUs, and NPUs, common in IoT and [edge computing](https://www.ultralytics.com/glossary/edge-computing).
- **[TensorRT](../integrations/tensorrt.md)** (`engine`): NVIDIA's high-performance runtime delivering top-tier GPU inference with FP16 and INT8 optimization.
- **[CoreML](../integrations/coreml.md)** (`coreml`): Apple's on-device format for iOS, macOS, watchOS, and tvOS, using the Apple Neural Engine.
- **[TF SavedModel](../integrations/tf-savedmodel.md)** (`saved_model`): TensorFlow's standard format for scalable server-side serving with TensorFlow Serving.
- **[TF GraphDef](../integrations/tf-graphdef.md)** (`pb`): A frozen static-graph TensorFlow format for environments that need a fixed computation graph.
- **[TF Edge TPU](../integrations/edge-tpu.md)** (`edgetpu`): Compiles `.tflite` models for Google Coral Edge TPU accelerators.
- **[LiteRT](../integrations/litert.md)** (`litert`): Google's on-device runtime (formerly TensorFlow Lite) for mobile, embedded, and browser inference from a single `.tflite` model, with FP32 and INT8 support and in-browser execution via LiteRT.js.
- **[PaddlePaddle](../integrations/paddlepaddle.md)** (`paddle`): Baidu's [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) framework, popular in China, with broad hardware support.
- **[MNN](../integrations/mnn.md)** (`mnn`): A lightweight, high-performance inference engine optimized for mobile and embedded ARM and x86-64 systems.
- **[NCNN](../integrations/ncnn.md)** (`ncnn`): A high-performance, lightweight inference framework tuned for mobile ARM devices.
- **[Sony IMX500](../integrations/sony-imx500.md)** (`imx`): Exports for Sony's IMX500 intelligent vision sensor with on-chip processing, such as the Raspberry Pi AI Camera.
- **[Rockchip RKNN](../integrations/rockchip-rknn.md)** (`rknn`): Targets Rockchip NPUs on embedded boards with FP16 and INT8 quantization.
- **[ExecuTorch](../integrations/executorch.md)** (`executorch`): PyTorch's native on-device runtime for mobile (iOS and Android) and embedded systems via XNNPACK.
- **[Axelera AI](../integrations/axelera.md)** (`axelera`): Compiles for Axelera's Metis AIPU (up to 856 TOPS) over PCIe or M.2 for high-throughput edge inference.
- **[DEEPX](../integrations/deepx.md)** (`deepx`): Targets DEEPX NPU hardware with INT8 quantization for embedded edge inference.
- **[Qualcomm QNN](../integrations/qnn.md)** (`qnn`): On-device inference on Snapdragon Hexagon NPU, Adreno GPU, and CPU through the Qualcomm AI stack.

The [Hailo integration](../integrations/hailo.md) exports YOLO detection, segmentation, pose, OBB, and classification models directly to Hailo HEF; see its compatibility table for validated models and targets. The [Ambarella integration](../integrations/ambarella.md) follows an ONNX-first workflow and compiles ONNX exports to the AmbaPB format with Ambarella's CVflow toolchain for CVflow® SoCs such as the CV72, optionally using SpongeTorch compression-aware training.

## Deployment Options Compared

The following table summarizes the deployment options for YOLO26 models across the criteria that usually drive the choice. For an in-depth look at each format, see the [export formats documentation](../modes/export.md#export-formats).

| Deployment Option | Performance Benchmarks                          | Compatibility and Integration                  | Community Support and Ecosystem               | Case Studies                                | Maintenance and Updates                        | Security Considerations                           | Hardware Acceleration               |
| ----------------- | ----------------------------------------------- | ---------------------------------------------- | --------------------------------------------- | ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------- | ----------------------------------- |
| PyTorch           | Good flexibility; may trade off raw performance | Excellent with Python libraries                | Extensive resources and community             | Research and prototypes                     | Regular, active development                    | Dependent on deployment environment               | CUDA support for GPU acceleration   |
| TorchScript       | Better for production than PyTorch              | Smooth transition from PyTorch to C++          | Specialized but narrower than PyTorch         | Industry where Python is a bottleneck       | Consistent updates with PyTorch                | Improved security without full Python             | Inherits CUDA support from PyTorch  |
| ONNX              | Variable depending on runtime                   | High across different frameworks               | Broad ecosystem, supported by many orgs       | Flexibility across ML frameworks            | Regular updates for new operations             | Ensure secure conversion and deployment practices | Various hardware optimizations      |
| OpenVINO          | Optimized for Intel hardware                    | Best within Intel ecosystem                    | Solid in computer vision domain               | IoT and edge with Intel hardware            | Regular updates for Intel hardware             | Robust features for sensitive applications        | Tailored for Intel hardware         |
| TensorRT          | Top-tier on NVIDIA GPUs                         | Best for NVIDIA hardware                       | Strong network through NVIDIA                 | Real-time video and image inference         | Frequent updates for new GPUs                  | Emphasis on security                              | Designed for NVIDIA GPUs            |
| CoreML            | Optimized for on-device Apple hardware          | Exclusive to Apple ecosystem                   | Strong Apple and developer support            | On-device ML on Apple products              | Regular Apple updates                          | Focus on privacy and security                     | Apple neural engine and GPU         |
| TF SavedModel     | Scalable in server environments                 | Wide compatibility in TensorFlow ecosystem     | Large support due to TensorFlow popularity    | Serving models at scale                     | Regular updates by Google and community        | Robust features for enterprise                    | Various hardware accelerations      |
| TF GraphDef       | Stable for static computation graphs            | Integrates well with TensorFlow infrastructure | Resources for optimizing static graphs        | Scenarios requiring static graphs           | Updates alongside TensorFlow core              | Established TensorFlow security practices         | TensorFlow acceleration options     |
| TF Edge TPU       | Optimized for Google's Edge TPU hardware        | Exclusive to Edge TPU devices                  | Growing with Google and third-party resources | IoT devices requiring real-time processing  | Improvements for new Edge TPU hardware         | Google's robust IoT security                      | Custom-designed for Google Coral    |
| LiteRT            | Speed and efficiency on mobile/embedded/web     | Mobile, embedded, edge, and browser support    | Robust community, Google backed               | On-device apps across Android, iOS, and web | Latest on-device runtime features              | Secure on-device and in-browser inference         | GPU, DSP, and WebGPU acceleration   |
| PaddlePaddle      | Competitive, easy to use and scalable           | Baidu ecosystem, wide application support      | Rapidly growing, especially in China          | Chinese market and language processing      | Focus on Chinese AI applications               | Emphasizes data privacy and security              | Including Baidu's Kunlun chips      |
| MNN               | High-performance for mobile devices             | Mobile and embedded ARM systems and X86-64 CPU | Mobile/embedded ML community                  | Mobile systems efficiency                   | High performance maintenance on mobile devices | On-device security advantages                     | ARM CPUs and GPUs optimizations     |
| NCNN              | Optimized for mobile ARM-based devices          | Mobile and embedded ARM systems                | Niche but active mobile/embedded ML community | Android and ARM systems efficiency          | High performance maintenance on ARM            | On-device security advantages                     | ARM CPUs and GPUs optimizations     |
| Sony IMX500       | On-sensor inference at very low power           | Sony IMX500 sensor, Raspberry Pi AI Camera     | Sony AITRIOS ecosystem                        | On-camera edge AI                           | Sony SDK and MCT toolchain updates             | Data stays on the sensor                          | Sony IMX500 on-chip accelerator     |
| Rockchip RKNN     | Optimized for Rockchip NPUs                     | Rockchip SoC boards (e.g. RK3588)              | Rockchip developer community                  | Embedded SBC and edge devices               | Rockchip RKNN-Toolkit updates                  | On-device local inference                         | Rockchip NPU                        |
| ExecuTorch        | Efficient on-device PyTorch runtime             | iOS, Android, embedded via XNNPACK             | Backed by the PyTorch project                 | Mobile and embedded apps                    | Maintained alongside PyTorch                   | On-device inference keeps data local              | XNNPACK and mobile CPU/GPU backends |
| Axelera AI        | Very high throughput (up to 856 TOPS)           | Metis AIPU over PCIe or M.2                    | Axelera Voyager SDK                           | High-throughput edge inference              | Axelera SDK updates                            | On-premises edge inference                        | Axelera Metis AIPU                  |
| DEEPX             | INT8-optimized NPU inference                    | DEEPX NPU hardware                             | DEEPX developer tools (dx_com, dx_engine)     | Embedded edge inference                     | DEEPX SDK and runtime updates                  | On-device local inference                         | DEEPX NPU                           |
| Qualcomm QNN      | Fast on-device Snapdragon inference             | Snapdragon Hexagon NPU, Adreno GPU, CPU        | Qualcomm AI Hub ecosystem                     | Mobile and edge Snapdragon devices          | Qualcomm AI stack (QAIRT) updates              | On-device inference keeps data local              | Snapdragon Hexagon NPU              |

This comparison gives you a high-level overview. For deployment, weigh the specific requirements and constraints of your project against each option, and consult the linked integration guide for the format you choose.

## Conclusion

YOLO26's wide range of export formats lets you tailor a model to almost any environment, from a cloud GPU server to an on-sensor edge camera. Once you have picked a format, follow [model deployment best practices](./model-deployment-practices.md) for optimization, troubleshooting, and security, and lean on the [Ultralytics community](https://github.com/orgs/ultralytics/discussions) when you hit a snag.

## FAQ

### What are the deployment options available for YOLO26 on different hardware platforms?

Ultralytics YOLO26 supports various deployment formats, each designed for specific environments and hardware platforms. Key formats include:
>>>>>>> origin/main

- **PyTorch** for research and prototyping, with excellent Python integration.
- **TorchScript** for production environments where Python is unavailable.
- **ONNX** for cross-platform compatibility and hardware acceleration.
- **OpenVINO** for optimized performance on Intel hardware.
- **TensorRT** for high-speed inference on NVIDIA GPUs.

Each format has unique advantages. For a detailed walkthrough, see our [export process documentation](../modes/export.md#usage-examples).

<<<<<<< HEAD
### How do I improve the inference speed of my YOLO11 model on an Intel CPU?

To enhance inference speed on Intel CPUs, you can deploy your YOLO11 model using Intel's OpenVINO toolkit. OpenVINO offers significant performance boosts by optimizing models to leverage Intel hardware efficiently.

1. Convert your YOLO11 model to the OpenVINO format using the `model.export()` function.
=======
### How do I improve the inference speed of my YOLO26 model on an Intel CPU?

To enhance inference speed on Intel CPUs, you can deploy your YOLO26 model using Intel's OpenVINO toolkit. OpenVINO offers significant performance boosts by optimizing models to leverage Intel hardware efficiently.

1. Convert your YOLO26 model to the OpenVINO format using the `model.export()` function.
>>>>>>> origin/main
2. Follow the detailed setup guide in the [Intel OpenVINO Export documentation](../integrations/openvino.md).

For more insights, check out our [blog post](https://www.ultralytics.com/blog/achieve-faster-inference-speeds-ultralytics-yolov8-openvino).

<<<<<<< HEAD
### Can I deploy YOLO11 models on mobile devices?

Yes, YOLO11 models can be deployed on mobile devices using [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite (TF Lite) for both Android and iOS platforms. TF Lite is designed for mobile and embedded devices, providing efficient on-device inference.
=======
### Can I deploy YOLO26 models on mobile devices?

Yes, YOLO26 models can be deployed on mobile devices using [LiteRT](../integrations/litert.md) (formerly TensorFlow Lite) and [NCNN](../integrations/ncnn.md) for Android, and [CoreML](../integrations/coreml.md) or LiteRT for iOS. LiteRT is Google's on-device runtime for mobile and embedded devices and runs the same model across Android, iOS, and the browser, providing efficient on-device inference.
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
<<<<<<< HEAD
        # Export command for TFLite format
        model.export(format="tflite")
=======
        # Export command for NCNN format
        model.export(format="ncnn")
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        # CLI command for TFLite export
        yolo export --format tflite
        ```

For more details on deploying models to mobile, refer to our [TF Lite integration guide](../integrations/tflite.md).

### What factors should I consider when choosing a deployment format for my YOLO11 model?

When choosing a deployment format for YOLO11, consider the following factors:

- **Performance**: Some formats like TensorRT provide exceptional speeds on NVIDIA GPUs, while OpenVINO is optimized for Intel hardware.
- **Compatibility**: ONNX offers broad compatibility across different platforms.
- **Ease of Integration**: Formats like CoreML or TF Lite are tailored for specific ecosystems like iOS and Android, respectively.
=======
        # CLI command for NCNN export
        yolo export model=yolo26n.pt format=ncnn
        ```

For more details on deploying models to mobile, refer to our [LiteRT integration guide](../integrations/litert.md).

### What factors should I consider when choosing a deployment format for my YOLO26 model?

When choosing a deployment format for YOLO26, consider the following factors:

- **Performance**: Some formats like TensorRT provide exceptional speeds on NVIDIA GPUs, while OpenVINO is optimized for Intel hardware.
- **Compatibility**: ONNX offers broad compatibility across different platforms.
- **Ease of Integration**: Formats like CoreML or LiteRT are tailored for specific ecosystems like iOS and Android, respectively.
>>>>>>> origin/main
- **Community Support**: Formats like [PyTorch](https://www.ultralytics.com/glossary/pytorch) and TensorFlow have extensive community resources and support.

For a comparative analysis, refer to our [export formats documentation](../modes/export.md#export-formats).

<<<<<<< HEAD
### How can I deploy YOLO11 models in a web application?

To deploy YOLO11 models in a web application, you can use TensorFlow.js (TF.js), which allows for running [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models directly in the browser. This approach eliminates the need for backend infrastructure and provides real-time performance.

1. Export the YOLO11 model to the TF.js format.
2. Integrate the exported model into your web application.

For step-by-step instructions, refer to our guide on [TensorFlow.js integration](../integrations/tfjs.md).
=======
### How can I deploy YOLO26 models in a web application?

To deploy YOLO26 models in a web application, you can use [LiteRT.js](https://developers.google.com/edge/litert/web), LiteRT's web runtime, which allows for running [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models directly in the browser and Node.js. This approach eliminates the need for backend infrastructure and provides real-time performance.

1. Export the YOLO26 model to the LiteRT format.
2. Integrate the exported model into your web application using LiteRT.js.

For step-by-step instructions, refer to our [LiteRT integration guide](../integrations/litert.md).
>>>>>>> origin/main

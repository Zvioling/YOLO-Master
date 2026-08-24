---
<<<<<<< HEAD
comments: true
description: Learn how to boost your Raspberry Pi's ML performance using Coral Edge TPU with Ultralytics YOLO11. Follow our detailed setup and installation guide.
keywords: Coral Edge TPU, Raspberry Pi, YOLO11, Ultralytics, TensorFlow Lite, ML inference, machine learning, AI, installation guide, setup tutorial
---

# Coral Edge TPU on a Raspberry Pi with Ultralytics YOLO11 🚀

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/edge-tpu-usb-accelerator-and-pi.avif" alt="Raspberry Pi single board computer with USB Edge TPU accelerator">
</p>

## What is a Coral Edge TPU?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system. It enables low-power, high-performance ML inference for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite models. Read more at the [Coral Edge TPU home page](https://developers.google.com/coral).
=======
title: YOLO26 on Raspberry Pi with Coral Edge TPU
comments: true
description: Accelerate Ultralytics YOLO26 inference on a Raspberry Pi with the Coral Edge TPU. Step-by-step guide to install the runtime, export to Edge TPU format, and run fast low-power inference.
keywords: Coral Edge TPU, Raspberry Pi, YOLO26, Ultralytics, TensorFlow Lite, tflite-runtime, edge AI, low-power inference, edgetpu export, USB Accelerator, embedded AI, ML inference
---

# How to Run Ultralytics YOLO26 on a Raspberry Pi with a Coral Edge TPU

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/edge-tpu-usb-accelerator-and-pi.avif" alt="Raspberry Pi with Edge TPU accelerator">
</p>

A [Raspberry Pi](raspberry-pi.md) is a power-efficient, affordable platform for running computer vision at the edge, but on-device inference is slow even with optimized formats like [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md). Pairing the Pi with a Coral Edge TPU coprocessor offloads inference to dedicated hardware and dramatically speeds it up. This guide shows you how to install the runtime, export an Ultralytics YOLO26 model to the Edge TPU format, and run accelerated inference.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/w4yHORvDBw0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Inference on Raspberry Pi using Google Coral Edge TPU
</p>

<<<<<<< HEAD
## Boost Raspberry Pi Model Performance with Coral Edge TPU

Many people want to run their models on an embedded or mobile device such as a Raspberry Pi, since they are very power efficient and can be used in many different applications. However, the inference performance on these devices is usually poor even when using formats like [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md). The Coral Edge TPU is a great solution to this problem, since it can be used with a Raspberry Pi and accelerate inference performance greatly.

## Edge TPU on Raspberry Pi with TensorFlow Lite (New)⭐

The [existing guide](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/) by Coral on how to use the Edge TPU with a Raspberry Pi is outdated, and the current Coral Edge TPU runtime builds do not work with the current TensorFlow Lite runtime versions anymore. In addition to that, Google seems to have completely abandoned the Coral project, and there have not been any updates between 2021 and 2025. This guide will show you how to get the Edge TPU working with the latest versions of the TensorFlow Lite runtime and an updated Coral Edge TPU runtime on a Raspberry Pi single board computer (SBC).
=======
## Why Use a Coral Edge TPU?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system, enabling low-power, high-performance ML inference for TensorFlow Lite models. It is a great fit for embedded and mobile deployments where a CPU alone cannot keep up:

- **Faster inference** — the Edge TPU accelerates quantized models far beyond what the Raspberry Pi CPU achieves on its own.
- **Low power draw** — it delivers high throughput per watt, ideal for battery- or solar-powered deployments.
- **Plug-and-play** — the [USB Accelerator](https://developers.google.com/coral) connects over USB 3.0, so no extra hardware integration is required.

!!! note "Updated runtime for current TensorFlow Lite"

    The [official Coral guide](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/) is outdated: the original Coral runtime builds no longer work with current TensorFlow Lite runtime versions, and the project saw no updates between 2021 and 2025. This guide uses an actively maintained Edge TPU runtime and the latest `tflite-runtime` so the accelerator works on a current Raspberry Pi OS install.
>>>>>>> origin/main

## Prerequisites

- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (2GB or more recommended) or [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (Recommended)
- [Raspberry Pi OS](https://www.raspberrypi.com/software/) Bullseye/Bookworm (64-bit) with desktop (Recommended)
- [Coral USB Accelerator](https://developers.google.com/coral)
<<<<<<< HEAD
- A non-ARM based platform for exporting an Ultralytics [PyTorch](https://www.ultralytics.com/glossary/pytorch) model

## Installation Walkthrough

This guide assumes that you already have a working Raspberry Pi OS install and have installed `ultralytics` and all dependencies. To get `ultralytics` installed, visit the [quickstart guide](../quickstart.md) to get set up before continuing here.

### Installing the Edge TPU runtime

First, we need to install the Edge TPU runtime. There are many different versions available, so you need to choose the right version for your operating system.
The high-frequency version runs the Edge TPU at a higher clock speed, which improves performance. However, it might result in Edge TPU thermal throttling, so it is recommended to have some sort of cooling mechanism in place.
=======
- A non-ARM platform (Google Colab, an x86_64 Linux machine, or the [Ultralytics Docker container](docker-quickstart.md)) for exporting the model, since the Edge TPU compiler is not available on ARM

This guide assumes you already have a working Raspberry Pi OS install with `ultralytics` and its dependencies installed. If not, follow the [quickstart guide](../quickstart.md) first.

With the prerequisites ready, the workflow has three steps: [install the Edge TPU runtime](#install-the-edge-tpu-runtime) on the Pi, [export your model](#export-your-model-to-edge-tpu-format) on a non-ARM machine, and [run inference](#run-inference-on-the-edge-tpu) back on the Pi.

## Install the Edge TPU Runtime

The runtime ships in several builds, so pick the one that matches your operating system. The high-frequency build runs the Edge TPU at a higher clock speed for better performance, but it can cause thermal throttling — use some form of cooling if you choose it.
>>>>>>> origin/main

| Raspberry Pi OS | High frequency mode | Version to download                        |
| --------------- | :-----------------: | ------------------------------------------ |
| Bullseye 32bit  |         No          | `libedgetpu1-std_ ... .bullseye_armhf.deb` |
| Bullseye 64bit  |         No          | `libedgetpu1-std_ ... .bullseye_arm64.deb` |
| Bullseye 32bit  |         Yes         | `libedgetpu1-max_ ... .bullseye_armhf.deb` |
| Bullseye 64bit  |         Yes         | `libedgetpu1-max_ ... .bullseye_arm64.deb` |
| Bookworm 32bit  |         No          | `libedgetpu1-std_ ... .bookworm_armhf.deb` |
| Bookworm 64bit  |         No          | `libedgetpu1-std_ ... .bookworm_arm64.deb` |
| Bookworm 32bit  |         Yes         | `libedgetpu1-max_ ... .bookworm_armhf.deb` |
| Bookworm 64bit  |         Yes         | `libedgetpu1-max_ ... .bookworm_arm64.deb` |

<<<<<<< HEAD
[Download the latest version from here](https://github.com/feranick/libedgetpu/releases).

After downloading the file, you can install it with the following command:
=======
[Download the latest version from here](https://github.com/feranick/libedgetpu/releases), then install the `.deb` package:
>>>>>>> origin/main

```bash
sudo dpkg -i path/to/package.deb
```

After installing the runtime, plug your Coral Edge TPU into a USB 3.0 port on the Raspberry Pi so the new `udev` rule can take effect.

<<<<<<< HEAD
???+ warning "Important"

    If you already have the Coral Edge TPU runtime installed, uninstall it using the following command.
=======
???+ warning "Remove any existing runtime first"

    If you already have the Coral Edge TPU runtime installed, uninstall it before installing a new build.
>>>>>>> origin/main

    ```bash
    # If you installed the standard version
    sudo apt remove libedgetpu1-std

    # If you installed the high-frequency version
    sudo apt remove libedgetpu1-max
    ```

<<<<<<< HEAD
## Export to Edge TPU

To use the Edge TPU, you need to convert your model into a compatible format. It is recommended that you run export on Google Colab, x86_64 Linux machine, using the official [Ultralytics Docker container](docker-quickstart.md), or using [Ultralytics HUB](../hub/quickstart.md), since the Edge TPU compiler is not available on ARM. See the [Export Mode](../modes/export.md) for the available arguments.
=======
## Export Your Model to Edge TPU Format

To use the Edge TPU, convert your model to a compatible format. Run the export on a non-ARM platform — Google Colab, an x86_64 Linux machine, the official [Ultralytics Docker container](docker-quickstart.md), or [Ultralytics Platform](../platform/quickstart.md) — since the Edge TPU compiler is not available on ARM. See the [Export mode](../modes/export.md) for the available arguments.
>>>>>>> origin/main

!!! example "Exporting the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/model.pt")  # Load an official model or custom model

        # Export the model
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model
        ```

<<<<<<< HEAD
The exported model will be saved in the `<model_name>_saved_model/` folder with the name `<model_name>_full_integer_quant_edgetpu.tflite`. Make sure the file name ends with the `_edgetpu.tflite` suffix; otherwise, Ultralytics will not detect that you're using an Edge TPU model.

## Running the model

Before you can actually run the model, you will need to install the correct libraries.

If you already have TensorFlow installed, uninstall it with the following command:
=======
The exported model is saved in the `<model_name>_saved_model/` folder as `<model_name>_full_integer_quant_edgetpu.tflite`.

!!! warning "Keep the `_edgetpu.tflite` suffix"

    The file name must end with `_edgetpu.tflite`. If you rename it to anything else, Ultralytics will load it as a plain TensorFlow Lite model instead of detecting the Edge TPU and the accelerator will not be used.

## Run Inference on the Edge TPU

Before running the model, install the correct libraries on the Raspberry Pi. If TensorFlow is already installed, uninstall it first:
>>>>>>> origin/main

```bash
pip uninstall tensorflow tensorflow-aarch64
```

Then install or update `tflite-runtime`:

```bash
pip install -U tflite-runtime
```

<<<<<<< HEAD
Now you can run inference using the following code:
=======
Now you can run inference:
>>>>>>> origin/main

!!! example "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo predict model=path/to/MODEL_NAME_full_integer_quant_edgetpu.tflite source=path/to/source.png # Load an official model or custom model
        ```

Find comprehensive information on the [Predict](../modes/predict.md) page for full prediction mode details.

!!! note "Inference with multiple Edge TPUs"

    If you have multiple Edge TPUs, you can use the following code to select a specific TPU.
=======
        yolo predict model=path/to/ source=path/to/source.png < model_name > _full_integer_quant_edgetpu.tflite # Load an official model or custom model
        ```

Find full prediction-mode details on the [Predict](../modes/predict.md) page.

!!! note "Inference with multiple Edge TPUs"

    If you have multiple Edge TPUs, you can select a specific one with the `device` argument.
>>>>>>> origin/main

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")  # Inference defaults to the first TPU

        model.predict("path/to/source.png", device="tpu:0")  # Select the first TPU

        model.predict("path/to/source.png", device="tpu:1")  # Select the second TPU
        ```

## Benchmarks

<<<<<<< HEAD
!!! tip "Benchmarks"

    Tested with Raspberry Pi OS Bookworm 64-bit and a USB Coral Edge TPU.

    !!! note

        Shown is the inference time, pre-/postprocessing is not included.

    === "Raspberry Pi 4B 2GB"

        | Image Size | Model   | Standard Inference Time (ms) | High-Frequency Inference Time (ms) |
        |------------|---------|------------------------------|------------------------------------|
        | 320        | YOLOv8n | 32.2                         | 26.7                               |
        | 320        | YOLOv8s | 47.1                         | 39.8                               |
        | 512        | YOLOv8n | 73.5                         | 60.7                               |
        | 512        | YOLOv8s | 149.6                        | 125.3                              |

    === "Raspberry Pi 5 8GB"

        | Image Size | Model   | Standard Inference Time (ms) | High Frequency Inference Time (ms) |
        |------------|---------|------------------------------|------------------------------------|
        | 320        | YOLOv8n | 22.2                         | 16.7                               |
        | 320        | YOLOv8s | 40.1                         | 32.2                               |
        | 512        | YOLOv8n | 53.5                         | 41.6                               |
        | 512        | YOLOv8s | 132.0                        | 103.3                              |

    On average:

    - The Raspberry Pi 5 is 22% faster with the standard mode than the Raspberry Pi 4B.
    - The Raspberry Pi 5 is 30.2% faster with the high-frequency mode than the Raspberry Pi 4B.
    - The high-frequency mode is 28.4% faster than the standard mode.

## FAQ

### What is a Coral Edge TPU and how does it enhance Raspberry Pi's performance with Ultralytics YOLO11?

The Coral Edge TPU is a compact device designed to add an Edge TPU coprocessor to your system. This coprocessor enables low-power, high-performance [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) inference, particularly optimized for TensorFlow Lite models. When using a Raspberry Pi, the Edge TPU accelerates ML model inference, significantly boosting performance, especially for Ultralytics YOLO11 models. You can read more about the Coral Edge TPU on their [home page](https://developers.google.com/coral).

### How do I install the Coral Edge TPU runtime on a Raspberry Pi?

To install the Coral Edge TPU runtime on your Raspberry Pi, download the appropriate `.deb` package for your Raspberry Pi OS version from [this link](https://github.com/feranick/libedgetpu/releases). Once downloaded, use the following command to install it:
=======
The figures below were measured with Raspberry Pi OS Bookworm 64-bit and a USB Coral Edge TPU. They show inference time only (pre-/postprocessing excluded) and serve as a relative reference for the acceleration the Edge TPU provides across Pi models and modes.

!!! note "About these numbers"

    These benchmarks were recorded with YOLOv8 models. Absolute inference times vary by model version and image size, but the relative speedups between Pi models and clock modes hold.

=== "Raspberry Pi 4B 2GB"

    | Image Size | Model   | Standard Inference Time (ms) | High-Frequency Inference Time (ms) |
    |------------|---------|------------------------------|------------------------------------|
    | 320        | YOLOv8n | 32.2                         | 26.7                               |
    | 320        | YOLOv8s | 47.1                         | 39.8                               |
    | 512        | YOLOv8n | 73.5                         | 60.7                               |
    | 512        | YOLOv8s | 149.6                        | 125.3                              |

=== "Raspberry Pi 5 8GB"

    | Image Size | Model   | Standard Inference Time (ms) | High Frequency Inference Time (ms) |
    |------------|---------|------------------------------|------------------------------------|
    | 320        | YOLOv8n | 22.2                         | 16.7                               |
    | 320        | YOLOv8s | 40.1                         | 32.2                               |
    | 512        | YOLOv8n | 53.5                         | 41.6                               |
    | 512        | YOLOv8s | 132.0                        | 103.3                              |

On average:

- The Raspberry Pi 5 is 22% faster with the standard mode than the Raspberry Pi 4B.
- The Raspberry Pi 5 is 30.2% faster with the high-frequency mode than the Raspberry Pi 4B.
- The high-frequency mode is 28.4% faster than the standard mode.

## Conclusion

A Coral Edge TPU turns a Raspberry Pi into a capable, low-power inference device for Ultralytics YOLO26. Export your model on a non-ARM machine, keep the `_edgetpu.tflite` suffix, and run it with `tflite-runtime` on the Pi to get accelerated edge inference. For more deployment options, see the [Raspberry Pi](raspberry-pi.md) guide.

## FAQ

### What is a Coral Edge TPU and how does it enhance Raspberry Pi's performance with Ultralytics YOLO26?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system. This coprocessor enables low-power, high-performance ML inference, particularly optimized for TensorFlow Lite models. On a Raspberry Pi, it accelerates inference well beyond what the CPU achieves alone, which significantly boosts performance for Ultralytics YOLO26 models.

### How do I install the Coral Edge TPU runtime on a Raspberry Pi?

Download the appropriate `.deb` package for your Raspberry Pi OS version from [this link](https://github.com/feranick/libedgetpu/releases), then install it:
>>>>>>> origin/main

```bash
sudo dpkg -i path/to/package.deb
```

<<<<<<< HEAD
Make sure to uninstall any previous Coral Edge TPU runtime versions by following the steps outlined in the [Installation Walkthrough](#installation-walkthrough) section.

### Can I export my Ultralytics YOLO11 model to be compatible with Coral Edge TPU?

Yes, you can export your Ultralytics YOLO11 model to be compatible with the Coral Edge TPU. It is recommended to perform the export on Google Colab, an x86_64 Linux machine, or using the [Ultralytics Docker container](docker-quickstart.md). You can also use [Ultralytics HUB](../hub/quickstart.md) for exporting. Here is how you can export your model using Python and CLI:
=======
Make sure to uninstall any previous Coral Edge TPU runtime versions by following the steps in the [Install the Edge TPU Runtime](#install-the-edge-tpu-runtime) section.

### Can I export my Ultralytics YOLO26 model to be compatible with Coral Edge TPU?

Yes. Run the export on Google Colab, an x86_64 Linux machine, or the [Ultralytics Docker container](docker-quickstart.md); you can also use [Ultralytics Platform](../platform/quickstart.md). Here is how to export with Python and CLI:
>>>>>>> origin/main

!!! example "Exporting the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/model.pt")  # Load an official model or custom model

        # Export the model
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model
        ```

<<<<<<< HEAD
For more information, refer to the [Export Mode](../modes/export.md) documentation.

### What should I do if TensorFlow is already installed on my Raspberry Pi, but I want to use tflite-runtime instead?

If you have TensorFlow installed on your Raspberry Pi and need to switch to `tflite-runtime`, you'll need to uninstall TensorFlow first using:
=======
For more information, refer to the [Export mode](../modes/export.md) documentation.

### What should I do if TensorFlow is already installed on my Raspberry Pi, but I want to use tflite-runtime instead?

If you have TensorFlow installed and need to switch to `tflite-runtime`, uninstall TensorFlow first:
>>>>>>> origin/main

```bash
pip uninstall tensorflow tensorflow-aarch64
```

<<<<<<< HEAD
Then, install or update `tflite-runtime` with the following command:
=======
Then install or update `tflite-runtime`:
>>>>>>> origin/main

```bash
pip install -U tflite-runtime
```

<<<<<<< HEAD
For detailed instructions, refer to the [Running the Model](#running-the-model) section.

### How do I run inference with an exported YOLO11 model on a Raspberry Pi using the Coral Edge TPU?

After exporting your YOLO11 model to an Edge TPU-compatible format, you can run inference using the following code snippets:
=======
For detailed instructions, refer to the [Run Inference on the Edge TPU](#run-inference-on-the-edge-tpu) section.

### How do I run inference with an exported YOLO26 model on a Raspberry Pi using the Coral Edge TPU?

After exporting your YOLO26 model to an Edge TPU-compatible format, run inference with the following snippets. The model file must keep the `_edgetpu.tflite` suffix so Ultralytics loads it on the Edge TPU:
>>>>>>> origin/main

!!! example "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("path/to/edgetpu_model.tflite")  # Load an official model or custom model
=======
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model
>>>>>>> origin/main

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png # Load an official model or custom model
        ```

Comprehensive details on full prediction mode features can be found on the [Predict Page](../modes/predict.md).
=======
        yolo predict model=path/to/ source=path/to/source.png < model_name > _full_integer_quant_edgetpu.tflite # Load an official model or custom model
        ```

Comprehensive details on prediction mode are on the [Predict](../modes/predict.md) page.
>>>>>>> origin/main

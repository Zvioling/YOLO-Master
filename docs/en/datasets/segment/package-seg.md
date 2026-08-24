---
<<<<<<< HEAD
comments: true
description: Explore the Package Segmentation Dataset. Optimize logistics and enhance vision models with curated images for package identification and sorting.
keywords: Package Segmentation Dataset, computer vision, package identification, logistics, warehouse automation, segmentation models, training data, Ultralytics YOLO
=======
title: Package-Seg Dataset
comments: true
creator:
    name: factorypackage
license:
    name: None
description: Train Ultralytics YOLO segmentation models on the Package Segmentation Dataset — 2,197 annotated images across a single package class for logistics AI.
keywords: Package Segmentation Dataset, Ultralytics, computer vision, package identification, logistics, warehouse automation, segmentation models, YOLO, deep learning
>>>>>>> origin/main
---

# Package Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-package-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Package Segmentation Dataset In Colab"></a>

<<<<<<< HEAD
The Package Segmentation Dataset, available on Roboflow Universe, is a curated collection of images specifically tailored for tasks related to package segmentation within the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This dataset is designed to assist researchers, developers, and enthusiasts working on projects involving package identification, sorting, and handling, primarily focusing on [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) tasks.
=======
The [Ultralytics](https://www.ultralytics.com/) Package Segmentation Dataset is a curated collection of 2,197 annotated images of packages for training [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) models on a single `package` class. Built for logistics and warehouse-automation use cases like package identification, sorting, and handling, it pairs directly with [Ultralytics YOLO](../../models/yolo26.md) for real-time package analysis in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) pipelines. Explore more segmentation datasets on our [datasets overview page](index.md).
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/im7xBCnPURg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> Train Package Segmentation Model using Ultralytics YOLO11 | Industrial Packages 🎉
</p>

Containing a diverse set of images showcasing various packages in different contexts and environments, the dataset serves as a valuable resource for training and evaluating segmentation models. Whether you are engaged in logistics, warehouse automation, or any application requiring precise package analysis, the Package Segmentation Dataset provides a targeted and comprehensive set of images to enhance the performance of your computer vision algorithms. Explore more datasets for segmentation tasks on our [datasets overview page](https://docs.ultralytics.com/datasets/segment/).

## Dataset Structure

The distribution of data in the Package Segmentation Dataset is structured as follows:

- **Training set**: Encompasses 1920 images accompanied by their corresponding annotations.
- **Testing set**: Consists of 89 images, each paired with its respective annotations.
- **Validation set**: Comprises 188 images, each with corresponding annotations.

## Applications

Package segmentation, facilitated by the Package Segmentation Dataset, is crucial for optimizing logistics, enhancing last-mile delivery, improving manufacturing quality control, and contributing to smart city solutions. From e-commerce to security applications, this dataset is a key resource, fostering innovation in computer vision for diverse and efficient package analysis applications.
=======
  <strong>Watch:</strong> Train a Package Segmentation Model using Ultralytics YOLO | Industrial Packages 🎉
</p>

## Dataset Structure

The Package Segmentation Dataset splits its 2,197 images as follows:

- **Training set**: 1,920 images used for [training](https://www.ultralytics.com/glossary/training-data) the [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model.
- **Validation set**: 188 images used during training to tune [hyperparameters](../../guides/hyperparameter-tuning.md) and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).
- **Testing set**: 89 images held out to evaluate the model after training.
- **Classes**: a single `package` class covering every annotated package.
- **Download size**: ~103 MB.

## Applications

Package segmentation optimizes logistics, last-mile delivery, manufacturing quality control, and smart-city systems, with applications spanning e-commerce fulfillment and security screening. Precise package masks let automated systems locate, count, and inspect parcels in real time.
>>>>>>> origin/main

### Smart Warehouses and Logistics

In modern warehouses, [vision AI solutions](https://www.ultralytics.com/solutions) can streamline operations by automating package identification and sorting. Computer vision models trained on this dataset can quickly detect and segment packages in real-time, even in challenging environments with dim lighting or cluttered spaces. This leads to faster processing times, reduced errors, and improved overall efficiency in [logistics operations](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics).

### Quality Control and Damage Detection

<<<<<<< HEAD
Package segmentation models can be used to identify damaged packages by analyzing their shape and appearance. By detecting irregularities or deformations in package outlines, these models help ensure that only intact packages proceed through the supply chain, reducing customer complaints and return rates. This is a key aspect of [quality control in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and is vital for maintaining product integrity.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths, classes, and other essential details. For the Package Segmentation dataset, the `package-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml).
=======
Package segmentation models can identify damaged packages by analyzing their shape and appearance. By detecting irregularities or deformations in package outlines, these models help ensure that only intact packages proceed through the supply chain, reducing customer complaints and return rates. This is a key aspect of [quality control in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and is vital for maintaining product integrity.

The complete Package Segmentation Dataset can also be browsed and managed on [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

A [YAML](https://www.ultralytics.com/glossary/yaml) file defines the dataset configuration, including paths, classes, and other essential details. For the Package Segmentation dataset, the `package-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```

## Usage

<<<<<<< HEAD
To train an [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) model on the Package Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training page](../../modes/train.md).
=======
To train an [Ultralytics YOLO26n](../../models/yolo26.md) model on the Package Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following code snippets. The dataset (~103 MB) downloads automatically on first use. For a comprehensive list of available arguments, refer to the model [Training page](../../modes/train.md).
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-seg.pt")  # load a pretrained segmentation model (recommended for training)
=======
        model = YOLO("yolo26n-seg.pt")  # load a pretrained segmentation model (recommended for training)
>>>>>>> origin/main

        # Train the model on the Package Segmentation dataset
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)

        # Validate the model
        results = model.val()

        # Perform inference on an image
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # Load a pretrained segmentation model and start training
<<<<<<< HEAD
        yolo segment train data=package-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
=======
        yolo segment train data=package-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
>>>>>>> origin/main

        # Resume training from the last checkpoint
        yolo segment train data=package-seg.yaml model=path/to/last.pt resume=True

        # Validate the trained model
        yolo segment val data=package-seg.yaml model=path/to/best.pt

        # Perform inference using the trained model
        yolo segment predict model=path/to/best.pt source=path/to/image.jpg
        ```

## Sample Data and Annotations

<<<<<<< HEAD
The Package Segmentation dataset comprises a varied collection of images captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective segmentation masks:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image-1.avif)

- This image displays an instance of package segmentation, featuring annotated masks outlining recognized package objects. The dataset incorporates a diverse collection of images taken in different locations, environments, and densities. It serves as a comprehensive resource for developing models specific to this [segmentation task](https://docs.ultralytics.com/tasks/segment/).
- The example emphasizes the diversity and complexity present in the dataset, underscoring the significance of high-quality data for computer vision tasks involving package segmentation.

## Benefits of Using YOLO11 for Package Segmentation

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) offers several advantages for package segmentation tasks:

1.  **Speed and Accuracy Balance**: YOLO11 achieves high precision and efficiency, making it ideal for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) in fast-paced logistics environments. It provides a strong balance compared to models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

2.  **Adaptability**: Models trained with YOLO11 can adapt to various warehouse conditions, from dim lighting to cluttered spaces, ensuring robust performance.

3.  **Scalability**: During peak periods like holiday seasons, YOLO11 models can efficiently scale to handle increased package volumes without compromising performance or [accuracy](https://www.ultralytics.com/glossary/accuracy).

4.  **Integration Capabilities**: YOLO11 can be easily integrated with existing warehouse management systems and deployed across various platforms using formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), facilitating end-to-end automated solutions.
=======
Below is an example from the Package Segmentation Dataset with its segmentation masks overlaid, outlining detected packages:

![Package segmentation dataset sample for logistics](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/package-seg-sample.avif)

The dataset spans varied locations, environments, and package densities, so models trained on it see the range of real-world logistics scenes they need to generalize across. See the [segmentation task](../../tasks/segment.md) page for related workflows.
>>>>>>> origin/main

## Citations and Acknowledgments

If you integrate the Package Segmentation dataset into your research or development initiatives, please cite the source appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ factory_package_dataset,
            title = { factory_package Dataset },
            type = { Open Source Dataset },
            author = { factorypackage },
            url = { https://universe.roboflow.com/factorypackage/factory_package },
<<<<<<< HEAD
            journal = { Roboflow Universe },
            publisher = { Roboflow },
=======
>>>>>>> origin/main
            year = { 2024 },
            month = { jan },
            note = { visited on 2024-01-24 },
        }
        ```

<<<<<<< HEAD
We express our gratitude to the creators of the Package Segmentation dataset for their contribution to the computer vision community. For further exploration of datasets and model training, consider visiting our [Ultralytics Datasets](https://docs.ultralytics.com/datasets/) page and our guide on [model training tips](https://docs.ultralytics.com/guides/model-training-tips/).

## FAQ

### What is the Package Segmentation Dataset and how can it help in computer vision projects?

- The Package Segmentation Dataset is a curated collection of images tailored for tasks involving package [image segmentation](https://www.ultralytics.com/glossary/image-segmentation). It includes diverse images of packages in various contexts, making it invaluable for training and evaluating segmentation models. This dataset is particularly useful for applications in logistics, warehouse automation, and any project requiring precise package analysis.

### How do I train an Ultralytics YOLO11 model on the Package Segmentation Dataset?

- You can train an [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) model using both Python and CLI methods. Use the code snippets provided in the [Usage](#usage) section. Refer to the model [Training page](../../modes/train.md) for more details on arguments and configurations.

### What are the components of the Package Segmentation Dataset, and how is it structured?

- The dataset is structured into three main components:
    - **Training set**: Contains 1920 images with annotations.
    - **Testing set**: Comprises 89 images with corresponding annotations.
    - **Validation set**: Includes 188 images with annotations.
- This structure ensures a balanced dataset for thorough model training, validation, and testing, following best practices outlined in [model evaluation guides](https://docs.ultralytics.com/guides/model-evaluation-insights/).

### Why should I use Ultralytics YOLO11 with the Package Segmentation Dataset?

- Ultralytics YOLO11 provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and segmentation tasks. Using it with the Package Segmentation Dataset allows you to leverage YOLO11's capabilities for precise package segmentation, which is especially beneficial for industries like [logistics](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics) and warehouse automation.

### How can I access and use the package-seg.yaml file for the Package Segmentation Dataset?

- The `package-seg.yaml` file is hosted on Ultralytics' GitHub repository and contains essential information about the dataset's paths, classes, and configuration. You can view or download it at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml>. This file is crucial for configuring your models to utilize the dataset efficiently. For more insights and practical examples, explore our [Python Usage](https://docs.ultralytics.com/usage/python/) section.
=======
We express our gratitude to the creators of the Package Segmentation dataset for their contribution to the computer vision community. For more datasets, visit the [Ultralytics Datasets collection](../index.md) and our guide on [model training tips](../../guides/model-training-tips.md).

## FAQ

### What is the Package Segmentation Dataset, and how is it used in Ultralytics YOLO26?

The Package Segmentation Dataset is a collection of 2,197 annotated images of packages for training and evaluating [instance segmentation](../../tasks/segment.md) models on a single `package` class. It targets logistics and warehouse-automation applications like package identification, sorting, and quality control, and is used directly with Ultralytics [YOLO26](../../models/yolo26.md) via the `package-seg.yaml` configuration file.

### How many images and classes does the Package Segmentation Dataset contain?

The dataset totals 2,197 images — 1,920 for training, 188 for validation, and 89 for testing — all annotated for a single `package` class. The full archive downloads automatically as a ~103 MB `.zip` on first use.

### How do I train an Ultralytics YOLO26 model on the Package Segmentation Dataset?

Load a pretrained segmentation model (e.g., `yolo26n-seg.pt`) and train it with the `package-seg.yaml` configuration using the Python or CLI snippets in the [Usage](#usage) section above. See the [Training guide](../../modes/train.md) for the full list of available arguments.

### Why use Ultralytics YOLO26 for package segmentation in logistics?

YOLO26 provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and real-time speed for [instance segmentation](../../tasks/segment.md), letting automated systems detect and sort packages reliably even in dim or cluttered warehouses — see the [Applications](#applications) section above. Trained models export to formats like [ONNX](../../integrations/onnx.md) and [TensorRT](../../integrations/tensorrt.md) for deployment across warehouse hardware.

### Where can I find the dataset configuration file for Package Segmentation?

The `package-seg.yaml` file, which defines the dataset paths and the single `package` class, is located in the Ultralytics GitHub repository: [package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml).
>>>>>>> origin/main

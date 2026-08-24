---
<<<<<<< HEAD
comments: true
description: Discover HomeObjects-3K, a rich indoor object detection dataset with 12 classes like bed, sofa, TV, and laptop. Ideal for computer vision in smart homes, robotics, and AR.
keywords: HomeObjects-3K, indoor dataset, household items, object detection, computer vision, YOLO11, smart home AI, robotics dataset
=======
title: HomeObjects-3K Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com/
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: Train YOLO26 on HomeObjects-3K — 2,689 indoor images across 12 household classes like bed, sofa, TV, and laptop for smart home, robotics, and AR detection.
keywords: HomeObjects-3K, indoor dataset, household items, object detection, computer vision, YOLO26, smart home AI, robotics dataset
>>>>>>> origin/main
---

# HomeObjects-3K Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="HomeObjects-3K Dataset In Colab"></a>

<<<<<<< HEAD
The HomeObjects-3K dataset is a curated collection of common household object images, designed for training, testing, and [benchmarking](../../modes/benchmark.md) [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models. Featuring ~3,000 images and 12 distinct object classes, this dataset is ideal for research and applications in indoor scene understanding, smart home devices, [robotics](https://www.ultralytics.com/glossary/robotics), and augmented reality.
=======
The Ultralytics HomeObjects-3K dataset is an indoor [object detection](../../tasks/detect.md) dataset of 2,689 images (2,285 training and 404 validation) labeled across 12 everyday household classes — bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, and photo frame. Built for training and [benchmarking](../../modes/benchmark.md) [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, it targets indoor scene understanding, smart home devices, [robotics](https://www.ultralytics.com/glossary/robotics), and augmented reality.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/v3iqOYoRBFQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on HomeObjects-3K Dataset | Detection, Validation & ONNX Export 🚀
=======
  <strong>Watch:</strong> How to Train Ultralytics YOLO on HomeObjects-3K Dataset | Detection, Validation & ONNX Export 🚀
>>>>>>> origin/main
</p>

## Dataset Structure

<<<<<<< HEAD
The HomeObjects-3K dataset is organized into the following subsets:

- **Training Set**: Comprises 2,285 annotated images featuring objects such as sofas, chairs, tables, lamps, and more.
- **Validation Set**: Includes 404 annotated images designated for evaluating model performance.

Each image is labeled using bounding boxes aligned with the [Ultralytics YOLO](../detect/index.md/#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) format. The diversity of indoor lighting, object scale, and orientations makes it robust for real-world deployment scenarios.

## Object Classes

The dataset supports 12 everyday object categories, covering furniture, electronics, and decorative items. These classes are chosen to reflect common items encountered in indoor domestic environments and support vision tasks like [object detection](../../tasks/detect.md) and [object tracking](../../modes/track.md).

!!! Tip "HomeObjects-3K classes"
=======
The HomeObjects-3K dataset is split into two predefined subsets, defined by the `HomeObjects-3K.yaml` configuration:

| Split      | Images | Description                                      |
| ---------- | ------ | ------------------------------------------------ |
| Train      | 2,285  | Indoor scenes with labeled household objects     |
| Validation | 404    | Held-out images for evaluating model performance |

Each image is labeled with bounding boxes in the [Ultralytics YOLO](../detect/index.md#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) format, ready for [object detection](../../tasks/detect.md) and [tracking](../../modes/track.md) pipelines.

## Object Classes

The dataset covers 12 everyday object categories spanning furniture, electronics, and decorative items commonly found in indoor domestic environments:

!!! tip "HomeObjects-3K classes"
>>>>>>> origin/main

    0. bed
    1. sofa
    2. chair
    3. table
    4. lamp
    5. tv
    6. laptop
    7. wardrobe
    8. window
    9. door
    10. potted plant
    11. photo frame

## Applications

<<<<<<< HEAD
HomeObjects-3K enables a wide spectrum of applications in indoor computer vision, spanning both research and real-world product development:

- **Indoor object detection**: Use models like [Ultralytics YOLO11](../../models/yolo11.md) to find and locate common home items like beds, chairs, lamps, and laptops in images. This helps with real-time understanding of indoor scenes.

- **Scene layout parsing**: In robotics and smart home systems, this helps devices understand how rooms are arranged, where objects like doors, windows, and furniture are, so they can navigate safely and interact with their environment properly.

- **AR applications**: Power [object recognition](http://ultralytics.com/glossary/image-recognition) features in apps that use augmented reality. For example, detect TVs or wardrobes and show extra information or effects on them.

- **Education and research**: Support learning and academic projects by giving students and researchers a ready-to-use dataset for practicing indoor object detection with real-world examples.

- **Home inventory and asset tracking**: Automatically detect and list home items in photos or videos, useful for managing belongings, organizing spaces, or visualizing furniture in real estate.

## Dataset YAML

The configuration for the HomeObjects-3K dataset is provided through a YAML file. This file outlines essential information such as image paths for train and validation directories, and the list of object classes.
You can access the `HomeObjects-3K.yaml` file directly from the Ultralytics repository at: [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml)
=======
HomeObjects-3K supports a range of indoor computer vision applications across research and product development:

- **Indoor object detection**: Use models like [Ultralytics YOLO26](../../models/yolo26.md) to find and locate common home items like beds, chairs, lamps, and laptops in images for real-time understanding of indoor scenes.

- **Scene layout parsing**: Help robotics and smart home systems understand how rooms are arranged — where doors, windows, and furniture sit — so devices can navigate safely and interact with their environment.

- **AR applications**: Power [object recognition](https://www.ultralytics.com/glossary/image-recognition) features in augmented reality apps. For example, detect TVs or wardrobes and overlay extra information or effects on them.

- **Education and research**: Give students and researchers a ready-to-use dataset for practicing indoor object detection with real-world examples.

- **Home inventory and asset tracking**: Automatically detect and list home items in photos or videos, useful for managing belongings, organizing spaces, or visualizing furniture in real estate.

To label your own indoor images, train, and manage dataset versions in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The `HomeObjects-3K.yaml` file defines the dataset configuration — the train and validation image paths and the list of object classes. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/HomeObjects-3K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/HomeObjects-3K.yaml"
    ```

## Usage

<<<<<<< HEAD
You can train a YOLO11n model on the HomeObjects-3K dataset for 100 epochs using an image size of 640. The examples below show how to get started. For more training options and detailed settings, check the [Training](../../modes/train.md) guide.
=======
You can train a YOLO26n model on the HomeObjects-3K dataset for 100 epochs using an image size of 640. The dataset (390 MB) downloads automatically on first use. The examples below show how to get started. For more training options and detailed settings, check the [Training](../../modes/train.md) guide.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
<<<<<<< HEAD
        model = YOLO("yolo11n.pt")
=======
        model = YOLO("yolo26n.pt")
>>>>>>> origin/main

        # Train the model on HomeObjects-3K dataset
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=HomeObjects-3K.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

## Sample Images and Annotations

<<<<<<< HEAD
The dataset features a rich collection of indoor scene images that capture a wide range of household objects in natural home environments. Below are sample visuals from the dataset, each paired with its corresponding annotations to illustrate object positions, scales, and spatial relationships.

![HomeObjects-3K dataset sample image, highlighting different objects i.e, beds, chair, door, sofas, and plants](https://github.com/ultralytics/docs/releases/download/0/homeobjects-3k-dataset-sample.avif)
=======
The sample below shows an indoor scene from the dataset with its bounding-box annotations, illustrating the object positions, scales, and spatial relationships that models learn to detect.

![HomeObjects-3K dataset sample with household objects](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/homeobjects-3k-dataset-sample.avif)
>>>>>>> origin/main

## License and Attribution

HomeObjects-3K is developed and released by the **[Ultralytics team](https://www.ultralytics.com/about)** under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), supporting open-source research and commercial use with proper attribution.

If you use this dataset in your research, please cite it using the mentioned details:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2025,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {May},
            title = {Ultralytics Datasets: HomeObjects-3K Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/homeobjects-3k/},
            version = {1.0.0},
            year = {2025}
        }
        ```

## FAQ

### What is the HomeObjects-3K dataset designed for?

<<<<<<< HEAD
HomeObjects-3K is crafted for advancing AI understanding of indoor scenes. It focuses on detecting everyday household items—like beds, sofas, TVs, and lamps—making it ideal for applications in smart homes, robotics, augmented reality, and interior monitoring systems. Whether you're training models for real-time edge devices or academic research, this dataset provides a balanced foundation.

### Which object categories are included, and why were they selected?

The dataset includes 12 of the most commonly encountered household items: bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, and photo frame. These objects were chosen to reflect realistic indoor environments and to support multipurpose tasks such as robotic navigation, or scene generation in AR/VR applications.

### How can I train a YOLO model using the HomeObjects-3K dataset?

To train a YOLO model like YOLO11n, you'll just need the `HomeObjects-3K.yaml` configuration file and the [pretrained model](../../models/index.md) weights. Whether you're using Python or the CLI, training can be launched with a single command. You can customize parameters such as epochs, image size, and batch size depending on your target performance and hardware setup.
=======
HomeObjects-3K is designed for detecting everyday household items — like beds, sofas, TVs, and lamps — in indoor scenes. This makes it well suited to smart homes, robotics, augmented reality, and interior monitoring systems, for both real-time edge deployment and academic research.

### How many images and classes are in the HomeObjects-3K dataset?

HomeObjects-3K contains 2,689 images total — 2,285 for training and 404 for validation — with no separate test split. Every image is labeled across 12 object classes: bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, and photo frame.

### Which object categories are included, and why were they selected?

The dataset includes 12 of the most commonly encountered household items: bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, and photo frame. These objects were chosen to reflect realistic indoor environments and to support tasks such as robotic navigation and scene understanding in AR/VR applications.

### How do I download the HomeObjects-3K dataset?

The dataset (390 MB) downloads automatically the first time you train with `data="HomeObjects-3K.yaml"` — no manual step is required. Ultralytics fetches the images and labels and unpacks them to your local datasets directory. You can browse related datasets in the [detection datasets overview](index.md).

### How can I train a YOLO model using the HomeObjects-3K dataset?

To train a YOLO model like YOLO26n, you need the `HomeObjects-3K.yaml` configuration file and the [pretrained model](../../models/index.md) weights. Training launches with a single Python or CLI command, and you can customize parameters such as epochs, image size, and batch size for your target performance and hardware.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
<<<<<<< HEAD
        model = YOLO("yolo11n.pt")
=======
        model = YOLO("yolo26n.pt")
>>>>>>> origin/main

        # Train the model on HomeObjects-3K dataset
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=HomeObjects-3K.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

### Is this dataset suitable for beginner-level projects?

<<<<<<< HEAD
Absolutely. With clean labeling, and standardized YOLO-compatible annotations, HomeObjects-3K is an excellent entry point for students and hobbyists who want to explore real-world object detection in indoor scenarios. It also scales well for more complex applications in commercial environments.
=======
Yes. Its standardized YOLO-format annotations and compact size make HomeObjects-3K a strong entry point for students and hobbyists exploring real-world object detection in indoor scenarios.
>>>>>>> origin/main

### Where can I find the annotation format and YAML?

Refer to the [Dataset YAML](#dataset-yaml) section. The format is standard YOLO, making it compatible with most object detection pipelines.

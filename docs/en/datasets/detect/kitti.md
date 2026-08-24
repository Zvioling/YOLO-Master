---
<<<<<<< HEAD
comments: true
description: Explore the Ultralytics kitti dataset, a benchmark dataset for computer vision tasks such as 3D object detection, depth estimation, and autonomous driving perception.
keywords: kitti, Ultralytics, dataset, object detection, 3D vision, YOLO11, training, validation, self-driving cars, computer vision
=======
title: KITTI Detection Dataset
comments: true
creator:
    name: KITTI
    url: https://www.cvlibs.net/datasets/kitti/
license:
    name: CC-BY-NC-SA-3.0
description: Ultralytics KITTI is a 2D object detection dataset for autonomous driving with 7,481 annotated images across 8 classes like car, pedestrian, and cyclist.
keywords: KITTI dataset, autonomous driving, 2D object detection, self-driving cars, YOLO26, computer vision, vehicle detection, pedestrian detection
>>>>>>> origin/main
---

# KITTI Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open KITTI Dataset In Colab"></a>

<<<<<<< HEAD
The kitti dataset is one of the most influential benchmark datasets for autonomous driving and computer vision. Released by the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago, it contains stereo camera, LiDAR, and GPS/IMU data collected from real-world driving scenarios.
=======
The Ultralytics KITTI dataset is a 2D [object detection](../../tasks/detect.md) dataset for autonomous driving, containing 7,481 annotated images (5,985 for training and 1,496 for validation) across 8 classes — car, van, truck, pedestrian, person_sitting, cyclist, tram, and misc. Released by the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago, its images come from real-world urban, rural, and highway driving scenes.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NNeDlTbq9pA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on the KITTI Dataset 🚀
</p>

It is widely used for evaluating algorithms in object detection, depth estimation, optical flow, and visual odometry. The dataset is fully compatible with Ultralytics YOLO11 for 2D object detection tasks and can be easily integrated into the Ultralytics platform for training and evaluation.
=======
  <strong>Watch:</strong> How to Train Ultralytics YOLO on the KITTI Dataset | Object Detection, Inference & ONNX Export 🚀
</p>

The broader KITTI Vision Benchmark Suite also spans depth estimation, optical flow, stereo vision, and visual odometry, but the Ultralytics `kitti.yaml` configuration here is set up for 2D object detection and is fully compatible with [Ultralytics YOLO26](../../models/yolo26.md).
>>>>>>> origin/main

## Dataset Structure

!!! warning

<<<<<<< HEAD
    Kitti original test set is excluded here since it does not contain ground-truth annotations.

In total, the dataset includes 7,481 images, each paired with detailed annotations for objects such as cars, pedestrians, cyclists, and other road elements. The dataset is divided into two main subsets:

- **Training set:** Contains 5,985 images with annotated labels used for model training.
- **Validation set:** Includes 1,496 images with corresponding annotations used for performance evaluation and benchmarking.

## Applications

Kitti dataset enables advancements in autonomous driving and robotics, supporting tasks like:

- **Autonomous vehicle perception**: Training models to detect and track vehicles, pedestrians, and obstacles for safe navigation in self-driving systems.
- **3D scene understanding**: Supporting depth estimation, stereo vision, and 3D object localization to help machines understand spatial environments.
- **Optical flow and motion prediction**: Enabling motion analysis to predict the movement of objects and improve trajectory planning in dynamic environments.
- **Computer vision benchmarking**: Serving as a standard benchmark for evaluating performance across multiple vision tasks, including object detection, and tracking.

## Dataset YAML

Ultralytics defines the kitti dataset configuration using a YAML file. This file specifies dataset paths, class labels, and metadata required for training. The configuration file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).
=======
    The original KITTI test set is excluded here because it has no public ground-truth annotations.

The dataset contains 7,481 annotated images covering objects such as cars, pedestrians, and cyclists, split into two predefined subsets defined by the `kitti.yaml` configuration:

| Split      | Images | Description                                     |
| ---------- | ------ | ----------------------------------------------- |
| Train      | 5,985  | Labeled images for model training               |
| Validation | 1,496  | Held-out images for evaluation and benchmarking |

## Object Classes

The `kitti.yaml` file defines 8 object classes spanning vehicles, people, and other road users commonly seen in driving scenes:

!!! tip "KITTI classes"

    0. car
    1. van
    2. truck
    3. pedestrian
    4. person_sitting
    5. cyclist
    6. tram
    7. misc

## Applications

The KITTI dataset supports a range of 2D detection applications in autonomous driving and robotics:

- **Autonomous vehicle perception**: Train models to detect and track cars, pedestrians, and cyclists so self-driving systems can navigate safely.
- **ADAS development**: Build driver-assistance features such as collision warning and pedestrian detection on real driving footage.
- **Traffic and road-scene analysis**: Detect and count vehicles and road users to study traffic flow and road safety.
- **Computer vision benchmarking**: Use KITTI as a standard benchmark for evaluating 2D [object detection](../../tasks/detect.md) and [tracking](../../modes/track.md) models.

To label your own driving imagery, train, and manage dataset versions in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

Ultralytics defines the KITTI dataset configuration using a YAML file. This file specifies the dataset paths, class labels, and metadata required for training. The configuration file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

## Usage

<<<<<<< HEAD
To train a YOLO11n model on the kitti dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following commands. For more details, refer to the [Training](../../modes/train.md) page.
=======
To train a YOLO26n model on the KITTI dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following commands. The dataset (390.5 MB) downloads automatically on first use. For more details, refer to the [Training](../../modes/train.md) page.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load a pretrained YOLO11 model
        model = YOLO("yolo11n.pt")
=======
        # Load a pretrained YOLO26 model
        model = YOLO("yolo26n.pt")
>>>>>>> origin/main

        # Train on kitti dataset
        results = model.train(data="kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo detect train data=kitti.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=kitti.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

You can also perform evaluation, [inference](../../modes/predict.md), and [export](../../modes/export.md) tasks directly from the command line or Python API using the same configuration file.

## Sample Images and Annotations

<<<<<<< HEAD
The kitti dataset provides diverse driving scenarios. Each image includes bounding box annotations for 2D object detection tasks. The example showcase the dataset rich variety, enabling robust model generalization across diverse real-world conditions.

<img src="https://github.com/ultralytics/docs/releases/download/0/kitti-dataset-sample.avif" alt="Kitti sample image" width="800">

## Citations and Acknowledgments

If you use the kitti dataset in your research, please cite the following paper:
=======
The sample below shows a driving scene from the dataset with its 2D bounding-box annotations. KITTI images span urban, rural, and highway scenes captured in real traffic, giving models varied object scales, viewpoints, and lighting.

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/kitti-dataset-sample.avif" alt="KITTI dataset vehicle detection sample" width="800">

## Citations and Acknowledgments

If you use the KITTI dataset in your research, please cite the following paper:
>>>>>>> origin/main

!!! quote

    === "BibTeX"

        ```bibtex
        @article{Geiger2013IJRR,
          author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
          title = {Vision meets Robotics: The KITTI Dataset},
          journal = {International Journal of Robotics Research (IJRR)},
          year = {2013}
        }
        ```

<<<<<<< HEAD
We acknowledge the KITTI Vision Benchmark Suite for providing this comprehensive dataset that continues to shape progress in computer vision, robotics, and autonomous systems. Visit the [kitti website](https://www.cvlibs.net/datasets/kitti/) for more information.

## FAQs

### What is the kitti dataset used for?

The kitti dataset is primarily used for computer vision research in autonomous driving, supporting tasks like object detection, depth estimation, optical flow, and 3D localization.

### How many images are included in the kitti dataset?

The dataset includes 5,985 labeled training images and 1,496 validation images captured across urban, rural, and highway scenes. The original test set is excluded here since it does not contain ground-truth annotations.

### Which object classes are annotated in the dataset?

kitti includes annotations for objects such as cars, pedestrians, cyclists, trucks, trams, and miscellaneous road users.

### Can I train Ultralytics YOLO11 models using the kitti dataset?

Yes, kitti is fully compatible with Ultralytics YOLO11. You can [train](../../modes/train.md) and [validate](../../modes/val.md), models directly using the provided YAML configuration file.

### Where can I find the kitti dataset configuration file?

You can access the YAML file at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).
=======
We acknowledge the KITTI Vision Benchmark Suite for providing this comprehensive dataset that continues to shape progress in computer vision, robotics, and autonomous systems. Visit the [KITTI website](https://www.cvlibs.net/datasets/kitti/) for more information.

## FAQ

### What is the KITTI dataset used for?

The Ultralytics KITTI dataset is used to train and evaluate 2D object detection models for autonomous driving. It provides 7,481 annotated images across 8 classes, including cars, pedestrians, and cyclists, and is widely used for benchmarking perception models.

### How many images and classes are in the KITTI dataset?

The Ultralytics KITTI configuration contains 7,481 images — 5,985 for training and 1,496 for validation — with no separate test split. Each image is annotated across 8 classes: car, van, truck, pedestrian, person_sitting, cyclist, tram, and misc.

### Does the KITTI dataset include a test split?

No. The Ultralytics KITTI configuration provides only train (5,985 images) and validation (1,496 images) splits. The original KITTI test set is excluded because it has no public ground-truth annotations.

### How do I download the KITTI dataset?

The dataset (390.5 MB) downloads automatically the first time you train with `data="kitti.yaml"` — no manual step is required. Ultralytics fetches the images and labels and unpacks them to your local datasets directory. You can browse related datasets in the [detection datasets overview](index.md).

### Can I train Ultralytics YOLO26 models using the KITTI dataset?

Yes, KITTI is fully compatible with Ultralytics YOLO26. You can [train](../../modes/train.md) and [validate](../../modes/val.md) models directly using the provided YAML configuration file.

### Where can I find the KITTI dataset configuration file?

You can access the `kitti.yaml` file at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml). It defines the dataset paths and the 8 class names used for training.
>>>>>>> origin/main

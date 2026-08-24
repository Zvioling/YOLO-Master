---
<<<<<<< HEAD
comments: true
description: Explore Ultralytics Tiger-Pose dataset with 263 diverse images. Ideal for testing, training, and refining pose estimation algorithms.
keywords: Ultralytics, Tiger-Pose, dataset, pose estimation, YOLO11, training data, machine learning, neural networks
=======
title: Tiger-Pose Estimation Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com/
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: "Explore the Ultralytics Tiger-Pose dataset: 263 images (210 train / 53 val) with 12 keypoints per tiger, ideal for testing pose estimation pipelines."
keywords: Ultralytics, Tiger-Pose, dataset, pose estimation, YOLO26, training data, machine learning, neural networks
>>>>>>> origin/main
---

# Tiger-Pose Dataset

## Introduction

<<<<<<< HEAD
[Ultralytics](https://www.ultralytics.com/) introduces the Tiger-Pose dataset, a versatile collection designed for pose estimation tasks. This dataset comprises 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0), with 210 images allocated for training and 53 for validation. It serves as an excellent resource for testing and troubleshooting pose estimation algorithms.

Despite its manageable training split of 210 images, the Tiger-Pose dataset offers diversity, making it suitable for assessing training pipelines, identifying potential errors, and serving as a valuable preliminary step before working with larger datasets for [pose estimation](https://docs.ultralytics.com/tasks/pose/).

This dataset is intended for use with [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).
=======
[Ultralytics](https://www.ultralytics.com/) introduces the Tiger-Pose dataset, a versatile collection designed for pose estimation tasks. This dataset comprises 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U), with 210 images allocated for training and 53 for validation. It serves as an excellent resource for testing and troubleshooting pose estimation algorithms.

Despite its manageable training split of 210 images, the Tiger-Pose dataset offers diversity, making it suitable for assessing training pipelines, identifying potential errors, and serving as a valuable preliminary step before working with larger datasets for [pose estimation](../../tasks/pose.md).

Once your pipeline trains clean on this small set, swap in your own animal or object keypoints and scale up training on [Ultralytics Platform](https://platform.ultralytics.com/) without leaving the browser.
>>>>>>> origin/main

## Dataset Structure

- **Total images**: 263 (210 train / 53 val).
- **Keypoints**: 12 per tiger (no visibility flag).
<<<<<<< HEAD
=======
- **Download size**: ~49.8 MB.
>>>>>>> origin/main
- **Directory layout**: YOLO-format keypoints stored under `labels/{train,val}` alongside `images/{train,val}` directories.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> Train YOLO11 Pose Model on Tiger-Pose Dataset Using Ultralytics HUB
=======
  <strong>Watch:</strong> Train an Ultralytics YOLO Pose Model on the Tiger-Pose Dataset
>>>>>>> origin/main
</p>

## Dataset YAML

<<<<<<< HEAD
A YAML (Yet Another Markup Language) file serves as the means to specify the configuration details of a dataset. It encompasses crucial data such as file paths, class definitions, and other pertinent information. Specifically, for the `tiger-pose.yaml` file, you can check [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).
=======
A YAML file serves as the means to specify the configuration details of a dataset. It encompasses crucial data such as file paths, class definitions, and other pertinent information. Specifically, for the `tiger-pose.yaml` file, you can check [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## Usage

<<<<<<< HEAD
To train a YOLO11n-pose model on the Tiger-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
=======
To train a YOLO26n-pose model on the Tiger-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo pose train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
=======
        yolo pose train data=tiger-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

## Sample Images and Annotations

Here are some examples of images from the Tiger-Pose dataset, along with their corresponding annotations:

<<<<<<< HEAD
<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-4.avif" alt="Dataset sample image" width="100%">
=======
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-4.avif" alt="Tiger pose estimation dataset mosaic training batch" width="100%">
>>>>>>> origin/main

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Tiger-Pose dataset and the benefits of using mosaicing during the training process.

## Inference Example

<<<<<<< HEAD
=======
After training, load your best checkpoint and run inference on new images or video — see the [Prediction](../../modes/predict.md) page for the full list of arguments.

>>>>>>> origin/main
!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a tiger-pose trained model

        # Run inference
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```

    === "CLI"

        ```bash
        # Run inference using a tiger-pose trained model
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

## Citations and Acknowledgments

<<<<<<< HEAD
The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
=======
Ultralytics releases the Tiger-Pose dataset annotations under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). The source video remains subject to its [original terms](https://www.youtube.com/watch?v=MIBAT6BGE6U), which should be reviewed before using or redistributing extracted frames.
>>>>>>> origin/main

## FAQ

### What is the Ultralytics Tiger-Pose dataset used for?

<<<<<<< HEAD
The Ultralytics Tiger-Pose dataset is designed for pose estimation tasks, consisting of 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0). The dataset is divided into 210 training images and 53 validation images. It is particularly useful for testing, training, and refining pose estimation algorithms using [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

### How do I train a YOLO11 model on the Tiger-Pose dataset?

To train a YOLO11n-pose model on the Tiger-Pose dataset for 100 epochs with an image size of 640, use the following code snippets. For more details, visit the [Training](../../modes/train.md) page:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### What configurations does the `tiger-pose.yaml` file include?

The `tiger-pose.yaml` file is used to specify the configuration details of the Tiger-Pose dataset. It includes crucial data such as file paths and class definitions. To see the exact configuration, you can check out the [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).

### How can I run inference using a YOLO11 model trained on the Tiger-Pose dataset?

To perform inference using a YOLO11 model trained on the Tiger-Pose dataset, you can use the following code snippets. For a detailed guide, visit the [Prediction](../../modes/predict.md) page:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a tiger-pose trained model

        # Run inference
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```


    === "CLI"

        ```bash
        # Run inference using a tiger-pose trained model
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

### What are the benefits of using the Tiger-Pose dataset for pose estimation?

The Tiger-Pose dataset, despite its manageable size of 210 images for training, provides a diverse collection of images that are ideal for testing pose estimation pipelines. The dataset helps identify potential errors and acts as a preliminary step before working with larger datasets. Additionally, the dataset supports the training and refinement of pose estimation algorithms using advanced tools like [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics), enhancing model performance and [accuracy](https://www.ultralytics.com/glossary/accuracy).
=======
The Ultralytics Tiger-Pose dataset is designed for pose estimation tasks, consisting of 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U). The dataset is divided into 210 training images and 53 validation images, making it well-suited for testing, training, and refining pose estimation algorithms.

### How do I train a YOLO26 model on the Tiger-Pose dataset?

Load `yolo26n-pose.pt` and call `model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)` — see the [Train Example](#usage) above for the full Python and CLI snippets, and the [Training](../../modes/train.md) page for a comprehensive list of arguments.

### What configurations does the `tiger-pose.yaml` file include?

The `tiger-pose.yaml` file defines the dataset path, train/val image directories, a single class (`tiger`), and `kpt_shape: [12, 2]` — 12 keypoints per instance with no visibility flag. See the [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml) for the exact configuration.

### How can I run inference using a YOLO26 model trained on the Tiger-Pose dataset?

Load your trained checkpoint (e.g., `path/to/best.pt`) and call `model.predict(source=..., show=True)` — see the [Inference Example](#inference-example) above for the full Python and CLI snippets, and the [Prediction](../../modes/predict.md) page for a comprehensive list of arguments.

### What are the benefits of using the Tiger-Pose dataset for pose estimation?

With 263 total images (210 train / 53 val), 1 class, 12 keypoints per instance, and a ~49.8 MB download, Tiger-Pose is small enough to manage quickly yet diverse enough to sanity-check a pose training pipeline and identify errors before working with larger datasets.
>>>>>>> origin/main

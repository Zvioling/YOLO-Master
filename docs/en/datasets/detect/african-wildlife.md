---
<<<<<<< HEAD
comments: true
description: Explore our African Wildlife Dataset featuring images of buffalo, elephant, rhino, and zebra for training computer vision models. Ideal for research and conservation.
keywords: African Wildlife Dataset, South African animals, object detection, computer vision, YOLO11, wildlife research, conservation, dataset
=======
title: African Wildlife Detection Dataset
comments: true
creator:
    name: Bianca Ferreira
    type: Person
    url: https://www.kaggle.com/datasets/biancaferreira/african-wildlife
license:
    name: None
description: Train YOLO object detection models on the African Wildlife Dataset — 1,504 images across 4 classes (buffalo, elephant, rhino, zebra) with automatic download.
keywords: African Wildlife Dataset, object detection, computer vision, YOLO26, buffalo, elephant, rhino, zebra, wildlife conservation, animal detection
>>>>>>> origin/main
---

# African Wildlife Dataset

<<<<<<< HEAD
This dataset showcases four common animal classes typically found in South African nature reserves. It includes images of African wildlife such as buffalo, elephant, rhino, and zebra, providing valuable insights into their characteristics. Essential for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) algorithms, this dataset aids in identifying animals in various habitats, from zoos to forests, and supports wildlife research.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/biIW5Z6GYl0"
=======
The Ultralytics African Wildlife Dataset is an [object detection](../../tasks/detect.md) dataset of 1,504 images across 4 animal classes — buffalo, elephant, rhino, and zebra — commonly found in South African nature reserves. The images are pre-split into 1,052 training, 225 validation, and 227 test images, and the dataset downloads automatically (~100 MB) the first time you train. It is a compact, ready-to-use benchmark for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models for wildlife monitoring, conservation, and ecological research.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EXYB-dbgJjY"
>>>>>>> origin/main
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> African Wildlife Animals Detection using Ultralytics YOLO11
=======
  <strong>Watch:</strong> How to Train Ultralytics YOLO26 on the African Wildlife Dataset | Inference, Metrics & ONNX Export 🐘
>>>>>>> origin/main
</p>

## Dataset Structure

<<<<<<< HEAD
The African wildlife objects detection dataset is split into three subsets:

- **Training set**: Contains 1052 images, each with corresponding annotations.
- **Validation set**: Includes 225 images, each with paired annotations.
- **Testing set**: Comprises 227 images, each with paired annotations.

## Applications

This dataset can be applied in various computer vision tasks such as [object detection](https://www.ultralytics.com/glossary/object-detection), object tracking, and research. Specifically, it can be used to train and evaluate models for identifying African wildlife objects in images, which can have applications in wildlife conservation, ecological research, and monitoring efforts in natural reserves and protected areas. Additionally, it can serve as a valuable resource for educational purposes, enabling students and researchers to study and understand the characteristics and behaviors of different animal species.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths, classes, and other pertinent details. For the African wildlife dataset, the `african-wildlife.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml).
=======
The Ultralytics African Wildlife Dataset contains **1,504 images** across **4 classes** (buffalo, elephant, rhino, and zebra), pre-split into three subsets:

- **Training set**: 1,052 images, each with corresponding annotations.
- **Validation set**: 225 images, each with paired annotations.
- **Testing set**: 227 images, each with paired annotations.

!!! tip "Automatic download"

    The African Wildlife Dataset (~100 MB) downloads automatically the first time you start training, so no manual download or preparation is required.

Explore [African Wildlife on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/african-wildlife) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

The Ultralytics African Wildlife Dataset supports a range of [object detection](../../tasks/detect.md) applications:

- **Wildlife conservation** — detect and count buffalo, elephant, rhino, and zebra to support [animal population monitoring](../../solutions/index.md) in nature reserves and protected areas.
- **Ecological research** — study species distribution and behavior across different habitats.
- **Anti-poaching surveillance** — flag animals in camera-trap or drone footage over large protected areas.
- **Education and prototyping** — a compact four-class dataset for learning [model training](../../modes/train.md) and [prediction](../../modes/predict.md).

## Dataset YAML

A YAML file defines the dataset configuration, including paths, classes, and other pertinent details. For the African Wildlife Dataset, the `african-wildlife.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/african-wildlife.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/african-wildlife.yaml"
    ```

## Usage

<<<<<<< HEAD
To train a YOLO11n model on the African wildlife dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.
=======
To train a YOLO26n model on the African Wildlife Dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # Train the model
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=african-wildlife.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

=======
        yolo detect train data=african-wildlife.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

Once trained, run [inference](../../modes/predict.md) with the fine-tuned model on new images:

>>>>>>> origin/main
!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load an African wildlife fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/african-wildlife-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/african-wildlife-sample.jpg"
        ```

## Sample Images and Annotations

<<<<<<< HEAD
The African wildlife dataset comprises a wide variety of images showcasing diverse animal species and their natural habitats. Below are examples of images from the dataset, each accompanied by its corresponding annotations.

![African wildlife dataset sample image](https://github.com/ultralytics/docs/releases/download/0/african-wildlife-dataset-sample.avif)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different object sizes, aspect ratios, and contexts.

This example illustrates the variety and complexity of images in the African wildlife dataset, emphasizing the benefits of including mosaicing during the training process.

## Citations, License and Acknowledgments

We'd like to thank the original dataset author, [Bianca Ferreira](https://www.kaggle.com/biancaferreira/datasets), for releasing this dataset to the community. The Ultralytics team has updated and adapted it internally so it can be used seamlessly with [Ultralytics YOLO](https://www.ultralytics.com/yolo) models. This dataset is available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
=======
The African Wildlife Dataset comprises a wide variety of images showcasing diverse animal species and their natural habitats. Below are examples of images from the dataset, each accompanied by its corresponding annotations.

![African wildlife dataset sample image](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/african-wildlife-dataset-sample.avif)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different object sizes, aspect ratios, and contexts.

## Citations, License and Acknowledgments

We'd like to thank the original dataset author, [Bianca Ferreira](https://www.kaggle.com/datasets/biancaferreira/african-wildlife), for releasing this dataset to the community. The Ultralytics team has updated and adapted it internally so it can be used seamlessly with [Ultralytics YOLO](https://www.ultralytics.com/yolo) models. The source dataset does not specify a license.
>>>>>>> origin/main

If you use this dataset in your research, please cite it using the mentioned details:

!!! quote ""

    === "BibTeX"

        ```bibtex

        @dataset{Ferreira_African_Wildlife_Ultralytics_Adaptation_2024,
            author  = {Ferreira, Bianca},
            title   = {African Wildlife Detection Dataset (Ultralytics YOLO Adaptation)},
            url     = {https://docs.ultralytics.com/datasets/detect/african-wildlife/},
            note    = {Original dataset by Bianca Ferreira; adapted for Ultralytics YOLO by Glenn Jocher and Muhammad Rizwan Munawar},
<<<<<<< HEAD
            license = {AGPL-3.0},
=======
            license = {Not specified},
>>>>>>> origin/main
            version = {1.0.0},
            year    = {2024}
        }
        ```

## FAQ

### What is the African Wildlife Dataset, and how can it be used in computer vision projects?

<<<<<<< HEAD
The African Wildlife Dataset includes images of four common animal species found in South African nature reserves: buffalo, elephant, rhino, and zebra. It is a valuable resource for training computer vision algorithms in object detection and animal identification. The dataset supports various tasks like object tracking, research, and conservation efforts. For more information on its structure and applications, refer to the [Dataset Structure](#dataset-structure) section and [Applications](#applications) of the dataset.

### How do I train a YOLO11 model using the African Wildlife Dataset?

You can train a YOLO11 model on the African Wildlife Dataset by using the `african-wildlife.yaml` configuration file. Below is an example of how to train the YOLO11n model for 100 epochs with an image size of 640:
=======
The African Wildlife Dataset is an [object detection](../../tasks/detect.md) dataset of 1,504 images across 4 animal classes — buffalo, elephant, rhino, and zebra — found in South African nature reserves. It is used to train and evaluate models for identifying African wildlife in images, which supports wildlife conservation, ecological research, and monitoring in natural reserves. It also serves as an accessible resource for students and researchers studying computer vision.

### How many images and classes are in the African Wildlife Dataset?

The Ultralytics African Wildlife Dataset contains 1,504 images across 4 classes: buffalo, elephant, rhino, and zebra. The images are pre-split into 1,052 training, 225 validation, and 227 test images, and the dataset downloads automatically (~100 MB) the first time you train.

### How do I train a YOLO26 model using the African Wildlife Dataset?

You can train a YOLO26 model on the African Wildlife Dataset by using the `african-wildlife.yaml` configuration file. Below is an example of how to train the YOLO26n model for 100 epochs with an image size of 640:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # Train the model
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=african-wildlife.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=african-wildlife.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

For additional training parameters and options, refer to the [Training](../../modes/train.md) documentation.

### Where can I find the YAML configuration file for the African Wildlife Dataset?

<<<<<<< HEAD
The YAML configuration file for the African Wildlife Dataset, named `african-wildlife.yaml`, can be found at [this GitHub link](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml). This file defines the dataset configuration, including paths, classes, and other details crucial for training [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models. See the [Dataset YAML](#dataset-yaml) section for more details.

### Can I see sample images and annotations from the African Wildlife Dataset?

Yes, the African Wildlife Dataset includes a wide variety of images showcasing diverse animal species in their natural habitats. You can view sample images and their corresponding annotations in the [Sample Images and Annotations](#sample-images-and-annotations) section. This section also illustrates the use of mosaicing technique to combine multiple images into one for enriched batch diversity, enhancing the model's generalization ability.

### How can the African Wildlife Dataset be used to support wildlife conservation and research?

The African Wildlife Dataset is ideal for supporting wildlife conservation and research by enabling the training and evaluation of models to identify African wildlife in different habitats. These models can assist in [monitoring animal populations](https://docs.ultralytics.com/solutions/), studying their behavior, and recognizing conservation needs. Additionally, the dataset can be utilized for educational purposes, helping students and researchers understand the characteristics and behaviors of different animal species. More details can be found in the [Applications](#applications) section.
=======
The YAML configuration file for the African Wildlife Dataset, named `african-wildlife.yaml`, can be found at [this GitHub link](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml). This file defines the dataset configuration, including paths, classes, and other details crucial for training [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models.

### What license is the African Wildlife Dataset released under?

The [source dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) does not specify a license. It was originally published on Kaggle by Bianca Ferreira and adapted by Ultralytics for seamless use with [Ultralytics YOLO](https://www.ultralytics.com/yolo) models. If you use the dataset in your research, please cite it using the BibTeX entry in the [Citations](#citations-license-and-acknowledgments) section.
>>>>>>> origin/main

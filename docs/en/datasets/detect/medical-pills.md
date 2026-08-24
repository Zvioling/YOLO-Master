---
<<<<<<< HEAD
comments: true
description: Explore the medical-pills detection dataset with labeled images. Essential for training AI models for pharmaceutical identification and automation.
keywords: medical-pills dataset, pill detection, pharmaceutical imaging, AI in healthcare, computer vision, object detection, medical automation, dataset for training
=======
title: Medical Pills Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com/
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: The Medical Pills dataset provides 115 labeled images across one class (pill) for training Ultralytics YOLO object detection models in pharmaceutical automation.
keywords: Medical Pills dataset, pill detection, pharmaceutical imaging, AI in healthcare, computer vision, object detection, YOLO26, medical automation
>>>>>>> origin/main
---

# Medical Pills Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Medical Pills Dataset In Colab"></a>

<<<<<<< HEAD
The medical-pills detection dataset is a proof-of-concept (POC) dataset, carefully curated to demonstrate the potential of AI in pharmaceutical applications. It contains labeled images specifically designed to train [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) [models](https://docs.ultralytics.com/models/) for identifying medical-pills.
=======
The Ultralytics **Medical Pills** dataset is a proof-of-concept (POC) [object detection](../../tasks/detect.md) dataset of 115 labeled images across a single class, `pill` — 92 for training and 23 for validation. It is built to demonstrate [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) [models](../../models/index.md) for pharmaceutical applications such as quality control, packaging automation, and sorting.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/8gePl_Zcs5c"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to train Ultralytics YOLO11 Model on Medical Pills Detection Dataset in <a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb">Google Colab</a>
</p>

This dataset serves as a foundational resource for automating essential [tasks](https://docs.ultralytics.com/tasks/) such as quality control, packaging automation, and efficient sorting in pharmaceutical workflows. By integrating this dataset into projects, researchers and developers can explore innovative [solutions](https://docs.ultralytics.com/solutions/) that enhance [accuracy](https://www.ultralytics.com/glossary/accuracy), streamline operations, and ultimately contribute to improved healthcare outcomes.

## Dataset Structure

The medical-pills dataset is divided into two subsets:

- **Training set**: Consisting of 92 images, each annotated with the class `pill`.
- **Validation set**: Comprising 23 images with corresponding annotations.

## Applications

Using computer vision for medical-pills detection enables automation in the pharmaceutical industry, supporting tasks like:
=======
  <strong>Watch:</strong> How to train an Ultralytics YOLO Model on the Medical Pills Detection Dataset in <a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb">Google Colab</a>
</p>

## Dataset Structure

The Medical Pills dataset contains 115 images annotated with a single class, `pill`, split into two subsets defined by the `medical-pills.yaml` configuration:

| Split      | Images | Description                                     |
| ---------- | ------ | ----------------------------------------------- |
| Train      | 92     | Labeled images for model training               |
| Validation | 23     | Held-out images for evaluation and benchmarking |

Explore [Medical Pills on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/medical-pills) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

Using computer vision for medical pills detection enables automation in the pharmaceutical industry, supporting tasks like:
>>>>>>> origin/main

- **Pharmaceutical Sorting**: Automating the sorting of pills based on size, shape, or color to enhance production efficiency.
- **AI Research and Development**: Serving as a benchmark for developing and testing computer vision algorithms in pharmaceutical use cases.
- **Digital Inventory Systems**: Powering smart inventory solutions by integrating automated pill recognition for real-time stock monitoring and replenishment planning.
- **Quality Control**: Ensuring consistency in pill production by identifying defects, irregularities, or contamination.
- **Counterfeit Detection**: Helping identify potentially counterfeit medications by analyzing visual characteristics against known standards.

## Dataset YAML

<<<<<<< HEAD
A YAML configuration file is provided to define the dataset's structure, including paths and classes. For the medical-pills dataset, the `medical-pills.yaml` file can be accessed at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml).
=======
The `medical-pills.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/medical-pills.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/medical-pills.yaml"
    ```

## Usage

<<<<<<< HEAD
To train a YOLO11n model on the medical-pills dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For detailed arguments, refer to the model's [Training](../../modes/train.md) page.
=======
To train a YOLO26n model on the Medical Pills dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For detailed arguments, refer to the model's [Training](../../modes/train.md) page.
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
        results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=medical-pills.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=medical-pills.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/medical-pills-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a fine-tuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/medical-pills-sample.jpg"
        ```

## Sample Images and Annotations

<<<<<<< HEAD
The medical-pills dataset features labeled images showcasing the diversity of pills. Below is an example of a labeled image from the dataset:

![Medical-pills dataset sample image](https://github.com/ultralytics/docs/releases/download/0/medical-pills-dataset-sample-image.avif)
=======
The Medical Pills dataset features labeled images showcasing the diversity of pills. Below is an example of a labeled image from the dataset:

![Medical Pills dataset sample image](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/medical-pills-dataset-sample-image.avif)
>>>>>>> origin/main

- **Mosaiced Image**: Displayed is a training batch comprising mosaiced dataset images. Mosaicing enhances training diversity by consolidating multiple images into one, improving model generalization.

## Integration with Other Datasets

<<<<<<< HEAD
For more comprehensive pharmaceutical analysis, consider combining the medical-pills dataset with other related datasets like [package-seg](../segment/package-seg.md) for packaging identification or medical imaging datasets like [brain-tumor](brain-tumor.md) to develop end-to-end healthcare AI solutions.
=======
For more comprehensive pharmaceutical analysis, consider combining the Medical Pills dataset with other related datasets like [package-seg](../segment/package-seg.md) for packaging identification or medical imaging datasets like [brain-tumor](brain-tumor.md) to develop end-to-end healthcare AI solutions.
>>>>>>> origin/main

## Citations and Acknowledgments

The dataset is available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

<<<<<<< HEAD
If you use the Medical-pills dataset in your research or development work, please cite it using the mentioned details:
=======
If you use the Medical Pills dataset in your research or development work, please cite it using the mentioned details:
>>>>>>> origin/main

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2024,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {Dec},
            title = {Ultralytics Datasets: Medical-pills Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/medical-pills/},
            version = {1.0.0},
            year = {2024}
        }
        ```

## FAQ

<<<<<<< HEAD
### What is the structure of the medical-pills dataset?

The dataset includes 92 images for training and 23 images for validation. Each image is annotated with the class `pill`, enabling effective training and evaluation of models for pharmaceutical applications.

### How can I train a YOLO11 model on the medical-pills dataset?

You can train a YOLO11 model for 100 epochs with an image size of 640px using the Python or CLI methods provided. Refer to the [Training Example](#usage) section for detailed instructions and check the [YOLO11 documentation](../../models/yolo11.md) for more information on model capabilities.

### What are the benefits of using the medical-pills dataset in AI projects?

The dataset enables automation in pill detection, contributing to counterfeit prevention, quality assurance, and pharmaceutical process optimization. It also serves as a valuable resource for developing AI solutions that can improve medication safety and supply chain efficiency.

### How do I perform inference on the medical-pills dataset?

Inference can be done using Python or CLI methods with a fine-tuned YOLO11 model. Refer to the [Inference Example](#usage) section for code snippets and the [Predict mode documentation](../../modes/predict.md) for additional options.

### Where can I find the YAML configuration file for the medical-pills dataset?
=======
### How many images and classes are in the Medical Pills dataset?

The Medical Pills dataset contains 115 images total — 92 for training and 23 for validation — with no separate test split. Each image is annotated with a single class, `pill`. It ships as an 8.19 MB automatic download defined in the [`medical-pills.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml) configuration.

### How can I train a YOLO26 model on the Medical Pills dataset?

You can train a YOLO26 model for 100 epochs with an image size of 640px using the Python or CLI methods provided. Refer to the [Training Example](#usage) section for detailed instructions and check the [YOLO26 documentation](../../models/yolo26.md) for more information on model capabilities.

### What are the benefits of using the Medical Pills dataset in AI projects?

The dataset enables automation in pill detection, contributing to counterfeit prevention, quality assurance, and pharmaceutical process optimization. It also serves as a valuable resource for developing AI solutions that can improve medication safety and supply chain efficiency.

### How do I perform inference on the Medical Pills dataset?

Inference can be done using Python or CLI methods with a fine-tuned YOLO26 model. Refer to the [Inference Example](#usage) section for code snippets and the [Predict mode documentation](../../modes/predict.md) for additional options.

### Where can I find the YAML configuration file for the Medical Pills dataset?
>>>>>>> origin/main

The YAML file is available at [medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml), containing dataset paths, classes, and additional configuration details essential for training models on this dataset.

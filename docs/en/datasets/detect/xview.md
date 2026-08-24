---
<<<<<<< HEAD
comments: true
description: Explore the xView dataset, a rich resource of 1M+ object instances in high-resolution satellite imagery. Enhance detection, learning efficiency, and more.
keywords: xView dataset, overhead imagery, satellite images, object detection, high resolution, bounding boxes, computer vision, TensorFlow, PyTorch, dataset structure
=======
title: xView Detection Dataset
comments: true
creator:
    name: Defense Innovation Unit
    url: https://www.diu.mil/
license:
    name: CC-BY-NC-SA-4.0
description: Train YOLO26 on the xView satellite dataset - 1M+ object instances across 60 classes in 0.3 m WorldView-3 imagery with automatic GeoJSON-to-YOLO conversion.
keywords: xView dataset, satellite imagery, overhead imagery, object detection, remote sensing, YOLO26, xView download, WorldView-3, bounding boxes, computer vision
>>>>>>> origin/main
---

# xView Dataset

<<<<<<< HEAD
The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available datasets of overhead imagery, containing images from complex scenes around the world annotated using bounding boxes. The goal of the xView dataset is to accelerate progress in four [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) frontiers:
=======
The [xView](https://xviewdataset.org/) dataset is one of the largest publicly available satellite-imagery benchmarks for [object detection](../../tasks/detect.md), providing over 1 million object instances across 60 classes annotated with [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) in more than 1,400 km² of 0.3 m WorldView-3 imagery. It was released for the DIUx xView 2018 Challenge by the U.S. National Geospatial-Intelligence Agency (NGA) and requires a manual download of about 20.7 GB.

The dataset was created to push four [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) frontiers:
>>>>>>> origin/main

1. Reduce minimum resolution for detection.
2. Improve learning efficiency.
3. Enable discovery of more object classes.
4. Improve detection of fine-grained classes.

<<<<<<< HEAD
xView builds on the success of challenges like [Common Objects in Context (COCO)](../detect/coco.md) and aims to leverage computer vision to analyze the growing amount of available imagery from space in order to understand the visual world in new ways and address a range of important applications.

!!! warning "Manual Download Required"

    The xView dataset is **not** automatically downloaded by Ultralytics scripts. You **must** manually download the dataset first from the official source:

    - **Source:** DIUx xView 2018 Challenge by U.S. National Geospatial-Intelligence Agency (NGA)
    - **URL:** [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)

    **Important:** After downloading the necessary files (e.g., `train_images.tif`, `val_images.tif`, `xView_train.geojson`), you need to extract them and place them into the correct directory structure, typically expected under a `datasets/xView/` folder, **before** running the training commands provided below. Ensure the dataset is properly set up as per the challenge instructions.

## Key Features

- xView contains over 1 million object instances across 60 classes.
- The dataset has a resolution of 0.3 meters, providing higher resolution imagery than most public satellite imagery datasets.
- xView features a diverse collection of small, rare, fine-grained, and multi-type objects with [bounding box](https://www.ultralytics.com/glossary/bounding-box) annotation.
- Comes with a pretrained baseline model using the [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) object detection API and an example for [PyTorch](https://www.ultralytics.com/glossary/pytorch).

## Dataset Structure

The xView dataset is composed of satellite images collected from WorldView-3 satellites at a 0.3m ground sample distance. It contains over 1 million objects across 60 classes in over 1,400 km² of imagery. The dataset is particularly valuable for [remote sensing](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery) applications and environmental monitoring.

## Applications

The xView dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for object detection in overhead imagery. The dataset's diverse set of object classes and high-resolution imagery make it a valuable resource for researchers and practitioners in the field of computer vision, especially for satellite imagery analysis. Applications include:
=======
Building on benchmarks like [COCO](coco.md), xView targets overhead imagery, where objects are far smaller and more densely packed than in ground-level photos.

!!! warning "Manual Download Required"

    The xView dataset is **not** downloaded automatically. Register at the [DIUx xView 2018 Challenge](https://challenge.xviewdataset.org/) website to download `train_images.zip` (~15 GB), `train_labels.zip`, and `val_images.zip` (~5 GB), then extract them under `datasets/xView/` so that it contains:

    ```text
    datasets/xView/
    ├── train_images/          # 847 TIF satellite images
    ├── val_images/            # 282 TIF images (no public labels)
    └── xView_train.geojson    # bounding-box annotations
    ```

    On the first training run, Ultralytics converts the GeoJSON annotations to YOLO format and splits the labeled images roughly 90/10 into training and validation sets automatically — no manual conversion is needed.

## Key Features

- **Fine-grained classes**: 60 object classes spanning aircraft, vehicles, railway stock, maritime vessels, construction equipment, and buildings — many small, rare, and visually similar.
- **High resolution**: 0.3 m ground sample distance collected from WorldView-3 satellites.
- **Dense annotation**: over 1 million object instances across more than 1,400 km² of imagery, all labeled with horizontal bounding boxes.
- **Automatic conversion**: the Ultralytics download script converts the original GeoJSON labels to YOLO format and generates the train/val split on first use.

## Dataset Structure

xView images are large satellite scenes in TIF format, and only the 847 training images ship with public labels — the 282-image challenge validation set has none. The Ultralytics `xView.yaml` configuration therefore splits the labeled images automatically on first use:

| Split      | Images      | Description                                                                             |
| ---------- | ----------- | --------------------------------------------------------------------------------------- |
| Train      | ~90% of 847 | Labeled images listed in `autosplit_train.txt`, generated on the first run              |
| Validation | ~10% of 847 | Labeled images listed in `autosplit_val.txt`, used for [evaluation](../../modes/val.md) |

The 60 classes cover fine-grained categories such as Fixed-wing Aircraft, Cargo Plane, Small Car, Bus, Locomotive, Maritime Vessel, Excavator, Building, Aircraft Hangar, and Storage Tank; the full list is in the [Dataset YAML](#dataset-yaml) below. During conversion, the original challenge class IDs (11–94) are remapped to contiguous indices 0–59.

## Applications

xView's fine-grained classes and high-resolution overhead viewpoint make it a standard benchmark for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in [remote sensing](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery). Common applications include:
>>>>>>> origin/main

- Military and defense reconnaissance
- Urban planning and development
- Environmental monitoring
- Disaster response and assessment
- Infrastructure mapping and management

<<<<<<< HEAD
## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the xView dataset, the `xView.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml).
=======
For other overhead-imagery benchmarks, see the drone-focused [VisDrone dataset](visdrone.md) or the oriented-box [DOTA-v2 dataset](../obb/dota-v2.md).

## Dataset YAML

The `xView.yaml` file defines the dataset configuration — the dataset paths, the 60 class names, and the download script that converts the GeoJSON annotations and generates the autosplit. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## Usage

<<<<<<< HEAD
=======
!!! note "20.7 GB manual download"

    Training expects the manual download described above to be extracted under `datasets/xView/`; annotation conversion and the train/val split then run automatically.

>>>>>>> origin/main
To train a model on the xView dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

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
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=xView.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The xView dataset contains high-resolution satellite images with a diverse set of objects annotated using bounding boxes. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/overhead-imagery-object-detection.avif)

- **Overhead Imagery**: This image demonstrates an example of [object detection](https://www.ultralytics.com/glossary/object-detection) in overhead imagery, where objects are annotated with bounding boxes. The dataset provides high-resolution satellite images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the xView dataset and highlights the importance of high-quality satellite imagery for object detection tasks.

## Related Datasets

If you're working with satellite imagery, you might also be interested in exploring these related datasets:

- [DOTA-v2](../obb/dota-v2.md): A dataset for oriented object detection in aerial images
- [VisDrone](../detect/visdrone.md): A dataset for object detection and tracking in drone-captured imagery
- [Argoverse](../detect/argoverse.md): A dataset for autonomous driving with 3D tracking annotations
=======
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

To label additional satellite images and manage xView training runs in your browser, use [Ultralytics Platform](https://platform.ultralytics.com/).

## Sample Data and Annotations

The sample below shows a typical xView scene: high-resolution overhead imagery in which small objects such as vehicles and buildings are annotated with bounding boxes, illustrating why [object detection](https://www.ultralytics.com/glossary/object-detection) in satellite imagery demands fine-grained localization.

![xView dataset overhead satellite imagery with object detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/overhead-imagery-object-detection.avif)
>>>>>>> origin/main

## Citations and Acknowledgments

If you use the xView dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

<<<<<<< HEAD
We would like to acknowledge the [Defense Innovation Unit](https://www.diu.mil/) (DIU) and the creators of the xView dataset for their valuable contribution to the computer vision research community. For more information about the xView dataset and its creators, visit the [xView dataset website](http://xviewdataset.org/).
=======
We would like to acknowledge the [Defense Innovation Unit](https://www.diu.mil/) (DIU) and the creators of the xView dataset for their valuable contribution to the computer vision research community. For more information, visit the [xView dataset website](https://xviewdataset.org/).
>>>>>>> origin/main

## FAQ

### What is the xView dataset and how does it benefit computer vision research?

<<<<<<< HEAD
The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available collections of high-resolution overhead imagery, containing over 1 million object instances across 60 classes. It is designed to enhance various facets of computer vision research such as reducing the minimum resolution for detection, improving learning efficiency, discovering more object classes, and advancing fine-grained object detection.

### How can I use Ultralytics YOLO to train a model on the xView dataset?

To train a model on the xView dataset using [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/), follow these steps:
=======
The [xView](https://xviewdataset.org/) dataset is a satellite-imagery benchmark released for the DIUx xView 2018 Challenge by the U.S. National Geospatial-Intelligence Agency, providing over 1 million object instances across 60 fine-grained classes in 0.3 m WorldView-3 imagery. It supports research on detecting small, rare, and fine-grained objects in overhead views, which are far harder targets than those in ground-level photos.

### How do I download and set up the xView dataset?

xView requires a manual download: register at the [DIUx xView 2018 Challenge](https://challenge.xviewdataset.org/) website, download `train_images.zip` (~15 GB), `train_labels.zip`, and `val_images.zip` (~5 GB) — about 20.7 GB in total — and extract them under `datasets/xView/` following the layout shown in the warning at the top of this page. On the first training run, Ultralytics automatically converts the GeoJSON annotations to YOLO format and creates the train/validation split.

### How many images and classes does xView have?

xView contains 847 labeled training images and 282 validation images without public labels, all captured by WorldView-3 satellites at 0.3 m resolution. Annotations cover over 1 million object instances across 60 classes. Because only the training labels are public, the Ultralytics `xView.yaml` configuration splits the 847 labeled images roughly 90/10 into training and validation sets; see [Dataset Structure](#dataset-structure) for details.

### How do I train a YOLO26 model on the xView dataset?

Train a [YOLO26n](../../models/yolo26.md) model on xView for 100 epochs at an image size of 640:
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
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

<<<<<<< HEAD

=======
>>>>>>> origin/main
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=xView.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

For detailed arguments and settings, refer to the model [Training](../../modes/train.md) page.

<<<<<<< HEAD
### What are the key features of the xView dataset?

The xView dataset stands out due to its comprehensive set of features:

- Over 1 million object instances across 60 distinct classes.
- High-resolution imagery at 0.3 meters.
- Diverse object types including small, rare, and fine-grained objects, all annotated with bounding boxes.
- Availability of a pretrained baseline model and examples in [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and PyTorch.

### What is the dataset structure of xView, and how is it annotated?

The xView dataset contains high-resolution satellite imagery captured by WorldView-3 satellites at a 0.3m ground sample distance, covering over 1 million objects across 60 distinct classes within approximately 1,400 km² of annotated imagery. Each object is labeled with bounding boxes, making the dataset highly suitable for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for object detection in overhead views. For a detailed breakdown, refer to the [Dataset Structure section](#dataset-structure).

### How do I cite the xView dataset in my research?

If you utilize the xView dataset in your research, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
            title={xView: Objects in Context in Overhead Imagery},
            author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
            year={2018},
            eprint={1802.07856},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
        ```

For more information about the xView dataset, visit the official [xView dataset website](http://xviewdataset.org/).
=======
### How do I cite the xView dataset in my research?

Cite the paper "xView: Objects in Context in Overhead Imagery" (Lam et al., arXiv:1802.07856, 2018); the full BibTeX entry is in the [Citations and Acknowledgments](#citations-and-acknowledgments) section above.
>>>>>>> origin/main

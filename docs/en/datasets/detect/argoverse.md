---
<<<<<<< HEAD
comments: true
description: Explore the comprehensive Argoverse dataset by Argo AI for 3D tracking, motion forecasting, and stereo depth estimation in autonomous driving research.
keywords: Argoverse dataset, autonomous driving, 3D tracking, motion forecasting, stereo depth estimation, Argo AI, LiDAR point clouds, high-resolution images, HD maps
=======
title: Argoverse Detection Dataset
comments: true
creator:
    name: Argo AI
    url: https://www.argoverse.org/av1.html
license:
    name: CC-BY-NC-SA-4.0
description: Train YOLO object detection models on the Argoverse (Argoverse-HD) dataset — 54,446 autonomous-driving images across 8 classes, from a ring-front-center camera.
keywords: Argoverse dataset, Argoverse-HD, object detection, 2D detection, autonomous driving, self-driving car dataset, YOLO26, traffic detection, Ultralytics
>>>>>>> origin/main
---

# Argoverse Dataset

<<<<<<< HEAD
The [Argoverse](https://www.argoverse.org/) dataset is a collection of data designed to support research in autonomous driving tasks, such as 3D tracking, motion forecasting, and stereo depth estimation. Developed by Argo AI, the dataset provides a wide range of high-quality sensor data, including high-resolution images, LiDAR point clouds, and map data.

!!! note

    The Argoverse dataset `*.zip` file required for training was removed from Amazon S3 after the shutdown of Argo AI by Ford, but we have made it available for manual download on [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link).

## Key Features

- Argoverse contains over 290K labeled 3D object tracks and 5 million object instances across 1,263 distinct scenes.
- The dataset includes high-resolution camera images, LiDAR point clouds, and richly annotated HD maps.
- Annotations include 3D bounding boxes for objects, object tracks, and trajectory information.
- Argoverse provides multiple subsets for different tasks, such as 3D tracking, motion forecasting, and stereo depth estimation.

## Dataset Structure

The Argoverse dataset is organized into three main subsets:

1. **Argoverse 3D Tracking**: This subset contains 113 scenes with over 290K labeled 3D object tracks, focusing on 3D object tracking tasks. It includes LiDAR point clouds, camera images, and sensor calibration information.
2. **Argoverse Motion Forecasting**: This subset consists of 324K vehicle trajectories collected from 60 hours of driving data, suitable for motion forecasting tasks.
3. **Argoverse Stereo Depth Estimation**: This subset is designed for stereo depth estimation tasks and includes over 10K stereo image pairs with corresponding LiDAR point clouds for ground truth depth estimation.

## Applications

The Argoverse dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in autonomous driving tasks such as 3D object tracking, motion forecasting, and stereo depth estimation. The dataset's diverse set of sensor data, object annotations, and map information make it a valuable resource for researchers and practitioners in the field of autonomous driving.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Argoverse dataset, the `Argoverse.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).
=======
The Ultralytics Argoverse dataset (Argoverse-HD) is a 2D [object detection](../../tasks/detect.md) dataset of 54,446 labeled autonomous-driving images — 39,384 for training and 15,062 for validation — across 8 classes: person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign. The images are captured from a vehicle's ring-front-center camera, and the annotations come from Carnegie Mellon University's streaming-perception project, built on [Argo AI](https://www.argoverse.org/)'s Argoverse 1.1 driving data. It is a large, real-world benchmark for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models to detect road objects in self-driving scenarios.

!!! note "Manual download required"

    The Argoverse-HD `*.zip` file (~31.5 GB) needed for training was removed from Amazon S3 after the shutdown of Argo AI by Ford. It is available for manual download from [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link) — automatic download will not work, so download the archive before training.

## Key Features

- **8 object-detection classes**: person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.
- **54,446 labeled images** — 39,384 for training and 15,062 for validation — plus an unlabeled test split reserved for the [eval.ai challenge](https://eval.ai/web/challenges/challenge-page/800/overview).
- **~31.5 GB** of high-resolution ring-front-center camera frames captured in urban autonomous-driving scenes.
- Annotations are converted to YOLO format automatically on first use, so the dataset trains directly with Ultralytics YOLO detection models.

## Dataset Structure

The Argoverse-HD dataset is split into three predefined subsets, defined by the `Argoverse.yaml` configuration:

| Split      | Images | Labels                                                                                      |
| ---------- | ------ | ------------------------------------------------------------------------------------------- |
| Train      | 39,384 | Yes                                                                                         |
| Validation | 15,062 | Yes                                                                                         |
| Test       | —      | Unlabeled ([eval.ai challenge](https://eval.ai/web/challenges/challenge-page/800/overview)) |

All images share the same 8 object classes (indices 0–7): person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.

!!! tip "Automatic YOLO conversion"

    After the manual download, Ultralytics converts the original Argoverse-HD annotations into YOLO detection labels automatically the first time you train, so no manual preprocessing is required.

## Applications

The Argoverse-HD dataset supports a range of [object detection](../../tasks/detect.md) applications in autonomous driving:

- **Self-driving perception** — detect vehicles, pedestrians, and cyclists from a forward-facing camera to support [autonomous-vehicle](https://www.ultralytics.com/solutions/ai-in-automotive) navigation.
- **Advanced driver-assistance systems (ADAS)** — recognize traffic lights and stop signs for real-time driver alerts.
- **Traffic monitoring** — count and track road users in urban scenes for smart-city analytics.
- **Research and prototyping** — a large, real-world benchmark for learning [model training](../../modes/train.md) and [prediction](../../modes/predict.md) on driving data.

## Dataset YAML

A YAML file defines the dataset configuration, including paths, classes, and other relevant details. For the Argoverse dataset, the `Argoverse.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).
>>>>>>> origin/main

!!! example "ultralytics/cfg/datasets/Argoverse.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Argoverse.yaml"
    ```

## Usage

<<<<<<< HEAD
To train a YOLO11n model on the Argoverse dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
=======
To train a YOLO26n model on the Argoverse dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following code samples. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
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
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=Argoverse.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=Argoverse.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

Once trained, run [inference](../../modes/predict.md) with the fine-tuned model on new driving images or video:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load an Argoverse fine-tuned model

        # Inference using the model
        results = model.predict("path/to/driving-scene.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="path/to/driving-scene.jpg"
>>>>>>> origin/main
        ```

## Sample Data and Annotations

<<<<<<< HEAD
The Argoverse dataset contains a diverse set of sensor data, including camera images, LiDAR point clouds, and HD map information, providing rich context for autonomous driving tasks. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/argoverse-3d-tracking-sample.avif)

- **Argoverse 3D Tracking**: This image demonstrates an example of 3D object tracking, where objects are annotated with 3D bounding boxes. The dataset provides LiDAR point clouds and camera images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Argoverse dataset and highlights the importance of high-quality sensor data for autonomous driving tasks.

## Citations and Acknowledgments

If you use the Argoverse dataset in your research or development work, please cite the following paper:
=======
The Argoverse-HD dataset contains high-resolution driving images captured from a ring-front-center camera, annotated with 2D bounding boxes for the 8 object classes. Below is an example image from the dataset with its corresponding annotations:

![Argoverse-HD autonomous driving scene with annotated road objects](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/argoverse-3d-tracking-sample.avif)

- **Annotated driving scene**: This image shows road objects — such as vehicles and pedestrians — labeled with 2D bounding boxes, the format YOLO models learn to predict during training.

## Citations and Acknowledgments

The Argoverse-HD 2D detection annotations used in this dataset come from Carnegie Mellon University's streaming-perception work. If you use the dataset in your research or development, please cite:
>>>>>>> origin/main

!!! quote ""

    === "BibTeX"

        ```bibtex
<<<<<<< HEAD
=======
        @inproceedings{li2020towards,
          title={Towards Streaming Perception},
          author={Li, Mengtian and Wang, Yu-Xiong and Ramanan, Deva},
          booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
          pages={473--488},
          year={2020}
        }

>>>>>>> origin/main
        @inproceedings{chang2019argoverse,
          title={Argoverse: 3D Tracking and Forecasting with Rich Maps},
          author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, Dequan and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={8748--8757},
          year={2019}
        }
        ```

<<<<<<< HEAD
We would like to acknowledge Argo AI for creating and maintaining the Argoverse dataset as a valuable resource for the autonomous driving research community. For more information about the Argoverse dataset and its creators, visit the [Argoverse dataset website](https://www.argoverse.org/).

## FAQ

### What is the Argoverse dataset and its key features?

The [Argoverse](https://www.argoverse.org/) dataset, developed by Argo AI, supports autonomous driving research. It includes over 290K labeled 3D object tracks and 5 million object instances across 1,263 distinct scenes. The dataset provides high-resolution camera images, LiDAR point clouds, and annotated HD maps, making it valuable for tasks like 3D tracking, motion forecasting, and stereo depth estimation.

### How can I train an Ultralytics YOLO model using the Argoverse dataset?

To train a YOLO11 model with the Argoverse dataset, use the provided YAML configuration file and the following code:

!!! example "Train Example"
=======
We would like to acknowledge Carnegie Mellon University for the Argoverse-HD detection annotations and [Argo AI](https://www.argoverse.org/) for creating the original Argoverse dataset as a valuable resource for the autonomous-driving research community.

## FAQ

### What is the Argoverse dataset, and what is it used for?

The Ultralytics Argoverse dataset (Argoverse-HD) is a 2D [object detection](../../tasks/detect.md) dataset of 54,446 autonomous-driving images across 8 classes — person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign. It is used to train and evaluate models that detect road objects from a forward-facing vehicle camera, supporting self-driving perception, ADAS, and traffic-monitoring research.

### How many classes and images are in the Argoverse dataset?

The Argoverse-HD dataset has **8 classes** (person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign) and **54,446 labeled images** — 39,384 for training and 15,062 for validation — plus an unlabeled test split reserved for the eval.ai challenge.

### Is the Argoverse dataset 2D or 3D detection in Ultralytics?

In Ultralytics it is a **2D object detection** dataset (Argoverse-HD camera frames with 2D bounding boxes), not the 3D-tracking, motion-forecasting, or LiDAR research suite from the broader Argoverse program. You train it with a standard detection model such as `yolo26n.pt`.

### How do I train a YOLO26 model using the Argoverse dataset?

Download the dataset manually first (see below), then train with the `Argoverse.yaml` configuration file:

!!! example
>>>>>>> origin/main

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
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```

<<<<<<< HEAD

=======
>>>>>>> origin/main
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo detect train data=Argoverse.yaml model=yolo11n.pt epochs=100 imgsz=640
=======
        yolo detect train data=Argoverse.yaml model=yolo26n.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

For a detailed explanation of the arguments, refer to the model [Training](../../modes/train.md) page.

<<<<<<< HEAD
### What types of data and annotations are available in the Argoverse dataset?

The Argoverse dataset includes various sensor data types such as high-resolution camera images, LiDAR point clouds, and HD map data. Annotations include 3D bounding boxes, object tracks, and trajectory information. These comprehensive annotations are essential for accurate model training in tasks like 3D object tracking, motion forecasting, and stereo depth estimation.

### How is the Argoverse dataset structured?

The dataset is divided into three main subsets:

1. **Argoverse 3D Tracking**: Contains 113 scenes with over 290K labeled 3D object tracks, focusing on 3D object tracking tasks. It includes LiDAR point clouds, camera images, and sensor calibration information.
2. **Argoverse Motion Forecasting**: Consists of 324K vehicle trajectories collected from 60 hours of driving data, suitable for motion forecasting tasks.
3. **Argoverse Stereo Depth Estimation**: Includes over 10K stereo image pairs with corresponding LiDAR point clouds for ground truth depth estimation.

### Where can I download the Argoverse dataset now that it has been removed from Amazon S3?

The Argoverse dataset `*.zip` file, previously available on Amazon S3, can now be manually downloaded from [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link).

### What is the YAML configuration file used for with the Argoverse dataset?

A YAML file contains the dataset's paths, classes, and other essential information. For the Argoverse dataset, the configuration file, `Argoverse.yaml`, can be found at the following link: [Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).

For more information about YAML configurations, see our [datasets](../index.md) guide.
=======
### Where can I download the Argoverse dataset now that it has been removed from Amazon S3?

The Argoverse-HD `*.zip` file (~31.5 GB), previously hosted on Amazon S3, can now be downloaded manually from [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link). Automatic download will not work, so fetch the archive before running your training command.

### Can I use the Argoverse dataset with Ultralytics Platform?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you upload and version large datasets like Argoverse-HD, then train and deploy [object detection](../../tasks/detect.md) models in the cloud without heavy local setup. You can also browse related datasets in the [detection datasets overview](index.md).
>>>>>>> origin/main

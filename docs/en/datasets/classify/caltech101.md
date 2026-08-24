---
<<<<<<< HEAD
comments: true
description: Explore the widely-used Caltech-101 dataset with 9,000 images across 101 categories. Ideal for object recognition tasks in machine learning and computer vision.
keywords: Caltech-101, dataset, object recognition, machine learning, computer vision, YOLO, deep learning, research, AI
=======
title: Caltech-101 Image Classification Dataset
comments: true
creator:
    name: California Institute of Technology
    url: https://data.caltech.edu/records/mzrjq-6wc02
license:
    name: CC-BY-4.0
description: Train YOLO image classification models on Caltech-101, a benchmark of 9,144 images across 101 object categories plus a background class, with automatic 80/20 splitting.
keywords: Caltech-101, dataset, image classification, object recognition, machine learning, computer vision, YOLO, deep learning, AI
>>>>>>> origin/main
---

# Caltech-101 Dataset

<<<<<<< HEAD
The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is a widely used dataset for object recognition tasks, containing around 9,000 images from 101 object categories. The categories were chosen to reflect a variety of real-world objects, and the images themselves were carefully selected and annotated to provide a challenging benchmark for object recognition algorithms.
=======
The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is a classic [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 9,144 images spanning 101 object categories plus one background class. Each category holds about 40 to 800 images of real-world objects — animals, vehicles, household items, and people — making it a compact yet challenging benchmark for object recognition models.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Train <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model using Caltech-256 Dataset with Ultralytics HUB
=======
  <strong>Watch:</strong> How to Train <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model using Caltech-101 Dataset with Ultralytics Platform
>>>>>>> origin/main
</p>

!!! note "Automatic Data Splitting"

<<<<<<< HEAD
    The Caltech-101 dataset, as provided, does not come with pre-defined train/validation splits. However, when you use the training commands provided in the usage examples below, the Ultralytics framework will automatically split the dataset for you. The default split used is 80% for the training set and 20% for the validation set.

## Key Features

- The Caltech-101 dataset comprises around 9,000 color images divided into 101 categories.
- The categories encompass a wide variety of objects, including animals, vehicles, household items, and people.
- The number of images per category varies, with about 40 to 800 images in each category.
- Images are of variable sizes, with most images being medium resolution.
- Caltech-101 is widely used for training and testing in the field of machine learning, particularly for object recognition tasks.

## Dataset Structure

Unlike many other datasets, the Caltech-101 dataset is not formally split into training and testing sets. Users typically create their own splits based on their specific needs. However, a common practice is to use a random subset of images for training (e.g., 30 images per category) and the remaining images for testing.

## Applications

The Caltech-101 dataset is extensively used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object recognition tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs), and various other machine learning algorithms. Its wide variety of categories and high-quality images make it an excellent dataset for research and development in the field of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Usage

To train a YOLO model on the Caltech-101 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch), you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
=======
    Caltech-101 ships without a predefined train/validation split. The training commands below automatically split it 80% train / 20% validation, so no manual preparation is needed.

## Key Features

- Caltech-101 contains 9,144 color images across 101 object categories plus one `BACKGROUND_Google` class (102 class folders in total).
- The categories span a wide variety of real-world objects, including animals, vehicles, household items, and people.
- Each category holds about 40 to 800 images, so class sizes are imbalanced.
- Images are of variable sizes, most roughly 300x200 pixels (medium resolution).
- Caltech-101 is widely used to benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition algorithms.

## Dataset Structure

Caltech-101 is distributed as 102 folders — one per class, covering 101 object categories plus a `BACKGROUND_Google` class — with no predefined train/validation split. When you launch training, Ultralytics automatically partitions the images so models train across all 102 classes without any manual setup:

- **Classes**: 102 (101 object categories + 1 background)
- **Total images**: 9,144
- **Train/validation split**: automatic 80% / 20% (≈7,280 train, ≈1,864 validation)
- **Images per class**: about 40 to 800 (imbalanced)

## Applications

Caltech-101 is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models, including [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs). Its broad category coverage and clean, labeled images make it a popular benchmark for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research and prototyping.

## Usage

Train a YOLO model on Caltech-101 for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 416. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo classify train data=caltech101 model=yolo11n-cls.pt epochs=100 imgsz=416
=======
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
>>>>>>> origin/main
        ```

## Sample Images and Annotations

The Caltech-101 dataset contains high-quality color images of various objects, providing a well-structured dataset for [image classification](https://www.ultralytics.com/glossary/image-classification) tasks. Here are some examples of images from the dataset:

<<<<<<< HEAD
![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/caltech101-sample-image.avif)

The example showcases the variety and complexity of the objects in the Caltech-101 dataset, emphasizing the significance of a diverse dataset for training robust object recognition models.
=======
![Caltech-101 image classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/caltech101-sample-image.avif)

The samples show the variety of categories and the natural, centered framing typical of Caltech-101, which makes it a clean starting point for training robust object recognition models.
>>>>>>> origin/main

## Citations and Acknowledgments

If you use the Caltech-101 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

We would like to acknowledge Li Fei-Fei, Rob Fergus, and Pietro Perona for creating and maintaining the Caltech-101 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the Caltech-101 dataset and its creators, visit the [Caltech-101 dataset website](https://data.caltech.edu/records/mzrjq-6wc02).

## FAQ

### What is the Caltech-101 dataset used for in machine learning?

<<<<<<< HEAD
The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is widely used in machine learning for object recognition tasks. It contains around 9,000 images across 101 categories, providing a challenging benchmark for evaluating object recognition algorithms. Researchers leverage it to train and test models, especially Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs), in computer vision.

### How can I train an Ultralytics YOLO model on the Caltech-101 dataset?

To train an Ultralytics YOLO model on the Caltech-101 dataset, you can use the provided code snippets. For example, to train for 100 epochs:
=======
The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is widely used to train and benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models. It contains 9,144 images across 101 object categories plus a background class, providing a challenging benchmark for evaluating algorithms such as Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs).

### How can I train an Ultralytics YOLO model on the Caltech-101 dataset?

To train an Ultralytics YOLO model on Caltech-101, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo classify train data=caltech101 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

For more detailed arguments and options, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the Caltech-101 dataset?

The Caltech-101 dataset includes:

- Around 9,000 color images across 101 categories.
- Categories covering a diverse range of objects, including animals, vehicles, and household items.
- Variable number of images per category, typically between 40 and 800.
- Variable image sizes, with most being medium resolution.

These features make it an excellent choice for training and evaluating object recognition models in machine learning and computer vision.

### Why should I cite the Caltech-101 dataset in my research?

Citing the Caltech-101 dataset in your research acknowledges the creators' contributions and provides a reference for others who might use the dataset. The recommended citation is:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

Citing helps in maintaining the integrity of academic work and assists peers in locating the original resource.

### Can I use Ultralytics HUB for training models on the Caltech-101 dataset?

Yes, you can use [Ultralytics HUB](https://www.ultralytics.com/hub) for training models on the Caltech-101 dataset. Ultralytics HUB provides an intuitive platform for managing datasets, training models, and deploying them without extensive coding. For a detailed guide, refer to the [how to train your custom models with Ultralytics HUB](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub) blog post.
=======
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

### How many classes does the Caltech-101 dataset have?

Caltech-101 contains 101 object categories plus one `BACKGROUND_Google` class, for 102 class folders and 9,144 images in total. When you train with Ultralytics, the model learns all 102 classes. Category sizes are imbalanced, ranging from about 40 to 800 images each.

### How is the Caltech-101 dataset split into training and validation sets?

Caltech-101 has no predefined split. The first time you train, Ultralytics automatically divides it 80% training / 20% validation — about 7,280 training and 1,864 validation images — so you do not need to create splits manually. To control the split yourself, organize the images into `train/` and `val/` folders before training.

### Can I use Ultralytics Platform for training models on the Caltech-101 dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run Caltech-101 experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
>>>>>>> origin/main

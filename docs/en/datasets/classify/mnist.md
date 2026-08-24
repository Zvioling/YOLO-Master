---
<<<<<<< HEAD
comments: true
description: Explore the MNIST dataset, a cornerstone in machine learning for handwritten digit recognition. Learn about its structure, features, and applications.
=======
title: MNIST Image Classification Dataset
comments: true
creator:
    name: Yann LeCun
    type: Person
    url: https://engineering.nyu.edu/faculty/yann-lecun
license:
    name: None
description: Train YOLO image classification models on MNIST, a benchmark of 70,000 28x28 grayscale handwritten digit images in 10 classes, split 60k/10k.
>>>>>>> origin/main
keywords: MNIST, dataset, handwritten digits, image classification, deep learning, machine learning, training set, testing set, NIST
---

# MNIST Dataset

<<<<<<< HEAD
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) (Modified National Institute of Standards and Technology) dataset is a large database of handwritten digits that is commonly used for training various image processing systems and machine learning models. It was created by "re-mixing" the samples from NIST's original datasets and has become a benchmark for evaluating the performance of [image classification](https://www.ultralytics.com/glossary/image-classification) algorithms.

## Key Features

- MNIST contains 60,000 training images and 10,000 testing images of handwritten digits.
- The dataset comprises grayscale images of size 28×28 pixels.
- The images are normalized to fit into a 28×28 pixel [bounding box](https://www.ultralytics.com/glossary/bounding-box) and anti-aliased, introducing grayscale levels.
- MNIST is widely used for training and testing in the field of machine learning, especially for image classification tasks.

## Dataset Structure

The MNIST dataset is split into two subsets:

1. **Training Set**: This subset contains 60,000 images of handwritten digits used for training machine learning models.
2. **Testing Set**: This subset consists of 10,000 images used for testing and benchmarking the trained models.

## Dataset Access

- **Original files**: Download the gzip archives from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) if you want direct control over preprocessing.
- **Ultralytics loader**: Use `data="mnist"` (or `data="mnist160"` for the subset below) in your command and the dataset will be downloaded, converted to PNG, and cached automatically.

Each image in the dataset is labeled with the corresponding digit (0-9), making it a supervised learning dataset ideal for classification tasks.

## Extended MNIST (EMNIST)

Extended MNIST (EMNIST) is a newer dataset developed and released by NIST to be the successor to MNIST. While MNIST included images only of handwritten digits, EMNIST includes all the images from NIST Special Database 19, which is a large database of handwritten uppercase and lowercase letters as well as digits. The images in EMNIST were converted into the same 28×28 pixel format, by the same process, as were the MNIST images. Accordingly, tools that work with the older, smaller MNIST dataset will likely work unmodified with EMNIST.

## Applications

The MNIST dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in image classification tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs), and various other machine learning algorithms. The dataset's simple and well-structured format makes it an essential resource for researchers and practitioners in the field of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).
=======
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) (Modified National Institute of Standards and Technology) dataset is an [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 70,000 28x28 grayscale images of handwritten digits spanning 10 classes — the digits 0 through 9. It ships with a predefined split of 60,000 training and 10,000 test images and has long served as the standard benchmark for evaluating [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) algorithms. For the harder clothing-image equivalent, see the related [Fashion-MNIST](fashion-mnist.md) dataset; for color images, see [CIFAR-10](cifar10.md).

## Key Features

- MNIST contains 60,000 training images and 10,000 test images of handwritten digits, for 70,000 in total.
- Every image is a 28x28 grayscale picture of a single digit, normalized and anti-aliased into a fixed 28x28 [bounding box](https://www.ultralytics.com/glossary/bounding-box).
- The 10 classes span the digits 0–9, with a roughly balanced number of images per class.
- It ships with a predefined train/test split, so no manual or automatic splitting is required.
- MNIST is a standard benchmark for [image classification](https://www.ultralytics.com/glossary/image-classification) and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) research.

## Dataset Structure

MNIST ships with an official, predefined split, so no automatic or manual partitioning is needed:

- **Classes**: 10 (handwritten digits 0–9)
- **Total images**: 70,000 (28x28 grayscale)
- **Training set**: 60,000 images
- **Test set**: 10,000 images

!!! note "Validation split"

    MNIST has no separate validation folder, so Ultralytics uses the 10,000-image test set as the validation split during training by default.

Each image is labeled with its corresponding digit (0–9), making MNIST a supervised dataset ideal for classification tasks.

## Applications

MNIST is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) models, from classic [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs) to modern deep architectures. Its small grayscale images and 10 digit classes make it a fast, reproducible benchmark for algorithm comparison and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) experimentation.
>>>>>>> origin/main

Some common applications include:

- Benchmarking new classification algorithms
- Educational purposes for teaching machine learning concepts
- Prototyping image recognition systems
- Testing model optimization techniques

## Usage

<<<<<<< HEAD
To train a CNN model on the MNIST dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 28×28, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
=======
Train a YOLO classification model on MNIST for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 28. The dataset downloads and caches automatically on first use; if you prefer full control over preprocessing, the original gzip archives are also available from the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database). For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.
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
        results = model.train(data="mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo classify train data=mnist model=yolo11n-cls.pt epochs=100 imgsz=28
=======
        yolo classify train data=mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

!!! tip "Quick tests with MNIST160"

    Ultralytics also exposes `data="mnist160"`, a 160-image slice containing the first eight images of each digit (0–9) from both the train and test splits. It mirrors the MNIST directory structure, so you can swap datasets without changing any other arguments — ideal for CI pipelines or sanity checks before committing to the full 70,000-image dataset.

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=5 imgsz=28
>>>>>>> origin/main
        ```

## Sample Images and Annotations

<<<<<<< HEAD
The MNIST dataset contains grayscale images of handwritten digits, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

The example showcases the variety and complexity of the handwritten digits in the MNIST dataset, highlighting the importance of a diverse dataset for training robust image classification models.
=======
Sample images from the MNIST dataset:

![MNIST handwritten digit classification dataset samples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

The samples show the range of handwriting styles the dataset captures across the 10 digit classes.
>>>>>>> origin/main

## Citations and Acknowledgments

If you use the MNIST dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{lecun2010mnist,
                 title={MNIST handwritten digit database},
                 author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
<<<<<<< HEAD
                 journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
=======
                 journal={ATT Labs [Online]},
>>>>>>> origin/main
                 volume={2},
                 year={2010}
        }
        ```

<<<<<<< HEAD
We would like to acknowledge Yann LeCun, Corinna Cortes, and Christopher J.C. Burges for creating and maintaining the MNIST dataset as a valuable resource for the machine learning and computer vision research community. For more information about the MNIST dataset and its creators, visit the [MNIST dataset website](https://en.wikipedia.org/wiki/MNIST_database).

## MNIST160 Quick Tests

Need a lightning-fast regression test? Ultralytics also exposes `data="mnist160"`, a 160-image slice containing the first eight samples from each digit class. It mirrors the MNIST directory structure, so you can swap datasets without changing any other arguments:

!!! example "Train Example with MNIST160"

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo11n-cls.pt epochs=5 imgsz=28
        ```

Use this subset for CI pipelines or sanity checks before committing to the full 70,000-image dataset.
=======
We would like to acknowledge Yann LeCun, Corinna Cortes, and Christopher J.C. Burges for creating and maintaining the MNIST dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about the MNIST dataset and its creators, visit the [MNIST dataset website](https://en.wikipedia.org/wiki/MNIST_database).
>>>>>>> origin/main

## FAQ

### What is the MNIST dataset, and why is it important in machine learning?

<<<<<<< HEAD
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, or Modified National Institute of Standards and Technology dataset, is a widely-used collection of handwritten digits designed for training and testing image classification systems. It includes 60,000 training images and 10,000 testing images, all of which are grayscale and 28×28 pixels in size. The dataset's importance lies in its role as a standard benchmark for evaluating image classification algorithms, helping researchers and engineers to compare methods and track progress in the field.

### How can I use Ultralytics YOLO to train a model on the MNIST dataset?

To train a model on the MNIST dataset using Ultralytics YOLO, you can follow these steps:
=======
The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset is a benchmark of 70,000 28x28 grayscale images of handwritten digits, split into 60,000 training and 10,000 test images across the 10 classes 0–9. It is the standard reference for evaluating [image classification](https://www.ultralytics.com/glossary/image-classification) algorithms — its small, uniform format lets researchers and engineers compare methods and track progress with minimal setup, which is why it remains a common first benchmark in machine learning.

### How many classes and images does the MNIST dataset have?

MNIST has 10 classes — the handwritten digits 0 through 9 — and 70,000 grayscale images in total, each 28x28 pixels. It ships with a predefined split of 60,000 training and 10,000 test images, with a roughly even number of examples per digit.

### How can I use Ultralytics YOLO to train a model on the MNIST dataset?

To train an Ultralytics YOLO model on MNIST, use the code snippets below. The dataset downloads automatically on first use. For a detailed list of available training arguments, refer to the [Training](../../modes/train.md) page.
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
        results = model.train(data="mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo classify train data=mnist model=yolo11n-cls.pt epochs=100 imgsz=28
        ```

For a detailed list of available training arguments, refer to the [Training](../../modes/train.md) page.

### What is the difference between the MNIST and EMNIST datasets?

The MNIST dataset contains only handwritten digits, whereas the Extended MNIST (EMNIST) dataset includes both digits and uppercase and lowercase letters. EMNIST was developed as a successor to MNIST and utilizes the same 28×28 pixel format for the images, making it compatible with tools and models designed for the original MNIST dataset. This broader range of characters in EMNIST makes it useful for a wider variety of machine learning applications.

### Can I use Ultralytics HUB to train models on custom datasets like MNIST?

Yes, you can use [Ultralytics HUB](https://docs.ultralytics.com/hub/) to train models on custom datasets like MNIST. Ultralytics HUB offers a user-friendly interface for uploading datasets, training models, and managing projects without needing extensive coding knowledge. For more details on how to get started, check out the [Ultralytics HUB Quickstart](https://docs.ultralytics.com/hub/quickstart/) page.

### How does MNIST compare to other image classification datasets?

MNIST is simpler than many modern datasets like [CIFAR-10](../classify/cifar10.md) or [ImageNet](../classify/imagenet.md), making it ideal for beginners and quick experimentation. While more complex datasets offer greater challenges with color images and diverse object categories, MNIST remains valuable for its simplicity, small file size, and historical significance in the development of machine learning algorithms. For more advanced classification tasks, consider using [Fashion-MNIST](../classify/fashion-mnist.md), which maintains the same structure but features clothing items instead of digits.
=======
        yolo classify train data=mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

### How is the MNIST dataset split into training and test sets?

MNIST ships with a predefined split of 60,000 training images and 10,000 test images. Unlike folder-based classification datasets that Ultralytics splits automatically, MNIST's official partition is used as-is, and the test set serves as the validation split during training by default.

### What is the difference between the MNIST and EMNIST datasets?

The MNIST dataset contains only handwritten digits, whereas the Extended MNIST (EMNIST) dataset includes both digits and uppercase and lowercase letters. EMNIST was developed as a successor to MNIST and uses the same 28x28 pixel format, making it compatible with tools and models designed for the original MNIST dataset. This broader range of characters makes EMNIST useful for a wider variety of machine learning applications.

### Can I use Ultralytics Platform to train models on datasets like MNIST?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you upload datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run MNIST experiments in the cloud — see the [classification datasets overview](index.md) for related options.

### How does MNIST compare to other image classification datasets?

MNIST is simpler than many modern datasets like [CIFAR-10](cifar10.md) or [ImageNet](imagenet.md), making it ideal for beginners and quick experimentation. While more complex datasets offer greater challenges with color images and diverse object categories, MNIST remains valuable for its simplicity, small file size, and historical significance in the development of machine learning algorithms. For a harder drop-in replacement with the same structure, see [Fashion-MNIST](fashion-mnist.md), which features clothing items instead of digits.
>>>>>>> origin/main

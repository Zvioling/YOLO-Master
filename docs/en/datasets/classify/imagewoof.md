---
<<<<<<< HEAD
comments: true
description: Explore the ImageWoof dataset, a challenging subset of ImageNet focusing on 10 dog breeds, designed to enhance image classification models. Learn more on Ultralytics Docs.
keywords: ImageWoof dataset, ImageNet subset, dog breeds, image classification, deep learning, machine learning, Ultralytics, training dataset, noisy labels
=======
title: ImageWoof Image Classification Dataset
comments: true
creator:
    name: fast.ai
    url: https://github.com/fastai/imagenette
license:
    name: Research-Only
    url: https://www.image-net.org/download.php
description: ImageWoof is a challenging 12,954-image subset of ImageNet with 10 dog breeds, built for fine-grained image classification training and benchmarking.
keywords: ImageWoof dataset, ImageNet subset, dog breeds, image classification, fine-grained classification, deep learning, machine learning, Ultralytics, noisy labels
>>>>>>> origin/main
---

# ImageWoof Dataset

<<<<<<< HEAD
The [ImageWoof](https://github.com/fastai/imagenette) dataset is a subset of the [ImageNet](imagenet.md) consisting of 10 classes that are challenging to classify, since they're all dog breeds. It was created as a more difficult task for [image classification](https://www.ultralytics.com/glossary/image-classification) algorithms to solve, aiming at encouraging development of more advanced models.

## Key Features

- ImageWoof contains images of 10 different dog breeds: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, and Old English sheepdog.
- The dataset provides images at various resolutions (full size, 320px, 160px), accommodating for different computational capabilities and research needs.
- It also includes a version with noisy labels, providing a more realistic scenario where labels might not always be reliable.

## Dataset Structure

The ImageWoof dataset structure is based on the dog breed classes, with each breed having its own directory of images. Similar to other classification datasets, it follows a split-directory format with separate folders for training and validation sets.

## Applications

The ImageWoof dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in image classification tasks, especially when it comes to more complex and similar classes. The dataset's challenge lies in the subtle differences between the dog breeds, pushing the limits of model's performance and generalization. It's particularly valuable for:

- Benchmarking classification model performance on fine-grained categories
- Testing model robustness against similar-looking classes
- Developing algorithms that can distinguish subtle visual differences
- Evaluating transfer learning capabilities from general to specific domains

## Usage

To train a CNN model on the ImageWoof dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
=======
The [ImageWoof](https://github.com/fastai/imagenette) dataset is a subset of [ImageNet](imagenet.md) consisting of 10 dog-breed classes that are deliberately hard to tell apart, created by [fast.ai](https://www.fast.ai/) as a tougher challenge for [image classification](https://www.ultralytics.com/glossary/image-classification) algorithms. It contains 12,954 color images — 9,025 for training and 3,929 for validation — across breeds such as Beagle, Shih-Tzu, and Golden retriever, pushing models to distinguish subtle fine-grained differences rather than obvious object categories.

## Key Features

- ImageWoof contains 12,954 images across 10 dog breeds: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, and Old English sheepdog.
- It ships with a predefined split of 9,025 training and 3,929 validation images, available at various resolutions (full size, 320px, 160px) to suit different computational budgets.
- It also includes a version with noisy labels, providing a more realistic scenario where labels are not always reliable.

## Dataset Structure

ImageWoof ships with a predefined train/validation split, with each dog breed stored in its own folder:

| Split      | Images | Classes |
| ---------- | ------ | ------- |
| Train      | 9,025  | 10      |
| Validation | 3,929  | 10      |

Because all 10 classes are dog breeds, the split is designed to test fine-grained classification — telling apart visually similar categories — rather than the broad object recognition of the full ImageNet dataset.

## Applications

The ImageWoof dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models on more complex and similar classes. Its challenge lies in the subtle differences between the dog breeds, pushing the limits of model performance and generalization. It is particularly valuable for:

- Benchmarking classification performance on fine-grained categories
- Testing model robustness against similar-looking classes
- Developing algorithms that can distinguish subtle visual differences
- Evaluating transfer learning from general to specific domains

## Usage

To train a classification model on the ImageWoof dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 224x224, use the code snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.
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
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
<<<<<<< HEAD
        yolo classify train data=imagewoof model=yolo11n-cls.pt epochs=100 imgsz=224
=======
        yolo classify train data=imagewoof model=yolo26n-cls.pt epochs=100 imgsz=224
>>>>>>> origin/main
        ```

## Dataset Variants

<<<<<<< HEAD
ImageWoof dataset comes in three different sizes to accommodate various research needs and computational capabilities:

1. **Full Size (imagewoof)**: This is the original version of the ImageWoof dataset. It contains full-sized images and is ideal for final training and performance benchmarking.

2. **Medium Size (imagewoof320)**: This version contains images resized to have a maximum edge length of 320 pixels. It's suitable for faster training without significantly sacrificing model performance.

3. **Small Size (imagewoof160)**: This version contains images resized to have a maximum edge length of 160 pixels. It's designed for rapid prototyping and experimentation where training speed is a priority.

To use these variants in your training, simply replace 'imagewoof' in the dataset argument with 'imagewoof320' or 'imagewoof160'. For example:
=======
ImageWoof comes in three sizes to accommodate different research needs and computational budgets:

1. **Full Size (`imagewoof`)**: The original version with full-sized images, ideal for final training and performance benchmarking.
2. **Medium Size (`imagewoof320`)**: Images resized to a maximum edge length of 320 pixels, suitable for faster training without significantly sacrificing model performance.
3. **Small Size (`imagewoof160`)**: Images resized to a maximum edge length of 160 pixels, designed for rapid prototyping and experimentation where training speed is a priority.

To use these variants, simply replace `imagewoof` in the dataset argument with `imagewoof320` or `imagewoof160`. For example:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
=======
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)
>>>>>>> origin/main

        # For medium-sized dataset
        model.train(data="imagewoof320", epochs=100, imgsz=224)

        # For small-sized dataset
        model.train(data="imagewoof160", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Load a pretrained model and train on the medium-sized dataset
<<<<<<< HEAD
        yolo classify train model=yolo11n-cls.pt data=imagewoof320 epochs=100 imgsz=224
        ```

It's important to note that using smaller images will likely yield lower performance in terms of classification accuracy. However, it's an excellent way to iterate quickly in the early stages of model development and prototyping.
=======
        yolo classify train model=yolo26n-cls.pt data=imagewoof320 epochs=100 imgsz=224
        ```

Note that smaller images will likely yield lower classification [accuracy](https://www.ultralytics.com/glossary/accuracy), but they are an excellent way to iterate quickly in the early stages of model development. You can also manage classification datasets and run training in the cloud with [Ultralytics Platform](https://platform.ultralytics.com/).
>>>>>>> origin/main

## Sample Images and Annotations

The ImageWoof dataset contains colorful images of various dog breeds, providing a challenging dataset for image classification tasks. Here are some examples of images from the dataset:

<<<<<<< HEAD
![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/imagewoof-dataset-sample.avif)

The example showcases the subtle differences and similarities among the different dog breeds in the ImageWoof dataset, highlighting the complexity and difficulty of the classification task.

## Citations and Acknowledgments

If you use the ImageWoof dataset in your research or development work, please make sure to acknowledge the creators of the dataset by linking to the [official dataset repository](https://github.com/fastai/imagenette).

We would like to acknowledge the [FastAI](https://www.fast.ai/) team for creating and maintaining the ImageWoof dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about the ImageWoof dataset, visit the [ImageWoof dataset repository](https://github.com/fastai/imagenette).
=======
![ImageWoof dog breed classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagewoof-dataset-sample.avif)

The example showcases the subtle differences and similarities among the different dog breeds, highlighting the complexity and difficulty of the classification task.

## Citations and Acknowledgments

If you use the ImageWoof dataset in your research or development work, please acknowledge the creators of the dataset by linking to the [official dataset repository](https://github.com/fastai/imagenette).

We would like to acknowledge the [fast.ai](https://www.fast.ai/) team for creating and maintaining ImageWoof as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about ImageWoof, visit the [ImageWoof dataset repository](https://github.com/fastai/imagenette).
>>>>>>> origin/main

## FAQ

### What is the ImageWoof dataset in Ultralytics?

<<<<<<< HEAD
The [ImageWoof](https://github.com/fastai/imagenette) dataset is a challenging subset of ImageNet focusing on 10 specific dog breeds. Created to push the limits of image classification models, it features breeds like Beagle, Shih-Tzu, and Golden Retriever. The dataset includes images at various resolutions (full size, 320px, 160px) and even noisy labels for more realistic training scenarios. This complexity makes ImageWoof ideal for developing more advanced deep learning models.

### How can I train a model using the ImageWoof dataset with Ultralytics YOLO?

To train a [Convolutional Neural Network](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNN) model on the ImageWoof dataset using Ultralytics YOLO for 100 epochs at an image size of 224x224, you can use the following code:
=======
The [ImageWoof](https://github.com/fastai/imagenette) dataset is a challenging subset of [ImageNet](imagenet.md) focused on 10 dog breeds, containing 12,954 images (9,025 training and 3,929 validation). Created by fast.ai to push the limits of image classification models, it features breeds such as Beagle, Shih-Tzu, and Golden retriever. The dataset is available at various resolutions (full size, 320px, 160px) and even includes noisy labels for more realistic training scenarios, making it ideal for developing advanced deep learning models.

### How many images and dog breeds does ImageWoof have?

ImageWoof contains 12,954 images in total — 9,025 for training and 3,929 for validation — across 10 dog breeds: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, and Old English sheepdog. Each breed is stored in its own folder, following the standard classification layout Ultralytics expects.

### How can I train a model using the ImageWoof dataset with Ultralytics YOLO?

To train a classification model on the ImageWoof dataset using Ultralytics YOLO for 100 epochs at an image size of 224x224, use the following code:
>>>>>>> origin/main

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        model = YOLO("yolo11n-cls.pt")  # Load a pretrained model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```


    === "CLI"

        ```bash
        yolo classify train data=imagewoof model=yolo11n-cls.pt epochs=100 imgsz=224
=======
        model = YOLO("yolo26n-cls.pt")  # Load a pretrained model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        yolo classify train data=imagewoof model=yolo26n-cls.pt epochs=100 imgsz=224
>>>>>>> origin/main
        ```

For more details on available training arguments, refer to the [Training](../../modes/train.md) page.

### What versions of the ImageWoof dataset are available?

The ImageWoof dataset comes in three sizes:

<<<<<<< HEAD
1. **Full Size (imagewoof)**: Ideal for final training and benchmarking, containing full-sized images.
2. **Medium Size (imagewoof320)**: Resized images with a maximum edge length of 320 pixels, suited for faster training.
3. **Small Size (imagewoof160)**: Resized images with a maximum edge length of 160 pixels, perfect for rapid prototyping.

Use these versions by replacing 'imagewoof' in the dataset argument accordingly. Note, however, that smaller images may yield lower classification [accuracy](https://www.ultralytics.com/glossary/accuracy) but can be useful for quicker iterations.

### How do noisy labels in the ImageWoof dataset benefit training?

Noisy labels in the ImageWoof dataset simulate real-world conditions where labels might not always be accurate. Training models with this data helps develop robustness and generalization in image classification tasks. This prepares the models to handle ambiguous or mislabeled data effectively, which is often encountered in practical applications.

### What are the key challenges of using the ImageWoof dataset?

The primary challenge of the ImageWoof dataset lies in the subtle differences among the dog breeds it includes. Since it focuses on 10 closely related breeds, distinguishing between them requires more advanced and fine-tuned image classification models. This makes ImageWoof an excellent benchmark to test the capabilities and improvements of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models.
=======
1. **Full Size (`imagewoof`)**: Ideal for final training and benchmarking, containing full-sized images.
2. **Medium Size (`imagewoof320`)**: Resized images with a maximum edge length of 320 pixels, suited for faster training.
3. **Small Size (`imagewoof160`)**: Resized images with a maximum edge length of 160 pixels, perfect for rapid prototyping.

Use these versions by replacing `imagewoof` in the dataset argument accordingly. Note that smaller images may yield lower classification [accuracy](https://www.ultralytics.com/glossary/accuracy) but are useful for quicker iterations.

### How do noisy labels in the ImageWoof dataset benefit training?

Noisy labels in the ImageWoof dataset simulate real-world conditions where labels are not always accurate. Training models with this data helps develop robustness and generalization in image classification tasks. It prepares models to handle ambiguous or mislabeled data effectively, which is often encountered in practical applications.

### What are the key challenges of using the ImageWoof dataset?

The primary challenge of ImageWoof lies in the subtle differences among the dog breeds it includes. Because it focuses on 10 closely related breeds, distinguishing between them requires more advanced and fine-tuned image classification models. This makes ImageWoof an excellent benchmark to test the capabilities and improvements of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models.
>>>>>>> origin/main

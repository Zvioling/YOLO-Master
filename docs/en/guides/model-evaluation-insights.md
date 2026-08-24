---
<<<<<<< HEAD
comments: true
description: Explore the most effective ways to assess and refine YOLO11 models for better performance. Learn about evaluation metrics, fine-tuning processes, and how to customize your model for specific needs.
keywords: Model Evaluation, Machine Learning Model Evaluation, Fine Tuning Machine Learning, Fine Tune Model, Evaluating Models, Model Fine Tuning, How to Fine Tune a Model
=======
title: YOLO Model Evaluation and Fine-Tuning
comments: true
description: Learn how to evaluate YOLO26 models with metrics like mAP and IoU, then fine-tune them in Python to boost detection accuracy on your own dataset.
keywords: model evaluation, fine-tuning YOLO, mAP, IoU, confidence score, model validation, YOLO26 metrics, fine-tune YOLO26, improve detection accuracy
>>>>>>> origin/main
---

# Insights on Model Evaluation and Fine-Tuning

<<<<<<< HEAD
## Introduction

Once you've [trained](./model-training-tips.md) your computer vision model, evaluating and refining it to perform optimally is essential. Just training your model isn't enough. You need to make sure that your model is accurate, efficient, and fulfills the [objective](./defining-project-goals.md) of your computer vision project. By evaluating and fine-tuning your model, you can identify weaknesses, improve its accuracy, and boost overall performance.
=======
After [training](./model-training-tips.md) a YOLO model, the next step is to measure how well it performs and fine-tune it to close the gaps. Evaluation uses metrics like [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and [IoU](https://www.ultralytics.com/glossary/intersection-over-union-iou) to quantify accuracy, while fine-tuning adjusts training parameters to strengthen weak spots so the model meets your [project's objective](./defining-project-goals.md). This guide explains the key evaluation metrics, how to read them, and the fine-tuning techniques that elevate your model's capabilities.
>>>>>>> origin/main

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/-aYO-6VaDrw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Insights into Model Evaluation and Fine-Tuning | Tips for Improving Mean Average Precision
</p>

<<<<<<< HEAD
In this guide, we share insights on model evaluation and fine-tuning to make this [step of a computer vision project](./steps-of-a-cv-project.md) more approachable. We discuss how to understand evaluation metrics and implement fine-tuning techniques, giving you the knowledge to elevate your model's capabilities.

## Evaluating Model Performance Using Metrics

Evaluating how well a model performs helps us understand how effectively it works. Various metrics are used to measure performance. These [performance metrics](./yolo-performance-metrics.md) provide clear, numerical insights that can guide improvements toward making sure the model meets its intended goals. Let's take a closer look at a few key metrics.
=======
Evaluation and fine-tuning sit near the end of the [computer vision project workflow](./steps-of-a-cv-project.md), once training is underway and you need to verify the model is accurate, efficient, and ready for deployment.

## Key Evaluation Metrics

Various metrics measure how effectively a model performs. These [performance metrics](./yolo-performance-metrics.md) provide clear, numerical insights that guide improvements toward making sure the model meets its intended goals.
>>>>>>> origin/main

### Confidence Score

The confidence score represents the model's certainty that a detected object belongs to a particular class. It ranges from 0 to 1, with higher scores indicating greater confidence. The confidence score helps filter predictions; only detections with confidence scores above a specified threshold are considered valid.

<<<<<<< HEAD
_Quick Tip:_ When running inferences, if you aren't seeing any predictions, and you've checked everything else, try lowering the confidence score. Sometimes, the threshold is too high, causing the model to ignore valid predictions. Lowering the score allows the model to consider more possibilities. This might not meet your project goals, but it's a good way to see what the model can do and decide how to fine-tune it.
=======
!!! tip "Seeing no predictions?"

    When running inference, if you aren't seeing any predictions and you've checked everything else, try lowering the confidence threshold. Sometimes the threshold is too high, causing the model to ignore valid predictions. Lowering it lets the model consider more possibilities. This might not meet your final project goals, but it's a good way to see what the model can do and decide how to fine-tune it.
>>>>>>> origin/main

### Intersection over Union

[Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) is a metric in [object detection](https://www.ultralytics.com/glossary/object-detection) that measures how well the predicted [bounding box](https://www.ultralytics.com/glossary/bounding-box) overlaps with the ground truth bounding box. IoU values range from 0 to 1, where one stands for a perfect match. IoU is essential because it measures how closely the predicted boundaries match the actual object boundaries.

<p align="center">
<<<<<<< HEAD
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/intersection-over-union-overview.avif" alt="Intersection over Union Overview">
=======
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/intersection-over-union-overview.avif" alt="Intersection over Union Overview">
>>>>>>> origin/main
</p>

### Mean Average Precision

<<<<<<< HEAD
[Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) is a way to measure how well an object detection model performs. It looks at the precision of detecting each object class, averages these scores, and gives an overall number that shows how accurately the model can identify and classify objects.

Let's focus on two specific mAP metrics:

- *mAP@.5:* Measures the average precision at a single IoU (Intersection over Union) threshold of 0.5. This metric checks if the model can correctly find objects with a looser [accuracy](https://www.ultralytics.com/glossary/accuracy) requirement. It focuses on whether the object is roughly in the right place, not needing perfect placement. It helps see if the model is generally good at spotting objects.
=======
[Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) measures how well an object detection model performs overall. It looks at the precision of detecting each object class, averages these scores, and gives a single number that shows how accurately the model can identify and classify objects.

Two mAP metrics are most commonly reported:

- *mAP@.5:* Measures the average precision at a single IoU threshold of 0.5. This metric checks if the model can correctly find objects with a looser accuracy requirement. It focuses on whether the object is roughly in the right place, not needing perfect placement, and helps see if the model is generally good at spotting objects.
>>>>>>> origin/main
- *mAP@.5:.95:* Averages the mAP values calculated at multiple IoU thresholds, from 0.5 to 0.95 in 0.05 increments. This metric is more detailed and strict. It gives a fuller picture of how accurately the model can find objects at different levels of strictness and is especially useful for applications that need precise object detection.

Other mAP metrics include mAP@0.75, which uses a stricter IoU threshold of 0.75, and mAP@small, medium, and large, which evaluate precision across objects of different sizes.

<p align="center">
<<<<<<< HEAD
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/mean-average-precision-overview.avif" alt="Mean Average Precision Overview">
</p>

## Evaluating YOLO11 Model Performance

With respect to YOLO11, you can use the [validation mode](../modes/val.md) to evaluate the model. Also, be sure to take a look at our guide that goes in-depth into [YOLO11 performance metrics](./yolo-performance-metrics.md) and how they can be interpreted.

### Common Community Questions

When evaluating your YOLO11 model, you might run into a few hiccups. Based on common community questions, here are some tips to help you get the most out of your YOLO11 model:

#### Handling Variable Image Sizes

Evaluating your YOLO11 model with images of different sizes can help you understand its performance on diverse datasets. Using the `rect=true` validation parameter, YOLO11 adjusts the network's stride for each batch based on the image sizes, allowing the model to handle rectangular images without forcing them to a single size.

The `imgsz` validation parameter sets the maximum dimension for image resizing, which is 640 by default. You can adjust this based on your dataset's maximum dimensions and the GPU memory available. Even with `imgsz` set, `rect=true` lets the model manage varying image sizes effectively by dynamically adjusting the stride.

#### Accessing YOLO11 Metrics

If you want to get a deeper understanding of your YOLO11 model's performance, you can easily access specific evaluation metrics with a few lines of Python code. The code snippet below will let you load your model, run an evaluation, and print out various metrics that show how well your model is doing.
=======
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mean-average-precision-overview.avif" alt="Mean average precision mAP metric">
</p>

## Evaluating a YOLO26 Model

You can evaluate a trained YOLO26 model with the [validation mode](../modes/val.md). For a deeper look at how each metric is computed and interpreted, see the [YOLO26 performance metrics](./yolo-performance-metrics.md) guide.

### Handling Variable Image Sizes

Evaluating your model on images of different sizes helps you understand its performance on diverse datasets. The `rect=true` validation parameter groups images by aspect ratio and pads each batch to the smallest shape that fits, so rectangular images are evaluated without being forced into a square.

The `imgsz` parameter sets the image size used during validation, applied as a square. If you don't set it explicitly, YOLO26 reuses the value saved in the model's settings (640 for the official pretrained models, or whatever size a custom checkpoint was trained at). With `rect=true`, YOLO26 constrains the longer side to `imgsz` and pads the shorter side to a stride multiple, preserving the aspect ratio. Adjust `imgsz` based on your dataset's dimensions and the GPU memory available.

### Accessing YOLO26 Metrics

To understand your model's performance in detail, you can access specific evaluation metrics with a few lines of Python. The snippet below loads a model, runs validation, and prints the most useful metrics.
>>>>>>> origin/main

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load the model
        model = YOLO("yolo11n.pt")

        # Run the evaluation
        results = model.val(data="coco8.yaml")

        # Print specific metrics
        print("Class indices with average precision:", results.ap_class_index)
        print("Average precision for all classes:", results.box.all_ap)
        print("Average precision:", results.box.ap)
        print("Average precision at IoU=0.50:", results.box.ap50)
        print("Class indices for average precision:", results.box.ap_class_index)
        print("Class-specific results:", results.box.class_result)
        print("F1 score:", results.box.f1)
        print("F1 score curve:", results.box.f1_curve)
        print("Overall fitness score:", results.box.fitness)
        print("Mean average precision:", results.box.map)
        print("Mean average precision at IoU=0.50:", results.box.map50)
        print("Mean average precision at IoU=0.75:", results.box.map75)
        print("Mean average precision for different IoU thresholds:", results.box.maps)
        print("Mean results for different metrics:", results.box.mean_results)
        print("Mean precision:", results.box.mp)
        print("Mean recall:", results.box.mr)
        print("Precision:", results.box.p)
        print("Precision curve:", results.box.p_curve)
        print("Precision values:", results.box.prec_values)
        print("Specific precision metrics:", results.box.px)
        print("Recall:", results.box.r)
        print("Recall curve:", results.box.r_curve)
        ```

The results object also includes speed metrics like preprocess time, inference time, loss, and postprocess time. By analyzing these metrics, you can fine-tune and optimize your YOLO11 model for better performance, making it more effective for your specific use case.

## How Does Fine-Tuning Work?

Fine-tuning involves taking a pretrained model and adjusting its parameters to improve performance on a specific task or dataset. The process, also known as model retraining, allows the model to better understand and predict outcomes for the specific data it will encounter in real-world applications. You can retrain your model based on your model evaluation to achieve optimal results.

## Tips for Fine-Tuning Your Model

Fine-tuning a model means paying close attention to several vital parameters and techniques to achieve optimal performance. Here are some essential tips to guide you through the process.

### Starting With a Higher Learning Rate

Usually, during the initial training [epochs](https://www.ultralytics.com/glossary/epoch), the learning rate starts low and gradually increases to stabilize the training process. However, since your model has already learned some features from the previous dataset, starting with a higher [learning rate](https://www.ultralytics.com/glossary/learning-rate) right away can be more beneficial.

When evaluating your YOLO11 model, you can set the `warmup_epochs` validation parameter to `warmup_epochs=0` to prevent the learning rate from starting too low. By following this process, the training will continue from the provided weights, adjusting to the nuances of your new data.

### Image Tiling for Small Objects

Image tiling can improve detection accuracy for small objects. By dividing larger images into smaller segments, such as splitting 1280x1280 images into multiple 640x640 segments, you maintain the original resolution, and the model can learn from high-resolution fragments. When using YOLO11, make sure to adjust your labels for these new segments correctly.

## Engage with the Community

Sharing your ideas and questions with other [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) enthusiasts can inspire creative solutions to roadblocks in your projects. Here are some excellent ways to learn, troubleshoot, and connect.

### Finding Help and Support

- **GitHub Issues:** Explore the YOLO11 GitHub repository and use the [Issues tab](https://github.com/ultralytics/ultralytics/issues) to ask questions, report bugs, and suggest features. The community and maintainers are available to assist with any issues you encounter.
- **Ultralytics Discord Server:** Join the [Ultralytics Discord server](https://discord.com/invite/ultralytics) to connect with other users and developers, get support, share knowledge, and brainstorm ideas.

### Official Documentation

- **Ultralytics YOLO11 Documentation:** Check out the [official YOLO11 documentation](./index.md) for comprehensive guides and valuable insights on various computer vision tasks and projects.

## Final Thoughts

Evaluating and fine-tuning your computer vision model are important steps for successful [model deployment](https://www.ultralytics.com/glossary/model-deployment). These steps help make sure that your model is accurate, efficient, and suited to your overall application. The key to training the best model possible is continuous experimentation and learning. Don't hesitate to tweak parameters, try new techniques, and explore different datasets. Keep experimenting and pushing the boundaries of what's possible!

## FAQ

### What are the key metrics for evaluating YOLO11 model performance?

To evaluate YOLO11 model performance, important metrics include Confidence Score, Intersection over Union (IoU), and Mean Average Precision (mAP). Confidence Score measures the model's certainty for each detected object class. IoU evaluates how well the predicted bounding box overlaps with the ground truth. Mean Average Precision (mAP) aggregates precision scores across classes, with mAP@.5 and mAP@.5:.95 being two common types for varying IoU thresholds. Learn more about these metrics in our [YOLO11 performance metrics guide](./yolo-performance-metrics.md).

### How can I fine-tune a pretrained YOLO11 model for my specific dataset?

Fine-tuning a pretrained YOLO11 model involves adjusting its parameters to improve performance on a specific task or dataset. Start by evaluating your model using metrics, then set a higher initial learning rate by adjusting the `warmup_epochs` parameter to 0 for immediate stability. Use parameters like `rect=true` for handling varied image sizes effectively. For more detailed guidance, refer to our section on [fine-tuning YOLO11 models](#how-does-fine-tuning-work).

### How can I handle variable image sizes when evaluating my YOLO11 model?

To handle variable image sizes during evaluation, use the `rect=true` parameter in YOLO11, which adjusts the network's stride for each batch based on image sizes. The `imgsz` parameter sets the maximum dimension for image resizing, defaulting to 640. Adjust `imgsz` to suit your dataset and GPU memory. For more details, visit our [section on handling variable image sizes](#handling-variable-image-sizes).

### What practical steps can I take to improve mean average precision for my YOLO11 model?

Improving mean average precision (mAP) for a YOLO11 model involves several steps:
=======
        # Load a model
        model = YOLO("yolo26n.pt")

        # Run validation on your dataset
        results = model.val(data="coco8.yaml")

        # Overall metrics
        print("mAP50-95:", results.box.map)  # mAP at IoU 0.50:0.95
        print("mAP50:", results.box.map50)  # mAP at IoU 0.50
        print("mAP75:", results.box.map75)  # mAP at IoU 0.75
        print("Mean precision:", results.box.mp)
        print("Mean recall:", results.box.mr)
        print("Fitness:", results.box.fitness())  # weighted score used for model selection

        # Per-class metrics
        print("Class indices evaluated:", results.box.ap_class_index)
        print("Per-class mAP50-95:", results.box.maps)

        # Per-image precision, recall, F1, TP, FP, and FN
        print("Per-image metrics:", results.box.image_metrics)

        # Per-stage timing breakdown in milliseconds per image
        print("Timing breakdown (ms/image):", results.speed)
        ```

Note that `fitness()` is a method and must be called with parentheses, while metrics like `map`, `map50`, and `mp` are properties accessed directly.

The `results.box.image_metrics` attribute is a per-image dictionary keyed by image filename, holding `precision`, `recall`, `f1`, `tp`, `fp`, and `fn` at IoU 0.5 for each image. Preprocessing, inference, loss, and postprocessing timings are reported separately in the `results.speed` dictionary. Together, these let you pinpoint which images the model struggles with and fine-tune accordingly.

## Fine-Tuning Your Model

Fine-tuning takes a pretrained model and adjusts its parameters to improve performance on a specific task or dataset. Also known as model retraining, it lets the model better understand and predict outcomes for the data it will encounter in real-world applications. Based on your evaluation results, you retrain the model to achieve optimal results by paying close attention to a few key parameters and techniques.

### Starting With a Higher Learning Rate

During normal training, the [learning rate](https://www.ultralytics.com/glossary/learning-rate) starts low and gradually increases over the first few epochs to stabilize early updates. When fine-tuning, the model already carries useful features from pretraining, so you can skip this warmup and start adapting to your new data right away.

Set the `warmup_epochs` training argument to `0` in `model.train()` to disable the warmup phase. Training then continues from the pretrained weights at the configured base learning rate (`lr0`) instead of ramping up to it, adjusting to the nuances of your new data.

!!! example "Fine-tune without learning-rate warmup"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Fine-tune with the warmup phase disabled
        model.train(data="coco8.yaml", epochs=10, warmup_epochs=0)
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=coco8.yaml epochs=10 warmup_epochs=0
        ```

### Image Tiling for Small Objects

Image tiling can improve detection accuracy for small objects. By dividing larger images into smaller segments, such as splitting 1280x1280 images into multiple 640x640 segments, you preserve the original resolution and let the model learn from high-resolution fragments. Ultralytics supports this at inference time through [SAHI tiled inference](./sahi-tiled-inference.md). When training on tiled images, make sure to adjust your labels for each new segment correctly.

## Conclusion

Evaluating and fine-tuning are what turn a trained model into a dependable, deployable one: metrics like mAP and IoU expose weaknesses, and targeted parameter changes address them. Start with the [validation mode](../modes/val.md) to benchmark your model, then apply the fine-tuning techniques above and keep iterating with new parameters, techniques, and datasets. If questions come up along the way, ask the community on the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/issues) or the [Ultralytics Discord server](https://discord.com/invite/ultralytics).

## FAQ

### What are the key metrics for evaluating YOLO26 model performance?

To evaluate YOLO26 model performance, important metrics include Confidence Score, Intersection over Union (IoU), and Mean Average Precision (mAP). Confidence Score measures the model's certainty for each detected object class. IoU evaluates how well the predicted bounding box overlaps with the ground truth. Mean Average Precision (mAP) aggregates precision scores across classes, with mAP@.5 and mAP@.5:.95 being two common types for varying IoU thresholds. Learn more about these metrics in our [YOLO26 performance metrics guide](./yolo-performance-metrics.md).

### How can I fine-tune a pretrained YOLO26 model for my specific dataset?

Fine-tuning a pretrained YOLO26 model involves adjusting its parameters to improve performance on a specific task or dataset. Start by evaluating your model with metrics, then set the `warmup_epochs` training argument to `0` in `model.train()` so the learning rate starts at the configured base value immediately instead of ramping up. During evaluation, parameters like `rect=true` help handle varied image sizes effectively. For more detailed guidance, refer to our section on [fine-tuning your model](#fine-tuning-your-model).

### How can I handle variable image sizes when evaluating my YOLO26 model?

To handle variable image sizes during evaluation, use the `rect=true` parameter in YOLO26, which groups images by aspect ratio and pads each batch instead of forcing every image to a square. The `imgsz` parameter sets the image size for validation; if you don't override it, YOLO26 reuses the model's saved value (640 for the official pretrained models). Adjust `imgsz` to suit your dataset and GPU memory. For more details, visit our [section on handling variable image sizes](#handling-variable-image-sizes).

### What practical steps can I take to improve mean average precision for my YOLO26 model?

Improving mean average precision (mAP) for a YOLO26 model involves several steps:
>>>>>>> origin/main

1. **Tuning Hyperparameters**: Experiment with different learning rates, [batch sizes](https://www.ultralytics.com/glossary/batch-size), and image augmentations.
2. **[Data Augmentation](https://www.ultralytics.com/glossary/data-augmentation)**: Use techniques like Mosaic and MixUp to create diverse training samples.
3. **Image Tiling**: Split larger images into smaller tiles to improve detection accuracy for small objects.
<<<<<<< HEAD
   Refer to our detailed guide on [model fine-tuning](#tips-for-fine-tuning-your-model) for specific strategies.

### How do I access YOLO11 model evaluation metrics in Python?

You can access YOLO11 model evaluation metrics using Python with the following steps:
=======

Refer to our detailed section on [fine-tuning your model](#fine-tuning-your-model) for specific strategies.

### How do I access YOLO26 model evaluation metrics in Python?

You can access YOLO26 model evaluation metrics using Python after running validation:
>>>>>>> origin/main

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

<<<<<<< HEAD
        # Load the model
        model = YOLO("yolo11n.pt")

        # Run the evaluation
        results = model.val(data="coco8.yaml")

        # Print specific metrics
        print("Class indices with average precision:", results.ap_class_index)
        print("Average precision for all classes:", results.box.all_ap)
        print("Mean average precision at IoU=0.50:", results.box.map50)
        print("Mean recall:", results.box.mr)
        ```

Analyzing these metrics helps fine-tune and optimize your YOLO11 model. For a deeper dive, check out our guide on [YOLO11 metrics](../modes/val.md).
=======
        # Load a model
        model = YOLO("yolo26n.pt")

        # Run validation
        results = model.val(data="coco8.yaml")

        # Access key metrics
        print("Mean average precision at IoU=0.50:", results.box.map50)
        print("Mean average precision at IoU=0.50:0.95:", results.box.map)
        print("Mean recall:", results.box.mr)
        print("Class indices evaluated:", results.box.ap_class_index)
        ```

Analyzing these metrics helps you fine-tune and optimize your YOLO26 model. For a deeper dive, check out our guide on [YOLO performance metrics](./yolo-performance-metrics.md).
>>>>>>> origin/main

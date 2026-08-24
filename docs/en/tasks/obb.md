---
comments: true
<<<<<<< HEAD
description: Discover how to detect objects with rotation for higher precision using YOLO11 OBB models. Learn, train, validate, and export OBB models effortlessly.
keywords: Oriented Bounding Boxes, OBB, Object Detection, YOLO11, Ultralytics, DOTAv1, Model Training, Model Export, AI, Machine Learning
model_name: yolo11n-obb
=======
description: Discover how to detect objects with rotation for higher precision using YOLO26 OBB models. Learn, train, validate, and export OBB models effortlessly.
keywords: Oriented Bounding Boxes, OBB, Object Detection, YOLO26, Ultralytics, DOTAv1, Model Training, Model Export, AI, Machine Learning
model_name: yolo26n-obb
>>>>>>> origin/main
---

# Oriented Bounding Boxes [Object Detection](https://www.ultralytics.com/glossary/object-detection)

<!-- obb task poster -->

Oriented object detection goes a step further than standard object detection by introducing an extra angle to locate objects more accurately in an image.

The output of an oriented object detector is a set of rotated bounding boxes that precisely enclose the objects in the image, along with class labels and confidence scores for each box. Oriented bounding boxes are particularly useful when objects appear at various angles, such as in aerial imagery, where traditional axis-aligned bounding boxes may include unnecessary background.

<!-- youtube video link for obb task -->

!!! tip

<<<<<<< HEAD
    YOLO11 OBB models use the `-obb` suffix, i.e., `yolo11n-obb.pt`, and are pretrained on [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Z7Z9pHF8wJc"
=======
    YOLO26 OBB models use the `-obb` suffix, i.e., `yolo26n-obb.pt`, and are pretrained on [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/128JhhR2DlM"
>>>>>>> origin/main
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> Object Detection using Ultralytics YOLO Oriented Bounding Boxes (YOLO-OBB)
=======
  <strong>Watch:</strong> How to Detect & Track Objects with Ultralytics YOLO26 Oriented Bounding Boxes (OBB) | Ship Tracking 🚢
>>>>>>> origin/main
</p>

## Visual Samples

<<<<<<< HEAD
|                                              Ships Detection using OBB                                               |                                               Vehicle Detection using OBB                                                |
| :------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| ![Ships Detection using OBB](https://github.com/ultralytics/docs/releases/download/0/ships-detection-using-obb.avif) | ![Vehicle Detection using OBB](https://github.com/ultralytics/docs/releases/download/0/vehicle-detection-using-obb.avif) |

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

YOLO11 pretrained OBB models are shown here, which are pretrained on the [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) dataset.
=======
|                                               Ships Detection using OBB                                               |                                                Vehicle Detection using OBB                                                |
| :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| ![Ships Detection using OBB](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ships-detection-using-obb.avif) | ![Vehicle Detection using OBB](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-detection-using-obb.avif) |

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained OBB models are shown here, which are pretrained on the [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) dataset.
>>>>>>> origin/main

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-obb-perf.md" %}

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`
<<<<<<< HEAD

## Train

Train YOLO11n-obb on the DOTA8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! note

    OBB angles are constrained to the range **0–90 degrees** (exclusive of 90). Angles of 90 degrees or greater are not supported.
=======
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-obb on the DOTA8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! note

    An OBB and its 180° rotation are identical, so rotation is defined modulo 180° and the box has no direction. Internally the angle is stored in radians and normalized to **`[-π/4, 3π/4)`** (`[-45°, 135°)`), the box width `w` is taken as the longer side, and the angle is defined as the clockwise angle from the positive x-axis to the direction of `w`. The `[0°, 90°)` form is the regularized DOTA-style convention and is not applied at training or inference.
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
=======
        model = YOLO("yolo26n-obb.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-obb.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-obb.yaml").load("yolo26n-obb.pt")  # build from YAML and transfer weights
>>>>>>> origin/main

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
<<<<<<< HEAD
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml pretrained=yolo11n-obb.pt epochs=100 imgsz=640
=======
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml pretrained=yolo26n-obb.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uZ7SymQfqKI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
<<<<<<< HEAD
  <strong>Watch:</strong> How to Train Ultralytics YOLO-OBB (Oriented Bounding Boxes) Models on DOTA Dataset using Ultralytics HUB
=======
  <strong>Watch:</strong> How to Train Ultralytics YOLO-OBB (Oriented Bounding Boxes) Models on DOTA Dataset using Ultralytics Platform
>>>>>>> origin/main
</p>

### Dataset format

<<<<<<< HEAD
OBB dataset format can be found in detail in the [Dataset Guide](../datasets/obb/index.md). The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1, following this structure:

```
=======
OBB dataset format can be found in detail in the [Dataset Guide](../datasets/obb/index.md). The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1, following this structure. [Ultralytics Platform](https://platform.ultralytics.com) supports OBB annotation with a dedicated oriented bounding box drawing tool:

```text
>>>>>>> origin/main
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Internally, YOLO processes losses and outputs in the `xywhr` format, which represents the [bounding box](https://www.ultralytics.com/glossary/bounding-box)'s center point (xy), width, height, and rotation.

## Val

<<<<<<< HEAD
Validate trained YOLO11n-obb model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the DOTA8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.
=======
Validate trained YOLO26n-obb model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the DOTA8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")  # load an official model
=======
        model = YOLO("yolo26n-obb.pt")  # load an official model
>>>>>>> origin/main
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val(data="dota8.yaml")  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list containing mAP50-95(B) for each category
<<<<<<< HEAD
=======
        metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
>>>>>>> origin/main
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml         # val official model
=======
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml         # val official model
>>>>>>> origin/main
        yolo obb val model=path/to/best.pt data=path/to/data.yaml # val custom model
        ```

## Predict

<<<<<<< HEAD
Use a trained YOLO11n-obb model to run predictions on images.
=======
Use a trained YOLO26n-obb model to run predictions on images.
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")  # load an official model
=======
        model = YOLO("yolo26n-obb.pt")  # load an official model
>>>>>>> origin/main
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

        # Access the results
        for result in results:
            xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
            confs = result.obb.conf  # confidence score of each box
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo obb predict model=yolo11n-obb.pt source='https://ultralytics.com/images/boats.jpg'  # predict with official model
=======
        yolo obb predict model=yolo26n-obb.pt source='https://ultralytics.com/images/boats.jpg'  # predict with official model
>>>>>>> origin/main
        yolo obb predict model=path/to/best.pt source='https://ultralytics.com/images/boats.jpg' # predict with custom model
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5XYdm5CYODA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Detect and Track Storage Tanks using Ultralytics YOLO-OBB | Oriented Bounding Boxes | DOTA
</p>

See full `predict` mode details in the [Predict](../modes/predict.md) page.

<<<<<<< HEAD
## Export

Export a YOLO11n-obb model to a different format like ONNX, CoreML, etc.
=======
### Results Output

Oriented bounding box detection returns one `Results` object per image. The primary prediction field is `result.obb`,
which contains rotated boxes, class IDs, and confidence scores for each detected object.

| Attribute             | Type            | Shape     | Description                              |
| --------------------- | --------------- | --------- | ---------------------------------------- |
| `result.obb`          | `OBB`           | `(N)`     | Oriented boxes.                          |
| `result.obb.data`     | `torch.float32` | `(N,7/8)` | Raw rotated boxes with confidence/class. |
| `result.obb.xywhr`    | `torch.float32` | `(N,5)`   | `xywhr` rotated boxes.                   |
| `result.obb.xyxyxyxy` | `torch.float32` | `(N,4,2)` | Four corner points.                      |
| `result.obb.conf`     | `torch.float32` | `(N,)`    | Confidence scores.                       |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n-obb model to a different format like ONNX, CoreML, etc.
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")  # load an official model
=======
        model = YOLO("yolo26n-obb.pt")  # load an official model
>>>>>>> origin/main
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo export model=yolo11n-obb.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO11-obb export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo11n-obb.onnx`. Usage examples are shown for your model after export completes.
=======
        yolo export model=yolo26n-obb.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-obb export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-obb.onnx`. Usage examples are shown for your model after export completes.
>>>>>>> origin/main

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## Real-World Applications

<<<<<<< HEAD
OBB detection with YOLO11 has numerous practical applications across various industries:
=======
OBB detection with YOLO26 has numerous practical applications across various industries:
>>>>>>> origin/main

- **Maritime and Port Management**: Detecting ships and vessels at various angles for [fleet management](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-obb-object-detection) and monitoring.
- **Urban Planning**: Analyzing buildings and infrastructure from aerial imagery.
- **Agriculture**: Monitoring crops and agricultural equipment from drone footage.
- **Energy Sector**: Inspecting solar panels and wind turbines at different orientations.
- **Transportation**: Tracking vehicles on roads and in parking lots from various perspectives.

These applications benefit from OBB's ability to precisely fit objects at any angle, providing more accurate detection than traditional bounding boxes.

## FAQ

### What are Oriented Bounding Boxes (OBB) and how do they differ from regular bounding boxes?

Oriented Bounding Boxes (OBB) include an additional angle to enhance object localization accuracy in images. Unlike regular bounding boxes, which are axis-aligned rectangles, OBBs can rotate to fit the orientation of the object better. This is particularly useful for applications requiring precise object placement, such as aerial or satellite imagery ([Dataset Guide](../datasets/obb/index.md)).

<<<<<<< HEAD
### How do I train a YOLO11n-obb model using a custom dataset?

To train a YOLO11n-obb model with a custom dataset, follow the example below using Python or CLI:
=======
### How do I train a YOLO26n-obb model using a custom dataset?

To train a YOLO26n-obb model with a custom dataset, follow the example below using Python or CLI:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")
=======
        model = YOLO("yolo26n-obb.pt")
>>>>>>> origin/main

        # Train the model
        results = model.train(data="path/to/custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo obb train data=path/to/custom_dataset.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
=======
        yolo obb train data=path/to/custom_dataset.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
>>>>>>> origin/main
        ```

For more training arguments, check the [Configuration](../usage/cfg.md) section.

<<<<<<< HEAD
### What datasets can I use for training YOLO11-OBB models?

YOLO11-OBB models are pretrained on datasets like [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) but you can use any dataset formatted for OBB. Detailed information on OBB dataset formats can be found in the [Dataset Guide](../datasets/obb/index.md).

### How can I export a YOLO11-OBB model to ONNX format?

Exporting a YOLO11-OBB model to ONNX format is straightforward using either Python or CLI:
=======
### What datasets can I use for training YOLO26-OBB models?

YOLO26-OBB models are pretrained on datasets like [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) but you can use any dataset formatted for OBB. Detailed information on OBB dataset formats can be found in the [Dataset Guide](../datasets/obb/index.md).

### How can I export a YOLO26-OBB model to ONNX format?

Exporting a YOLO26-OBB model to ONNX format is straightforward using either Python or CLI:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")
=======
        model = YOLO("yolo26n-obb.pt")
>>>>>>> origin/main

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo export model=yolo11n-obb.pt format=onnx
=======
        yolo export model=yolo26n-obb.pt format=onnx
>>>>>>> origin/main
        ```

For more export formats and details, refer to the [Export](../modes/export.md) page.

<<<<<<< HEAD
### How do I validate the accuracy of a YOLO11n-obb model?

To validate a YOLO11n-obb model, you can use Python or CLI commands as shown below:
=======
### How do I validate the accuracy of a YOLO26n-obb model?

To validate a YOLO26n-obb model, you can use Python or CLI commands as shown below:
>>>>>>> origin/main

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
<<<<<<< HEAD
        model = YOLO("yolo11n-obb.pt")
=======
        model = YOLO("yolo26n-obb.pt")
>>>>>>> origin/main

        # Validate the model
        metrics = model.val(data="dota8.yaml")
        ```

    === "CLI"

        ```bash
<<<<<<< HEAD
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml
=======
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml
>>>>>>> origin/main
        ```

See full validation details in the [Val](../modes/val.md) section.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_esmoe_int8.py
======================
ES-MoE INT8 量化导出（端侧部署专用）。

功能
----
- 将 ES-MoE 模型导出为 FP16 / INT8 ONNX
- 端侧部署：移动端、嵌入式设备使用 TensorRT Lite / ONNX Runtime Mobile
- 量化策略：
  - Weight-only INT8（无需校准数据）
  - Static INT8（需要少量校准数据）
  - Dynamic INT8（介于两者之间）

使用方法：
    # 导出 FP16 ONNX
    python scripts/export_esmoe_int8.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --output experiments_zviolin/runs/esmoe_export/fp16

    # 导出 INT8 ONNX（dynamic quantization）
    python scripts/export_esmoe_int8.py \\
        --model experiments_zviolin/runs/esmoe_v0_k2-2/weights/best.pt \\
        --quant int8_dynamic --output experiments_zviolin/runs/esmoe_export/int8_dyn
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def export_fp16(model_path: str, output_dir: Path, imgsz: int = 640):
    """导出 FP16 ONNX（GPU 友好）。"""
    print(f"[export_fp16] Loading {model_path}")
    yolo = YOLO(model_path)
    out = yolo.export(format="onnx", imgsz=imgsz, half=True, simplify=True)
    onnx_src = Path(out)
    onnx_dst = output_dir / "model_fp16.onnx"
    if onnx_src != onnx_dst:
        import shutil
        shutil.copy(str(onnx_src), str(onnx_dst))
    size_mb = onnx_dst.stat().st_size / 1024 / 1024
    print(f"[export_fp16] {onnx_dst} ({size_mb:.2f} MB)")
    return onnx_dst


def export_int8_dynamic(model_path: str, output_dir: Path, imgsz: int = 640):
    """导出 INT8 ONNX（dynamic quantization，weight-only）。"""
    print(f"[export_int8_dynamic] Loading {model_path}")
    yolo = YOLO(model_path)

    # 先导出 FP32 ONNX
    out_fp32 = yolo.export(format="onnx", imgsz=imgsz, simplify=True)
    onnx_fp32 = Path(out_fp32)

    # 动态量化
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        onnx_int8 = output_dir / "model_int8_dynamic.onnx"
        quantize_dynamic(
            model_input=str(onnx_fp32),
            model_output=str(onnx_int8),
            weight_type=QuantType.QInt8,
        )
        size_mb = onnx_int8.stat().st_size / 1024 / 1024
        print(f"[export_int8_dynamic] {onnx_int8} ({size_mb:.2f} MB)")
        return onnx_int8
    except ImportError:
        print("[export_int8_dynamic] onnxruntime.quantization not available, falling back to FP16")
        return export_fp16(model_path, output_dir, imgsz)


def export_int8_static(
    model_path: str,
    output_dir: Path,
    data_yaml: str,
    imgsz: int = 640,
    num_calib: int = 50,
):
    """导出 INT8 ONNX（static quantization，需要校准数据）。"""
    print(f"[export_int8_static] Loading {model_path}")
    yolo = YOLO(model_path)

    # 先导出 FP32 ONNX
    out_fp32 = yolo.export(format="onnx", imgsz=imgsz, simplify=True)
    onnx_fp32 = Path(out_fp32)

    # 加载校准数据
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(data_yaml)
    val_path = data.get("val", "")
    val_images = sorted(Path(val_path).glob("*.jpg"))[:num_calib]
    if not val_images:
        print("[export_int8_static] No validation images, skip calibration")
        return None

    # 创建校准数据读取器
    class CalibDataReader:
        def __init__(self, images, imgsz):
            self.images = images
            self.imgsz = imgsz
            self.index = 0

        def get_next(self):
            if self.index >= len(self.images):
                return None
            img_path = self.images[self.index]
            self.index += 1
            from PIL import Image
            import numpy as np
            img = Image.open(img_path).convert("RGB").resize((self.imgsz, self.imgsz))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[None]  # [1, 3, H, W]
            return arr

        def rewind(self):
            self.index = 0

    try:
        import onnx
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

        # 读取 ONNX 输入名（校准 reader 必须返回 {input_name: ndarray}）
        onnx_model = onnx.load(str(onnx_fp32))
        input_name = onnx_model.graph.input[0].name

        class CalibReaderWrapper(CalibrationDataReader):
            def __init__(self, reader):
                self.reader = reader

            def get_next(self):
                arr = self.reader.get_next()
                if arr is None:
                    return None
                return {input_name: arr}

            def rewind(self):
                self.reader.rewind()

        reader = CalibReaderWrapper(CalibDataReader(val_images, imgsz))
        onnx_int8 = output_dir / "model_int8_static.onnx"
        quantize_static(
            model_input=str(onnx_fp32),
            model_output=str(onnx_int8),
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        size_mb = onnx_int8.stat().st_size / 1024 / 1024
        print(f"[export_int8_static] {onnx_int8} ({size_mb:.2f} MB)")
        return onnx_int8
    except ImportError:
        print("[export_int8_static] onnxruntime.quantization not available")
        return None


def main():
    parser = argparse.ArgumentParser(description="ES-MoE 端侧量化导出")
    parser.add_argument("--model", type=str, required=True, help="ES-MoE 权重路径")
    parser.add_argument("--quant", type=str, default="fp16",
                        choices=["fp16", "int8_dynamic", "int8_static", "all"],
                        help="量化类型")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num-calib", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiments_zviolin/runs/esmoe_export")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export_esmoe_int8] Output dir: {output_dir}")

    if args.quant == "fp16" or args.quant == "all":
        export_fp16(args.model, output_dir, args.imgsz)

    if args.quant == "int8_dynamic" or args.quant == "all":
        export_int8_dynamic(args.model, output_dir, args.imgsz)

    if args.quant == "int8_static" or args.quant == "all":
        export_int8_static(args.model, output_dir, args.data, args.imgsz, args.num_calib)

    # 打印对比
    print(f"\n=== Export Summary ===")
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
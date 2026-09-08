"""
使用 ONNX 识别模型评估训练集标注。

脚本复用训练和推理共用的图像预处理、动态宽度填充及 CTC 解码逻辑，
逐条比较 ONNX 预测与 ``train_annotation.txt`` 中的标注，并输出整体准确率、
字符错误率、短输出和漏末字统计。错误明细与汇总结果写入 ``test_results``。

运行方式：
    .venv/bin/python recognition/evaluate_onnx_train.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from recognition.components.converter import LabelConverter
from recognition.components.dataset import (
    TextRecognitionDataset,
    dynamic_padding_collate_fn,
)


# ================= 配置区域 =================

ONNX_MODEL_PATH = (
    PROJECT_ROOT / "fulltrain_0831_205615_best_cer_0.0028_epoch_672.onnx"
)
TRAIN_ANNOTATION_PATH = PROJECT_ROOT / "recognition/data/train_annotation.txt"
KEYS_PATH = PROJECT_ROOT / "recognition/keys.txt"

ERROR_REPORT_PATH = PROJECT_ROOT / "test_results/recognition_onnx_train_errors.tsv"
SUMMARY_REPORT_PATH = PROJECT_ROOT / "test_results/recognition_onnx_train_summary.json"

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
NUM_CHANNELS = 1
BATCH_SIZE = 64
NUM_WORKERS = 0
TOP_ERROR_LIMIT = 30

# 设为正整数时只评估前 N 条标注；设为 None 时评估完整训练集。
MAX_SAMPLES: int | None = None


@dataclass(frozen=True)
class ErrorCase:
    """
    保存单个识别错误的完整上下文。

    Attributes:
        sample_index: 样本在有效标注中的一基序号。
        image_path: 相对于项目根目录的图片路径。
        image_width: 图片预处理后、批量填充前的有效像素宽度。
        expected: 训练标注中的期望文本。
        predicted: ONNX 模型的预测文本。
        edit_distance: 期望文本与预测文本的字符编辑距离。
        sample_cer: 当前样本的字符错误率。
        error_type: 根据输出长度和内容划分的错误类型。
    """

    sample_index: int
    image_path: str
    image_width: int
    expected: str
    predicted: str
    edit_distance: int
    sample_cer: float
    error_type: str


def levenshtein_distance(expected: str, predicted: str) -> int:
    """
    计算两个字符串之间的字符级 Levenshtein 编辑距离。

    Args:
        expected: 作为基准的训练标注。
        predicted: ONNX 模型输出的识别文本。

    Returns:
        将预测文本转换为标注文本所需的最少插入、删除和替换次数。
    """
    if len(expected) < len(predicted):
        expected, predicted = predicted, expected

    if not predicted:
        return len(expected)

    previous_row = list(range(len(predicted) + 1))
    for expected_index, expected_char in enumerate(expected, start=1):
        current_row = [expected_index]
        for predicted_index, predicted_char in enumerate(predicted, start=1):
            insertion_cost = current_row[predicted_index - 1] + 1
            deletion_cost = previous_row[predicted_index] + 1
            substitution_cost = (
                previous_row[predicted_index - 1]
                + (expected_char != predicted_char)
            )
            current_row.append(
                min(insertion_cost, deletion_cost, substitution_cost)
            )
        previous_row = current_row

    return previous_row[-1]


def classify_error(expected: str, predicted: str) -> str:
    """
    根据预测与标注的关系确定错误类型。

    Args:
        expected: 训练标注中的期望文本。
        predicted: ONNX 模型输出的识别文本。

    Returns:
        ``漏末字``、``输出较短``、``输出较长`` 或 ``字符错误``。
    """
    if len(expected) >= 2 and predicted == expected[:-1]:
        return "漏末字"
    if len(predicted) < len(expected):
        return "输出较短"
    if len(predicted) > len(expected):
        return "输出较长"
    return "字符错误"


def load_evaluation_dataset(
    annotation_path: Path,
    max_samples: int | None,
) -> TextRecognitionDataset:
    """
    加载训练标注，并构造关闭数据增强的确定性评估数据集。

    Args:
        annotation_path: 训练标注文件路径。
        max_samples: 最大评估样本数；为 ``None`` 时保留全部样本。

    Returns:
        与正式推理预处理一致的文本识别数据集。

    Raises:
        FileNotFoundError: 标注文件或标注引用的图片不存在时抛出。
        ValueError: 标注为空、包含空标签或 ``max_samples`` 非法时抛出。
    """
    if not annotation_path.is_file():
        raise FileNotFoundError(f"训练标注文件不存在: {annotation_path}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("MAX_SAMPLES 必须是正整数或 None")

    dataset = TextRecognitionDataset(
        annotation_path=str(annotation_path),
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        num_channels=NUM_CHANNELS,
        is_train=False,
    )

    if max_samples is not None:
        dataset.image_paths = dataset.image_paths[:max_samples]
        dataset.labels = dataset.labels[:max_samples]

    if not dataset.image_paths:
        raise ValueError(f"训练标注中没有有效样本: {annotation_path}")

    empty_label_indices = [
        index + 1
        for index, label in enumerate(dataset.labels)
        if not label
    ]
    if empty_label_indices:
        preview = empty_label_indices[:10]
        raise ValueError(f"训练标注包含空标签，样本序号示例: {preview}")

    missing_paths = [
        image_path
        for image_path in dataset.image_paths
        if not Path(image_path).is_file()
    ]
    if missing_paths:
        preview = missing_paths[:10]
        raise FileNotFoundError(
            f"训练标注引用了 {len(missing_paths)} 张不存在的图片，示例: {preview}"
        )

    return dataset


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    """
    创建 ONNX Runtime 会话并验证识别模型的输入契约。

    Args:
        model_path: 待评估的 ONNX 模型路径。

    Returns:
        使用 CPUExecutionProvider 的 ONNX Runtime 推理会话。

    Raises:
        FileNotFoundError: ONNX 模型不存在时抛出。
        ValueError: 模型缺少 ``input`` 或 ``img_widths`` 输入时抛出。
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session_options.intra_op_num_threads = os.cpu_count() or 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {model_input.name for model_input in session.get_inputs()}
    required_inputs = {"input", "img_widths"}
    missing_inputs = sorted(required_inputs - input_names)
    if missing_inputs:
        raise ValueError(
            f"ONNX 模型缺少必要输入 {missing_inputs}，实际输入为 {sorted(input_names)}"
        )

    return session


def infer_batch(
    session: ort.InferenceSession,
    converter: LabelConverter,
    images: torch.Tensor,
    image_widths: torch.Tensor,
) -> list[str]:
    """
    对一个动态宽度批次执行 ONNX 推理并完成 CTC 贪心解码。

    Args:
        session: 已加载的 ONNX Runtime 会话。
        converter: 与训练字典一致的标签转换器。
        images: 形状为 ``[B, C, 32, W]`` 的批量图片张量。
        image_widths: 形状为 ``[B]`` 的填充前有效像素宽度。

    Returns:
        与批次样本顺序一致的预测文本列表。

    Raises:
        ValueError: ONNX 输出形状、批量维或类别数不符合识别契约时抛出。
    """
    feeds = {
        "input": images.numpy().astype(np.float32, copy=False),
        "img_widths": image_widths.numpy().astype(np.int64, copy=False),
    }
    output = session.run(None, feeds)[0]

    if output.ndim != 3:
        raise ValueError(f"ONNX 输出应为三维 [T,B,C]，实际形状为 {output.shape}")
    if output.shape[1] != images.shape[0]:
        raise ValueError(
            "ONNX 输出批量维不匹配: "
            f"输出形状 {output.shape}，输入批量 {images.shape[0]}"
        )
    if output.shape[2] != converter.num_classes:
        raise ValueError(
            "ONNX 输出类别数与字典不匹配: "
            f"模型 {output.shape[2]}，字典 {converter.num_classes}"
        )

    return converter.decode(torch.from_numpy(output))


def write_error_report(error_cases: list[ErrorCase], output_path: Path) -> None:
    """
    将所有错误样本写入制表符分隔的明细文件。

    Args:
        error_cases: 评估过程中收集的错误样本。
        output_path: TSV 错误报告输出路径。

    Returns:
        无返回值。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "样本序号",
                "图片路径",
                "处理后宽度",
                "标注",
                "预测",
                "标注长度",
                "预测长度",
                "编辑距离",
                "样本CER",
                "错误类型",
            ]
        )
        for case in error_cases:
            writer.writerow(
                [
                    case.sample_index,
                    case.image_path,
                    case.image_width,
                    case.expected,
                    case.predicted,
                    len(case.expected),
                    len(case.predicted),
                    case.edit_distance,
                    f"{case.sample_cer:.8f}",
                    case.error_type,
                ]
            )


def write_summary_report(summary: dict[str, object], output_path: Path) -> None:
    """
    将评估汇总指标写入 JSON 文件。

    Args:
        summary: 包含数据路径、准确率、CER 及专项统计的汇总字典。
        output_path: JSON 汇总报告输出路径。

    Returns:
        无返回值。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def evaluate_onnx(
    model_path: Path,
    annotation_path: Path,
    keys_path: Path,
    error_report_path: Path,
    summary_report_path: Path,
    batch_size: int,
    num_workers: int,
    max_samples: int | None = None,
) -> dict[str, object]:
    """
    使用 ONNX 模型评估标注数据集并生成错误报告。

    Args:
        model_path: ONNX 识别模型路径。
        annotation_path: 待检查的训练标注文件路径。
        keys_path: 与模型训练一致的字符字典路径。
        error_report_path: 错误样本 TSV 报告路径。
        summary_report_path: 汇总指标 JSON 报告路径。
        batch_size: ONNX 推理批量大小。
        num_workers: PyTorch DataLoader 的工作进程数。
        max_samples: 最大评估样本数；为 ``None`` 时评估全部标注。

    Returns:
        包含整体指标和两字符专项指标的汇总字典。

    Raises:
        FileNotFoundError: 模型、字典、标注或图片不存在时抛出。
        ValueError: 配置、输入输出契约或标注内容不合法时抛出。
    """
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE 必须大于 0")
    if num_workers < 0:
        raise ValueError("NUM_WORKERS 不能小于 0")
    if not keys_path.is_file():
        raise FileNotFoundError(f"字符字典不存在: {keys_path}")

    dataset = load_evaluation_dataset(annotation_path, max_samples)
    converter = LabelConverter(str(keys_path))
    session = create_onnx_session(model_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=dynamic_padding_collate_fn,
    )

    total_samples = 0
    correct_samples = 0
    total_edit_distance = 0
    total_characters = 0
    short_output_count = 0
    missing_last_count = 0
    two_char_total = 0
    two_char_correct = 0
    two_char_short_count = 0
    two_char_missing_last_count = 0
    error_cases: list[ErrorCase] = []

    start_time = time.perf_counter()
    print(f"ONNX 模型: {model_path}")
    print(f"训练标注: {annotation_path}")
    print(f"评估样本: {len(dataset)}")
    print(f"批量大小: {batch_size}")

    progress = tqdm(loader, desc="评估训练集", unit="batch")
    for images, labels, image_widths in progress:
        predictions = infer_batch(session, converter, images, image_widths)
        batch_count = len(labels)
        if len(predictions) != batch_count:
            raise ValueError(
                f"预测数量 {len(predictions)} 与批次样本数 {batch_count} 不一致"
            )

        batch_paths = dataset.image_paths[
            total_samples: total_samples + batch_count
        ]
        for local_index, (image_path, image_width, expected, predicted) in enumerate(
            zip(batch_paths, image_widths.tolist(), labels, predictions),
            start=1,
        ):
            sample_index = total_samples + local_index
            edit_distance = levenshtein_distance(expected, predicted)
            total_edit_distance += edit_distance
            total_characters += len(expected)

            if predicted == expected:
                correct_samples += 1

            if len(predicted) < len(expected):
                short_output_count += 1

            is_missing_last = len(expected) >= 2 and predicted == expected[:-1]
            if is_missing_last:
                missing_last_count += 1

            if len(expected) == 2:
                two_char_total += 1
                if predicted == expected:
                    two_char_correct += 1
                if len(predicted) < len(expected):
                    two_char_short_count += 1
                if is_missing_last:
                    two_char_missing_last_count += 1

            if edit_distance > 0:
                relative_path = os.path.relpath(image_path, PROJECT_ROOT)
                error_cases.append(
                    ErrorCase(
                        sample_index=sample_index,
                        image_path=relative_path,
                        image_width=image_width,
                        expected=expected,
                        predicted=predicted,
                        edit_distance=edit_distance,
                        sample_cer=edit_distance / len(expected),
                        error_type=classify_error(expected, predicted),
                    )
                )

        total_samples += batch_count
        progress.set_postfix(
            accuracy=f"{correct_samples / total_samples:.3%}",
            errors=len(error_cases),
            missing_last=missing_last_count,
        )

    elapsed_seconds = time.perf_counter() - start_time
    accuracy = correct_samples / total_samples
    cer = total_edit_distance / total_characters
    two_char_accuracy = (
        two_char_correct / two_char_total if two_char_total else 0.0
    )

    summary: dict[str, object] = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "model_path": os.path.relpath(model_path, PROJECT_ROOT),
        "annotation_path": os.path.relpath(annotation_path, PROJECT_ROOT),
        "keys_path": os.path.relpath(keys_path, PROJECT_ROOT),
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "error_samples": len(error_cases),
        "accuracy": accuracy,
        "total_characters": total_characters,
        "total_edit_distance": total_edit_distance,
        "cer": cer,
        "short_output_count": short_output_count,
        "missing_last_count": missing_last_count,
        "two_char_total": two_char_total,
        "two_char_correct": two_char_correct,
        "two_char_accuracy": two_char_accuracy,
        "two_char_short_output_count": two_char_short_count,
        "two_char_missing_last_count": two_char_missing_last_count,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_samples / elapsed_seconds,
    }

    write_error_report(error_cases, error_report_path)
    write_summary_report(summary, summary_report_path)

    print("\n" + "=" * 72)
    print("ONNX 训练集评估完成")
    print(f"总样本数             : {total_samples}")
    print(f"全串正确数           : {correct_samples}")
    print(f"全串准确率           : {accuracy:.4%}")
    print(f"字符错误率 CER       : {cer:.4%}")
    print(f"短输出数量           : {short_output_count}")
    print(f"恰好漏末字数量       : {missing_last_count}")
    print(f"两字符样本数         : {two_char_total}")
    print(f"两字符准确率         : {two_char_accuracy:.4%}")
    print(f"两字符短输出数量     : {two_char_short_count}")
    print(f"两字符恰好漏末字数量 : {two_char_missing_last_count}")
    print(f"耗时                 : {elapsed_seconds:.2f} 秒")
    print(f"吞吐量               : {total_samples / elapsed_seconds:.2f} 张/秒")

    if error_cases:
        ranked_errors = sorted(
            error_cases,
            key=lambda case: (case.sample_cer, case.edit_distance),
            reverse=True,
        )
        print(f"\n前 {min(TOP_ERROR_LIMIT, len(ranked_errors))} 个错误样本:")
        for case in ranked_errors[:TOP_ERROR_LIMIT]:
            print(
                f"[{case.error_type}] {case.image_path} "
                f"标注={case.expected!r} 预测={case.predicted!r} "
                f"宽度={case.image_width} CER={case.sample_cer:.2%}"
            )

    print(f"\n错误明细: {error_report_path}")
    print(f"汇总指标: {summary_report_path}")
    print("=" * 72)

    return summary


def main() -> None:
    """
    使用文件顶部常量执行完整训练集 ONNX 评估。

    Returns:
        无返回值。
    """
    evaluate_onnx(
        model_path=ONNX_MODEL_PATH,
        annotation_path=TRAIN_ANNOTATION_PATH,
        keys_path=KEYS_PATH,
        error_report_path=ERROR_REPORT_PATH,
        summary_report_path=SUMMARY_REPORT_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
    )


if __name__ == "__main__":
    main()

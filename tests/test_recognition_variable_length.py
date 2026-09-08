"""识别模型变长序列行为的回归测试。"""

import torch

from recognition.components.dataset import (
    CNN_RIGHT_CONTEXT,
    dynamic_padding_collate_fn,
)
from recognition.finetune import calculate_ctc_input_lengths
from recognition.models.model import CRNN, calculate_cnn_output_widths


def test_cnn_output_widths_match_mobile_feature_extractor() -> None:
    """验证统一宽度公式与 MobileFeatureExtractor 的实际输出一致。"""
    model = CRNN(img_height=32, nc=1, n_class=8, nh=8).eval()

    for pixel_width in range(48, 68):
        image = torch.ones(1, 1, 32, pixel_width)
        with torch.no_grad():
            actual_width = model.cnn(image).size(3)

        calculated_width = calculate_cnn_output_widths(
            torch.tensor([pixel_width]),
        ).item()
        assert calculated_width == actual_width


def test_dynamic_padding_reserves_right_cnn_context() -> None:
    """验证批量画布为最宽图片保留固定的右侧白色上下文。"""
    short_image = torch.zeros(1, 32, 57)
    long_image = torch.zeros(1, 32, 101)

    padded_images, labels, image_widths = dynamic_padding_collate_fn(
        [(short_image, "短"), (long_image, "长")],
    )

    expected_width = ((101 + CNN_RIGHT_CONTEXT + 7) // 8) * 8
    assert padded_images.shape == (2, 1, 32, expected_width)
    assert labels == ("短", "长")
    assert image_widths.tolist() == [57, 101]
    assert torch.all(padded_images[0, :, :, 57:] == 1.0)
    assert torch.all(padded_images[1, :, :, 101:] == 1.0)


def test_finetune_ctc_lengths_use_the_shared_width_formula() -> None:
    """验证微调 CTC 长度使用统一公式并受输出时间维限制。"""
    image_widths = torch.tensor([57, 59, 63], dtype=torch.int32)

    input_lengths = calculate_ctc_input_lengths(
        image_widths,
        max_seq_len=15,
    )

    assert input_lengths.dtype == torch.int32
    assert input_lengths.tolist() == [14, 15, 15]


def test_valid_logits_do_not_depend_on_batch_canvas_width() -> None:
    """验证有效时间步输出不随右侧批量画布宽度变化。"""
    torch.manual_seed(0)
    model = CRNN(img_height=32, nc=1, n_class=8, nh=8).eval()

    valid_pixel_width = 57
    narrow_canvas = torch.ones(1, 1, 32, 96)
    wide_canvas = torch.ones(1, 1, 32, 376)
    content = torch.randn(1, 1, 32, valid_pixel_width)
    narrow_canvas[:, :, :, :valid_pixel_width] = content
    wide_canvas[:, :, :, :valid_pixel_width] = content
    image_widths = torch.tensor([valid_pixel_width], dtype=torch.int32)

    with torch.no_grad():
        narrow_output = model(narrow_canvas, img_widths=image_widths)
        wide_output = model(wide_canvas, img_widths=image_widths)

    valid_seq_len = calculate_cnn_output_widths(image_widths).item()
    torch.testing.assert_close(
        narrow_output[:valid_seq_len],
        wide_output[:valid_seq_len],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.all(narrow_output[valid_seq_len:].argmax(dim=2) == 0)
    assert torch.all(wide_output[valid_seq_len:].argmax(dim=2) == 0)

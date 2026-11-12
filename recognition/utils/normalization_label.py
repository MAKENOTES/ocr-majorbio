import os
import re


def _is_cjk_char(c: str) -> bool:
    """判断单个字符是否属于中文语境字符。

    用于空格规一化时识别"与中文相邻的排版空格"。

    参数:
        c (str): 单个字符。

    返回值:
        bool: 若字符属于 CJK 统一表意文字、CJK 标点、全角形式
            或中文语境常用通用标点，返回 True，否则 False。
    """
    return (
        '一' <= c <= '鿿'      # CJK 统一表意文字（汉字）
        or '　' <= c <= '〿'   # CJK 符号和标点（含全角空格、。、《》等）
        or '＀' <= c <= '￯'   # 全角 ASCII 变体（，：；＜～等）
        or c in '—‘’“”·•…'            # 中文语境常用的通用标点
    )


def normalize_spaces(label: str) -> str:
    """规一化标签/识别文本中的空格。

    约定：英文内部的空格（前后均为非中文字符）是有效字符，必须保留；
    与中文（汉字、中文标点、全角字符）相邻的空格属于排版间距，
    不是有效字符，一律删除；连续空格夹在非中文字符之间时折叠为一个。

    典型行为:
        "Sodium Phosphate Buffer" -> "Sodium Phosphate Buffer"（保留）
        "第 1 页 共 3 页"          -> "第1页共3页"（删除）
        "二代 next seq 或"         -> "二代next seq或"（边界删除、词内保留）
        "a  b"                    -> "a b"（折叠）

    参数:
        label (str): 原始标签文本或识别输出文本。

    返回值:
        str: 规一化后的文本。入参为空字符串时原样返回。
    """
    if not label:
        return label

    result = []
    i, n = 0, len(label)
    while i < n:
        c = label[i]
        if c == ' ':
            # 找到连续空格段的结束位置
            j = i
            while j < n and label[j] == ' ':
                j += 1
            left = result[-1] if result else None    # 串首视为无左字符
            right = label[j] if j < n else None      # 串尾视为无右字符
            # 仅当左右字符都存在且都不是中文语境字符时保留一个空格
            if (left is not None and right is not None
                    and not _is_cjk_char(left) and not _is_cjk_char(right)):
                result.append(' ')
            i = j
        else:
            result.append(c)
            i += 1
    return ''.join(result)


def replace_english_comma_in_labels(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    processed_lines = 0
    modified_lines = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            original_line = line.strip('\n\r')
            if not original_line:
                fout.write('\n')
                continue

            # 定义一个替换函数，只处理双引号内部的匹配项
            def replacer(match):
                inner_text = match.group(1)
                # 如果内部包含英文逗号，就替换成中文逗号
                if ',' in inner_text:
                    nonlocal modified_lines
                    modified_lines += 1
                    inner_text = inner_text.replace(',', '，')
                return f'"{inner_text}"'

            # 用正则精准匹配双引号内的内容，并调用 replacer 替换
            # 这样前半部分的 `图片路径, ` (带空格) 会被完美保留，一字不差
            new_line = re.sub(r'"(.*?)"', replacer, original_line)

            fout.write(f"{new_line}\n")
            processed_lines += 1

    print("=" * 40)
    print(f"✅ 处理完成！")
    print(f"总计处理行数: {processed_lines}")
    print(f"包含英文逗号并成功替换的行数: {modified_lines}")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    # 请根据实际情况修改路径
    INPUT_TXT = "recognition/data/val_annotation.txt"
    OUTPUT_TXT = "recognition/data/val_annotation_fixed.txt"

    replace_english_comma_in_labels(INPUT_TXT, OUTPUT_TXT)
"""
用于CTC Loss的标签-文本转换器。
"""
import torch

class LabelConverter:
    """
    在文本标签和用于CTC Loss的整数序列之间进行转换。
    """
    def __init__(self, keys_path: str):
        """
        初始化转换器。
        参数:
            keys_path (str): 包含所有字符的keys.txt文件路径。
        """
        with open(keys_path, 'r', encoding='utf-8') as f:
            self.keys = f.read().strip()
        
        # 0被CTC的'blank'标记保留
        self.char2idx = {'[blank]': 0}
        for i, char in enumerate(self.keys):
            # 字符索引从1开始
            self.char2idx[char] = i + 1
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.num_classes = len(self.idx2char)

    def encode(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将文本字符串列表编码为用于CTC Loss的整数张量。
        参数:
            text (list[str]): 文本字符串列表。
        返回:
            tuple[torch.Tensor, torch.Tensor]: 一个元组，包含:
                - 所有文本连接在一起的整数序列张量。
                - 每个文本序列长度的张量。
        """
        encoded_texts = []
        lengths = []
        for s in text:
            length = 0
            encoded_s = []
            for char in s:
                if char in self.char2idx:
                    encoded_s.append(self.char2idx[char])
                    length += 1
            encoded_texts.extend(encoded_s)
            lengths.append(length)
        
        return torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, preds: torch.Tensor) -> list[str]:
        """
        将模型预测的整数序列解码为文本字符串。
        参数:
            preds (torch.Tensor): 模型的预测张量，
                                  形状通常为 (batch_size, seq_len, num_classes)。
        返回:
            list[str]: 解码后的文本字符串列表。
        """
        # 在每个时间步获取概率最高的字符
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        preds_size = torch.tensor([preds.size(0)], dtype=torch.long)
        
        decoded_texts = []
        sim_preds = []
        
        # 移除连续的重复字符和'blank'标记
        for i in range(preds.size(0)):
            if preds[i] != 0 and (i == 0 or preds[i] != preds[i-1]):
                sim_preds.append(self.idx2char[preds[i].item()])
        
        decoded_texts.append("".join(sim_preds))
        return decoded_texts

import torch
import torch.nn as nn


class GammaPredictor(nn.Module):
    """预测接受长度的三层MLP模型 - 三分类版本"""

    def __init__(self, hidden_dim=16384, embedding_dim=4096, dropout_rate=0.2):
        """
        初始化三层MLP模型

        参数:
        hidden_dim: hidden state的维度，默认16384
        embedding_dim: embedding的维度，默认4096
        dropout_rate: dropout比率，用于防止过拟合
        """
        super(GammaPredictor, self).__init__()

        # 特征维度
        self.embedding_dim = embedding_dim
        input_dim = hidden_dim
        if embedding_dim is not None:
            input_dim += embedding_dim

        # 设计三层MLP网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3)  # 三分类输出: 0,1-6,7+
        )

    def load_pretrained(self, model_path):
        """加载预训练模型"""
        try:
            # 尝试加载完整的模型字典
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 如果是完整的保存格式
                    self.load_state_dict(checkpoint['model_state_dict'])
                    # 更新维度信息（如果存在）
                    if 'hidden_dim' in checkpoint:
                        self.hidden_dim = checkpoint['hidden_dim']
                    if 'embedding_dim' in checkpoint:
                        self.embedding_dim = checkpoint['embedding_dim']
                else:
                    # 如果是直接保存的state_dict
                    self.load_state_dict(checkpoint)
            else:
                # 如果是直接保存的state_dict
                self.load_state_dict(checkpoint)
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise

        self.eval()

    def forward(self, hidden_state, embedding=None):
        """
        预测下一轮的gamma值

        参数:
        hidden_state: target model最后一个token的hidden state [batch_size, hidden_dim] 或 [batch_size, seq_len, hidden_dim]
        embedding: target model最后一个token的embedding (可选) [batch_size, embedding_dim] 或 [batch_size, seq_len, embedding_dim]

        返回:
        gamma: 预测的下一轮gamma值 (整数)
        """
        # 转换输入类型为float32
        hidden_state = hidden_state.float()
        if embedding is not None:
            embedding = embedding.float()

        # 处理hidden_state的维度
        if len(hidden_state.shape) == 4:  # [batch, seq_len, num_layers, hidden_dim]
            hidden_state = hidden_state[:, -1, -1, :]
        elif len(hidden_state.shape) == 3:  # [batch, num_layers, hidden_dim]
            hidden_state = hidden_state[:, -1, :]
        elif len(hidden_state.shape) > 2:
            hidden_state = hidden_state.squeeze(1)  # 移除多余的维度

        # 处理embedding的维度
        if embedding is not None:
            if len(embedding.shape) == 4:  # [batch, seq_len, num_layers, embedding_dim]
                embedding = embedding[:, -1, -1, :]
            elif len(embedding.shape) == 3:  # [batch, num_layers, embedding_dim]
                embedding = embedding[:, -1, :]
            elif len(embedding.shape) > 2:
                embedding = embedding.squeeze(1)  # 移除多余的维度

        # 合并特征
        if embedding is not None and self.embedding_dim is not None:
            x = torch.cat([hidden_state, embedding], dim=1)
        else:
            x = hidden_state

        # 前向传播
        with torch.no_grad():
            outputs = self.layers(x)
            if len(outputs.shape) > 2:
                outputs = outputs.squeeze(1)  # 移除多余的维度

            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            # 根据预测类别确定gamma值
            if pred_class == 0:
                return 1  # 类别0对应gamma=1 表示进入混合状态1
            elif pred_class == 1:
                return 4  # 类别1对应gamma=4（1-6的中间值） 表示进入混合状态2 利用小模型的confidence
            else:
                return 7  # 类别2对应gamma=7（7及以上） 表示进入混合状态3
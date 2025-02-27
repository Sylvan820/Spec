import torch
import ipdb
from .util import norm_logits, sample, make_mask    


class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, topk, temp) -> torch.Tensor:
        """ forward the model with kvcache

        Args:
            input_ids (torch.Tensor): the input ids

        Returns:
            torch.Tensor: the logits
        """
        # 如果kvcache为空，则生成新的kvcache
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            # 计算概率历史
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            # 归一化概率历史
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], temp, topk,
                                                          self._top_p)
            # 更新kvcache
            self._past_key_values = outputs.past_key_values
            # 返回最后一个token的概率
            last_q = self._prob_history[:, -1, :]
        # 如果kvcache不为空，则继续生成
        else:
            # 获取kvcache的长度
            cached_len = self._past_key_values[0][0].shape[2]
            # 获取最后一个token的输入id
            last_input_id = input_ids[:, cached_len:]
            # 如果最后一个token的输入id为1维，则扩展为2维
            if last_input_id.dim() == 1:
                # 扩展为2维
                last_input_id = torch.unsqueeze(last_input_id, 0)
            # 生成下一个token
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            # 计算概率历史
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            # 如果概率历史为2维，则扩展为3维
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            # 归一化概率历史
            for i in range(not_cached_q.shape[-2]):
                # 归一化概率历史
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temp, topk, self._top_p)
                # 连接概率历史
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            # 更新kvcache
            self._past_key_values = outputs.past_key_values
            # 返回最后一个token的概率
            last_q = not_cached_q[:, -1, :]
            ipdb.set_trace()

        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor,
                               gamma: int) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix
        # 生成gamma次
        for _ in range(gamma):
            # 计算下一个token的概率
            q_1 = self._forward_with_kvcache(x,0,0)
            # 采样下一个token
            next_tok = sample(q_1)
            # 连接下一个token
            x = torch.cat((x, next_tok), dim=1)
        # 返回prefix+generated tokens
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output

    def branch(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_branches(input, gamma)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """ rollback the kvcache

        Args:
            end_pos (int): the end position
        """
        # 裁剪kvcache
        past_key_values_trimmed = []
        # 断言kvcache不为空
        assert self._past_key_values
        # 裁剪kvcache
        for kv in self._past_key_values:
            # 获取kvcache
            k, v = kv
            # 裁剪kvcache
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            # 更新kvcache
            kv_trimmed = (k, v)
            # 添加到裁剪后的kvcache
            past_key_values_trimmed.append(kv_trimmed)
        # 更新kvcache
        self._past_key_values = past_key_values_trimmed
        # 裁剪概率历史
        self._prob_history = self._prob_history[:, :end_pos, :]

    @torch.no_grad()
    def _generate_branches(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        """并行生成多个分支的输出序列

        Args:
            input (torch.Tensor): 输入序列
            gamma (int): 每个分支生成的token数量

        Returns:
            torch.Tensor: 所有分支的生成结果，形状为 [batch_size, num_branches, seq_len]
        """
        device = input.device
        prefix_len = input.shape[1]
        
        # 确定分支数量
        num_branches = self._top_k if self._top_k > 0 else 3
        
        # 生成第一个token的概率分布
        q_1 = self._forward_with_kvcache(input, 3, 1)
        
        # 采样多个候选token作为不同分支的起始token
        first_tokens = sample(q_1, num_samples=num_branches)
        
        # 创建包含所有分支的输入序列
        # 每个分支的输入序列都是原始输入加上对应的首个token
        branch_inputs = []
        for i in range(num_branches):
            branch_input = torch.cat((input, first_tokens[:, i:i+1]), dim=1)
            branch_inputs.append(branch_input)
        
        # 将所有分支的输入拼接成一个批次
        # 形状从 [num_branches, batch_size, prefix_len+1] 变为 [batch_size*num_branches, prefix_len+1]
        batched_inputs = torch.cat(branch_inputs, dim=0)
        
        # 保存当前的KV cache和概率历史，以便后续恢复
        original_kv_cache = self._past_key_values
        original_prob_history = self._prob_history
    
        
        # 并行生成剩余的gamma-1个token
        for step in range(gamma - 1):
            # 计算当前序列长度
            current_seq_len = batched_inputs.shape[1]
            # 计算新生成token的数量
            new_tokens_len = current_seq_len - prefix_len
            
            # 为每个分支创建位置ID
            position_ids = torch.arange(
                prefix_len, current_seq_len, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0).expand(batched_inputs.shape[0], -1)
            
            # 创建因果掩码
            causal_mask = make_mask(
                past_len=prefix_len,
                seq_len=new_tokens_len,
                device=device
            )
            
            # 使用掩码和位置ID前向传播模型
            outputs = self._model(
                batched_inputs,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=self._past_key_values,
                use_cache=True
            )
            
            # 计算概率历史
            logits = outputs.logits[:, -1, :self.vocab_size]  # 只取最后一个token的logits
            
            # 归一化概率
            probs = norm_logits(logits, 0, 0, self._top_p)
            
            # 对每个分支采样下一个token
            next_tokens = sample(probs)
            
            # 将新token添加到输入序列
            batched_inputs = torch.cat((batched_inputs, next_tokens), dim=1)
            
            # 更新KV cache
            self._past_key_values = outputs.past_key_values
        
        # 将批次结果重新组织为分支形式
        # 形状从 [batch_size*num_branches, prefix_len+gamma] 变为 [batch_size, num_branches, prefix_len+gamma]
        branch_results = []
        for i in range(num_branches):
            start_idx = i * input.shape[0]
            end_idx = (i + 1) * input.shape[0]
            branch_results.append(batched_inputs[start_idx:end_idx])
        
        # 恢复原始的KV cache和概率历史
        self._past_key_values = original_kv_cache
        self._prob_history = original_prob_history
        
        # 返回包含所有分支结果的张量
        return torch.stack(branch_results, dim=1)
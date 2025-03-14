import torch
from .util import norm_logits, sample
import ipdb


class BranchModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self.temp_cache = None
        self.temp_prob = None

        self.rollback_mark = 0


    def _forward_with_kvcache(self, input_ids: torch.Tensor, topk, temp) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], temp, topk,
                                                          self._top_p)[0]
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

            not_cached_q = outputs.logits[:, :, :self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temp, topk, self._top_p)[0]

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

        return last_q

    def _branch_forward(
            self,
            input_ids: torch.Tensor,
            kv_cache: torch.Tensor,
            prob_history: torch.Tensor
    ) -> torch.Tensor:

        outputs = self._model(input_ids, past_key_values=kv_cache, use_cache=True)

        not_cached_q = outputs.logits[:, :, :self.vocab_size]

        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)

        for i in range(not_cached_q.shape[-2]):
            if i < not_cached_q.shape[-2] - 1:
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)[
                    0]
            else:
                not_cached_q[:, i, :], confidence_s = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k,
                                                                  self._top_p)

        prob_history = torch.cat([prob_history, not_cached_q], dim=1)
        # ipdb.set_trace()
        last_q = not_cached_q[:, -1, :]
        next_cache = outputs.past_key_values
        # ipdb.set_trace()

        return last_q, next_cache, prob_history, confidence_s

    def _generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        for _ in range(gamma):
            q = self._forward_with_kvcache(input_ids, self._past_key_values, self._temperature)
            next_tok = sample(q)
            input_ids = torch.cat((input_ids, next_tok), dim=1)
        return input_ids

    def _branch_generate(
            self,
            input_ids: torch.Tensor,
            gamma: int,
            branches: int = 1
    ) -> torch.Tensor:
        if self._temperature == 0:
            output_logits = self._forward_with_kvcache(input_ids, branches, 1)
        else:
            output_logits = self._forward_with_kvcache(input_ids, self._top_k, self._temperature)

        next_tok = sample(output_logits, branches)
        q_next = next_tok.transpose(0, 1)

        output_extended = input_ids.repeat(branches, 1)
        output_extended = torch.cat([output_extended, q_next], dim=1)
        cache_next = [
            (layer[0].repeat(branches, 1, 1, 1), layer[1].repeat(branches, 1, 1, 1))
            for layer in self._past_key_values
        ]
        prob_history = self._prob_history.repeat(branches, 1, 1)

        marked_indices = [False] * branches  # 用于标记每个分支是否已标记
        marked_values = [None] * branches  # 存储标记的值

        # Initialize self.confidence_branch as a list to store confidence values for each branch
        self.confidence_branch = [None] * branches

        for i in range(gamma - 1):
            logits, cache_next, prob_history, confidence_s = self._branch_forward(q_next, cache_next, prob_history)
            q_next = sample(logits)
            output_extended = torch.cat([output_extended, q_next], dim=1)

            # 检查每个分支的 confidence 值
            for j in range(branches):
                if torch.max(confidence_s[j], dim=-1)[0] <= 0.3 and not marked_indices[j]:  # 只在第一次出现时标记
                    marked_indices[j] = True
                    marked_values[j] = (j, i)  # 标记当前分支的索引和对应的值

        self.temp_cache = cache_next
        self.temp_prob = prob_history
        for b in range(branches):
            if marked_values[b] is None:
                marked_values[b] = (b, gamma - 1)
        # 可以在这里使用 marked_values 进行后续处理
        return output_extended, marked_values

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, branches: int) -> torch.Tensor:
        if branches > 1:
            output, marked_values = self._branch_generate(input, gamma, branches)
        else:
            output = self._generate(input, gamma)
        return output, marked_values

    @torch.no_grad()
    def rollback(self, end_pos: int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

    def select_branch(self, branch_id: int, marked_values, prefix_len):
        self._past_key_values = [
            (layer[0][branch_id:branch_id + 1], layer[1][branch_id:branch_id + 1])
            for layer in self.temp_cache
        ]
        self._prob_history = self.temp_prob[branch_id:branch_id + 1]
        self.temp_cache = None
        self.temp_prob = None
        if marked_values < 9:
            self.rollback(prefix_len + marked_values)

    def select_branch(self, branch_id: int, marked_values, prefix_len):
        self._past_key_values = [
            (layer[0][branch_id:branch_id + 1], layer[1][branch_id:branch_id + 1])
            for layer in self.temp_cache
        ]
        self._prob_history = self.temp_prob[branch_id:branch_id + 1]
        self.temp_cache = None
        self.temp_prob = None
        if marked_values < 9:
            self.rollback(prefix_len + marked_values)
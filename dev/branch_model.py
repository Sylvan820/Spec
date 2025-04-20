import torch
from .util import norm_logits, sample_greedy, calculate_processed_entropy
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
        self.invalid_logits = None
        self.trace_mode = False
        self.mode = 0
        self.first_logit = None
        self._last_actual_gamma = None
        self.unconfident_logits = None

    def _forward_with_kvcache(self, input_ids: torch.Tensor, temperature=None) -> torch.Tensor:
        temperature = temperature if temperature is not None else self._temperature
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], temperature, self._top_k,
                                                          self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            try:
                outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            except:
                ipdb.set_trace()
            not_cached_q = outputs.logits[:, :, :self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temperature, self._top_k, self._top_p)

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

        # confidences = torch.softmax(not_cached_q[:, -1, :], dim=-1)
        not_cached_q[:, -1, :] = norm_logits(not_cached_q[:, -1, :], self._temperature, self._top_k, self._top_p)

        prob_history = torch.cat([prob_history, not_cached_q], dim=1)
        last_q = not_cached_q[:, -1, :]

        next_cache = outputs.past_key_values

        return last_q, next_cache, prob_history

    def _generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:

        x = input_ids
        actual_gamma = gamma

        for i in range(gamma):
            q= self._forward_with_kvcache(x)

            confidence = torch.max(q, dim=-1).values
            if confidence < 0.3:
                actual_gamma = i+1  # Update actual gamma
                self._last_actual_gamma = actual_gamma
                next_tok = sample_greedy(q)
                x = torch.cat((x, next_tok), dim=1)

                return x

            next_tok = sample_greedy(q)
            x = torch.cat((x, next_tok), dim=1)

        self._last_actual_gamma = actual_gamma
        return x

    # def _generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
    #
    #     x = input_ids
    #     actual_gamma = gamma
    #     max_gamma = 7
    #
    #     # print(gamma)
    #
    #     if gamma != 1:
    #         for i in range(max_gamma):
    #             q = self._forward_with_kvcache(x)
    #             # Check confidence when gamma is
    #             # Get the maximum probability as confidence
    #             if gamma == 4 and i != 0:
    #                 confidence = torch.max(q, dim=-1).values
    #                 # print('i:',i)
    #             # confidence = self.calculate_processed_entropy(q)
    #             # print(confidence)
    #             # ipdb.set_trace()
    #                 if confidence < 0.3:
    #                     # print('hahahha')
    #                     actual_gamma = i + 1  # Update actual gamma
    #                     self._last_actual_gamma = actual_gamma
    #
    #                     self.unconfident_logits = q
    #                     next_tok = sample_greedy(q)
    #                     x = torch.cat((x, next_tok), dim=1)
    #                     return x
    #
    #             next_tok = sample_greedy(q)
    #             x = torch.cat((x, next_tok), dim=1)
    #             actual_gamma = max_gamma
    #     else:
    #         q = self._forward_with_kvcache(x)
    #         self.unconfident_logits = q
    #         next_tok = sample_greedy(q)
    #         x = torch.cat((x, next_tok), dim=1)
    #
    #     self._last_actual_gamma = actual_gamma
    #     return x


    # def _generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
    #
    #     x = input_ids
    #
    #     for _ in range(gamma):
    #         q = self._forward_with_kvcache(x)
    #         next_tok = sample_greedy(q)
    #         x = torch.cat((x, next_tok), dim=1)
    #     return x


    def _branch_generate(
            self,
            input_ids: torch.Tensor,
            gamma: int,
            branches: int = 1
    ) -> torch.Tensor:


        output_logits = self._forward_with_kvcache(input_ids, temperature=1) if not self.trace_mode else \
        self.invalid_logits[0].unsqueeze(0)

        next_tok = sample_greedy(output_logits, branches)
        q_next = next_tok.transpose(0, 1)

        output_extended = input_ids.repeat(branches, 1)
        # ipdb.set_trace()
        output_extended = torch.cat([output_extended, q_next], dim=1)

        cache_next = [
            (layer[0].repeat(branches, 1, 1, 1), layer[1].repeat(branches, 1, 1, 1))
            for layer in self._past_key_values
        ]
        prob_history = self._prob_history.repeat(branches, 1, 1)

        invalid_indices = [None] * branches
        self.invalid_logits = [None] * branches
        self.first_logit = [None] * branches

        for i in range(gamma - 1):
            logits, cache_next, prob_history = self._branch_forward(q_next, cache_next, prob_history)
            # max_confidence = torch.max(confidences, dim=-1).values
            # entropy = calculate_processed_entropy(confidences)
            # print(confidences[0].sum())
            # print(entropy)
            q_next = sample_greedy(logits)
            output_extended = torch.cat([output_extended, q_next], dim=1)

            # check if the confidence is low
            for b in range(branches):
                if i == 0:
                    self.first_logit[b] = logits[b]
                else:
                    confidence = torch.max(logits[b], dim=-1).values
                    if confidence <= 0 and invalid_indices[b] is None:
                    # if entropy[b] <= -10 and invalid_indices[b] is None:

                        invalid_indices[b] = i
                        self.invalid_logits[b] = logits[b]

        for b in range(branches):
            if invalid_indices[b] is None:
                invalid_indices[b] = gamma - 1
        self.temp_cache = cache_next
        self.temp_prob = prob_history

        return output_extended, invalid_indices

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, branches: int) -> torch.Tensor:
        if branches > 1:
            output, bad_indices = self._branch_generate(input, gamma, branches)
        else:
            output = self._generate(input, gamma)
            # print(output)
            bad_indices = [gamma - 1]
        return output, bad_indices

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

    def select_branch(self, branch_id: int):
        self._past_key_values = [
            (layer[0][branch_id:branch_id + 1], layer[1][branch_id:branch_id + 1])
            for layer in self.temp_cache
        ]
        self._prob_history = self.temp_prob[branch_id:branch_id + 1]
        self.invalid_logits[0] = self.invalid_logits[branch_id]
        self.first_logit[0] = self.first_logit[branch_id]
        self.temp_cache = None
        self.temp_prob = None

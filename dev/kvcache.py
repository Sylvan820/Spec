import torch
from .util import norm_logits, sample


class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._hidden_states = None
        self._last_four_layers = None  # 只存储最后四层
        self._embeddings = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            with torch.no_grad():
                if hasattr(self._model, 'get_input_embeddings'):
                    embedding_layer = self._model.get_input_embeddings()
                    self._embeddings = embedding_layer(input_ids)
                else:
                    self._embeddings = None

            outputs = self._model(input_ids, output_hidden_states=True)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k,
                                                          self._top_p)
            self._past_key_values = outputs.past_key_values
            # self._hidden_states = outputs.hidden_states[-1]
            # 只保存最后四层
            self._last_four_layers = [layer.clone() for layer in outputs.hidden_states[-4:]]
            last_q = self._prob_history[:, -1, :]
        else:
            cached_len = self._past_key_values[0][0].shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            with torch.no_grad():
                if hasattr(self._model, 'get_input_embeddings') and self._embeddings is not None:
                    embedding_layer = self._model.get_input_embeddings()
                    new_embeddings = embedding_layer(last_input_id)
                    self._embeddings = torch.cat([self._embeddings, new_embeddings], dim=1)

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True,
                                  output_hidden_states=True)

            not_cached_q = outputs.logits[:, :, :self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            # self._hidden_states = torch.cat([self._hidden_states, outputs.hidden_states[-1]], dim=1)

            # 更新最后四层
            new_last_four = outputs.hidden_states[-4:]
            if self._last_four_layers is None:
                self._last_four_layers = [layer.clone() for layer in new_last_four]
            else:
                self._last_four_layers = [torch.cat([prev, curr], dim=1)
                                          for prev, curr in zip(self._last_four_layers, new_last_four)]

            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

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

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output

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
        if self._hidden_states is not None:
            self._hidden_states = self._hidden_states[:, :end_pos, :]
        if self._last_four_layers is not None:
            self._last_four_layers = [layer[:, :end_pos, :] for layer in self._last_four_layers]
        if self._embeddings is not None:
            self._embeddings = self._embeddings[:, :end_pos, :]

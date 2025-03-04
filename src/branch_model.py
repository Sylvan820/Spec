import torch
from .util import norm_logits, sample


class BranchModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        
        self.prefix_len = 0

    def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
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
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q
    
    def _branch_forward(
        self, 
        input_ids : torch.Tensor,
        num_branches : int,
        pos : int,
        prefix_len : int,
        ) -> torch.Tensor:
        
        position_ids = torch.tensor(
                [pos] * num_branches,
                dtype=torch.long,
                device=self._model.device,
            ).unsqueeze(0)
        
        mask = self.make_mask(prefix_len, pos, num_branches)
        
        outputs = self._model(
            input_ids, 
            past_key_values=self._past_key_values, 
            position_ids=position_ids,
            attention_mask=mask,
            use_cache=True)
        
        not_cached_q = outputs.logits[:, :, :self.vocab_size]
        
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
        for i in range(not_cached_q.shape[-2]):   
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            
        self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
        self._past_key_values = outputs.past_key_values
        
        next_tokens = []
        for i in range(num_branches):
            branch_q = not_cached_q[:, -num_branches+i,  :]
            next_token = sample(branch_q)
            next_tokens.append(next_token)
        
        next_ids = torch.cat(next_tokens, dim=1)
        return next_ids, not_cached_q

    def _generate(self, input_ids : torch.Tensor, gamma : int) -> torch.Tensor:
        for _ in range(gamma):
            q = self._forward_with_kvcache(input_ids)
            next_tok = sample(q)
            input_ids = torch.cat((input_ids, next_tok), dim=1)
        return input_ids
    
    def _branch_generate(
        self, 
        input_ids : torch.Tensor, 
        gamma : int,
        branches: int,
        ) -> torch.Tensor:
        print('input_ids:', input_ids)
        output_logits = self._forward_with_kvcache(input_ids)
        sampled_toks = sample(output_logits, branches)
        next_toks = sampled_toks.squeeze(-1)
        
        pos = input_ids.shape[1]
        self.prefix_len = pos
        branch_toks = next_toks.transpose(0, 1)
        branch_history = self._prob_history.repeat(branches, 1, 1)
        
        for step in range(gamma - 1):
            pos += 1
            next_toks, next_history = self._branch_forward(next_toks, branches, pos, self.prefix_len)
            branch_toks = torch.cat((branch_toks, next_toks.transpose(0,1)), dim=1)
            branch_history = torch.cat((branch_history, next_history.transpose(0,1)), dim=1)
        
        branch_output = torch.cat((input_ids.repeat(branches,1), branch_toks), dim=1)
        return branch_output, branch_history

    @torch.no_grad()
    def generate(self, input_tensor : torch.Tensor, gamma : int, branches: int) -> torch.Tensor:
        if branches > 1:
            output= self._branch_generate(input_tensor, gamma, branches)
        else:
            output = self._generate(input_tensor, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
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

    def make_mask(self, prefix_len, pos, branches):
        epoch = pos - prefix_len
        seq_len = prefix_len + epoch * branches
        
        mask = torch.ones(
            (branches, seq_len), 
            dtype=self._model.dtype, 
            device=self._model.device)
        
        mask[:, :prefix_len] = 0
        for b in range(branches):
            for i in range(epoch):
                mask[b, prefix_len + i * branches + b] = 0
        
        mask = mask[None, None, :, :].expand(1, 1, -1, -1)
        return mask
    
    def select_branch(self, branch_id, gamma, branches):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            k_new = k[:, :, :self.prefix_len, :]
            v_new = v[:, :, :self.prefix_len, :]
            for i in range(gamma-1):
                k_new = torch.cat((k_new, k[:, :, [self.prefix_len + i * branches + branch_id], :]), dim=2)
                v_new = torch.cat((v_new, v[:, :, [self.prefix_len + i * branches + branch_id], :]), dim=2)
            kv_trimmed = (k_new, v_new)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        
        new_prob_history = self._prob_history[:, :self.prefix_len, :]
        for i in range(gamma-1):
            new_prob_history = torch.cat((
                new_prob_history, 
                self._prob_history[:, [self.prefix_len + i * branches + branch_id], :]), dim=1)
        self._prob_history = new_prob_history
    
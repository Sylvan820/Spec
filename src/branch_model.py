import torch
from .util import norm_logits, sample, sample_greedy
from typing import Tuple, List


class BranchModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self.branch_caches = None
        self.branch_probs = None
        self.invalid_logits = None
        self.trace_mode = False

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

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

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

        confidences = torch.softmax(not_cached_q[:, -1, :], dim=-1)
        not_cached_q[:, -1, :] = norm_logits(not_cached_q[:, -1, :], self._temperature, self._top_k, self._top_p)

        prob_history = torch.cat([prob_history, not_cached_q], dim=1)
        last_q = not_cached_q[:, -1, :]

        next_cache = outputs.past_key_values

        return last_q, next_cache, prob_history, confidences

    def _generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        for _ in range(gamma):
            q = self._forward_with_kvcache(input_ids)
            next_tok = sample(q)
            input_ids = torch.cat((input_ids, next_tok), dim=1)
        return input_ids

    def _branch_generate(
            self,
            input_ids: torch.Tensor,
            gamma: int,
            branches: int = 1
    ) -> Tuple[torch.Tensor, List[int]]:

        output_logits = (self._forward_with_kvcache(input_ids, temperature=1)
                        if not self.trace_mode
                        else self.invalid_logits[0].unsqueeze(0))

        next_tok = sample_greedy(output_logits, branches)
        q_next = next_tok.transpose(0, 1)

        output_ids = input_ids.repeat(branches, 1)
        output_ids = torch.cat([output_ids, q_next], dim=1)
        cache_next = [
            (layer[0].repeat(branches, 1, 1, 1), layer[1].repeat(branches, 1, 1, 1))
            for layer in self._past_key_values
        ]
        prob_history = self._prob_history.repeat(branches, 1, 1)

        invalid_indices = [None] * branches
        self.invalid_logits = [None] * branches
        self.first_logit = [None] * branches

        for i in range(gamma - 1):
            logits, cache_next, prob_history, confidences = self._branch_forward(q_next, cache_next, prob_history)
            max_confidence = torch.max(confidences, dim=-1).values
            q_next = sample(logits)
            output_ids = torch.cat([output_ids, q_next], dim=1)

            for b in range(branches):
                if i == 0:
                    self.first_logit[b] = logits[b]
                if max_confidence[b] <= 0.2 and invalid_indices[b] is None:
                    invalid_indices[b] = i
                    self.invalid_logits[b] = confidences[b]

        for b in range(branches):
            if invalid_indices[b] is None:
                invalid_indices[b] = gamma - 1
                
        self.branch_caches = cache_next
        self.branch_probs = prob_history

        return output_ids, invalid_indices
    
    def _branch_generate_early_stop(
        self,
        input_ids: torch.Tensor,
        gamma: int,
        branches: int = 1
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate outputs with early stopping using branch sampling.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            gamma (int): Total number of generation steps.
            branches (int): Number of branches to sample.

        Returns:
            Tuple[torch.Tensor, List[int]]: A tuple containing the generated outputs
                (with padded tokens for invalid branches) and a list of generation steps
                (indices) at which branches were marked as invalid.
        """
        # Get initial logits.
        # Use self.invalid_logits when in trace_mode; otherwise, perform forward pass with KV cache.
        output_logits = (self._forward_with_kvcache(input_ids, temperature=1)
                        if not self.trace_mode
                        else self.invalid_logits[0].unsqueeze(0))
        
        # Sample the next token using sample1 and transpose to make branch dimension first.
        next_tok = sample_greedy(output_logits, branches)
        q_next = next_tok.transpose(0, 1)
        
        # Repeat the input for each branch and append the newly sampled token.
        running_ids = [input_ids] * branches
        for i in range(branches):
            running_ids[i] = torch.cat([running_ids[i], q_next[i].unsqueeze(0)], dim=1)
        
        # Repeat the cached key-value pairs for each branch.
        cache_next = [
            (layer[0].repeat(branches, 1, 1, 1), layer[1].repeat(branches, 1, 1, 1))
            for layer in self._past_key_values
        ]
        
        # Repeat probability history for each branch.
        prob_history = self._prob_history.repeat(branches, 1, 1)
        
        invalid_indices = []       # List to record at which generation step each branch becomes invalid.
        self.invalid_logits = []   # Store logits for branches that fall below confidence threshold.
        self.branch_caches = []       # Temporary storage for padded cache of finished branches.
        self.branch_probs = []        # Temporary storage for padded probability history.
        output_ids = []        # Store the output IDs for each branch.

        # Loop over generation steps (excluding the initial token).
        for i in range(gamma - 1):
            # Perform a forward pass for the current branches.
            logits, cache_next, prob_history, confidences = self._branch_forward(q_next, cache_next, prob_history)
            bs = logits.shape[0]
            
            # Compute maximum confidence for each branch.
            max_confidence = torch.max(confidences, dim=-1).values
            
            # Sample next tokens for each branch.
            q_next = sample(logits)
            # Append new tokens to the output sequences.
            for b in range(bs):
                running_ids[b] = torch.cat([running_ids[b], q_next[b].unsqueeze(0)], dim=1)
            
            # Iterate over branches in reverse order to safely remove invalid branches.
            all_invalid = False
            for b in reversed(range(bs)):
                if max_confidence[b] <= 0.2:
                    invalid_indices.append(i)
                    self.invalid_logits.append(confidences[b])
                    if q_next.shape[0] == 1: # Only one branch left
                        all_invalid = True
                        padded = self.pad_branch(cache_next, prob_history, i, gamma)
                    else: # More than one branch left
                        padded, remained = self.pop_and_pad_branch(cache_next, prob_history, b, i, gamma)
                        q_next = torch.cat([q_next[:b], q_next[b + 1:]], dim=0)
                        cache_next, prob_history = remained
                    # Store the padded cache and probability history for the invalid branch.
                    self.branch_caches.append(padded[0])
                    self.branch_probs.append(padded[1])
                    output_ids.append(running_ids.pop(b))
            if all_invalid:
                break
        # Process any remaining valid branches.
        passed = branches - len(invalid_indices)
        if passed > 0:
            for i in reversed(range(passed)):
                self.branch_caches.append(self.cache_get_by_index(cache_next, i))
                self.branch_probs.append(prob_history[i].unsqueeze(0))
                invalid_indices.append(gamma - 1)  # Mark these branches as valid until the last step.
                self.invalid_logits.append(confidences[i])
                output_ids.append(running_ids[i].unsqueeze(0))
        
        # Concatenate all branch outputs (both invalid and remaining valid branches).
        output_ids = torch.cat(output_ids, dim=0)
        return output_ids, invalid_indices

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, branches: int) -> torch.Tensor:
        if branches > 1:
            output, bad_indices = self._branch_generate(input, gamma, branches)
        else:
            output = self._generate(input, gamma)
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
        if isinstance(self.branch_caches, list):
            self._past_key_values = self.branch_caches[branch_id]
            self._prob_history = self.branch_probs[branch_id]
        else:
            self._past_key_values = [
                (layer[0][branch_id:branch_id + 1], layer[1][branch_id:branch_id + 1])
                for layer in self.branch_caches
            ]
            self._prob_history = self.branch_probs[branch_id:branch_id + 1]
        self.invalid_logits[0] = self.invalid_logits[branch_id]
        self.branch_caches = None
        self.branch_probs = None
        
    def pop_and_pad_branch(self, cache_next, prob_history, b, i, gamma):
        """
        Remove an invalid branch and pad its cache, probability history, and output.

        Args:
            cache_next: List of cached key-value pairs for all branches.
            prob_history (torch.Tensor): Probability history for all branches.
            output_extended (torch.Tensor): Generated output tokens for all branches.
            b (int): Index of the branch to remove.
            i (int): Current generation step.
            gamma (int): Total number of generation steps.

        Returns:
            Tuple containing:
                - (padded_cache, padded_prob): The padded data for the removed branch.
                - (remain_cache, remain_prob): The remaining branch data after removal.
        """
        padded_cache = []
        remain_cache = []
        pad_len = gamma - i - 1  # Number of padding steps required.
        
        # Process each layer's cache.
        for layer in cache_next:
            k, v = layer
            # Extract cache corresponding to the invalid branch.
            _k = k[b:b + 1]
            _v = v[b:b + 1]
            # Create zero padding for remaining steps.
            k_pad = torch.zeros(1, k.shape[1], pad_len, k.shape[3], device=k.device)
            v_pad = torch.zeros(1, v.shape[1], pad_len, v.shape[3], device=v.device)
            # Concatenate the invalid branch cache with padding.
            k_pad = torch.cat([_k, k_pad], dim=2)
            v_pad = torch.cat([_v, v_pad], dim=2)
            padded_cache.append((k_pad, v_pad))
            
            # Remove the invalid branch from cache.
            _k1 = k[:b]
            _v1 = v[:b]
            _k2 = k[b + 1:]
            _v2 = v[b + 1:]
            remain_cache.append((torch.cat([_k1, _k2], dim=0), torch.cat([_v1, _v2], dim=0)))
        
        # Process probability history.
        prob = prob_history[b:b + 1]
        prob_pad = torch.zeros(1, pad_len, prob.shape[2], device=prob.device)
        padded_prob = torch.cat([prob, prob_pad], dim=1)
        
        # Remove the invalid branch from probability history.
        prob1 = prob_history[:b]
        prob2 = prob_history[b + 1:]
        remain_prob = torch.cat([prob1, prob2], dim=0)
        
        return (padded_cache, padded_prob), (remain_cache, remain_prob)

    def cache_get_by_index(self, cache, i):
        """
        Pop the cache for the branch at index i.

        Args:
            cache: List of cached key-value pairs for all branches.
            i (int): Index of the branch to pop.

        Returns:
            List of key-value pairs for the specified branch.
        """
        popped_cache = []
        for layer in cache:
            k, v = layer
            _k = k[i:i + 1]
            _v = v[i:i + 1]
            popped_cache.append((_k, _v))
        return popped_cache
    
    def pad_branch(self, cache_next, prob_history, i, gamma):
        """
        Pad the cache and probability history for a finished branch.

        Args:
            cache_next: List of cached key-value pairs for all branches.
            prob_history (torch.Tensor): Probability history for all branches.
            i (int): Index of the current generation step.
            gamma (int): Total number of generation steps.

        Returns:
            Tuple containing:
                - padded_cache: The padded cache for the finished branch.
                - padded_prob: The padded probability history for the finished branch.
        """
        padded_cache = []
        pad_len = gamma - i - 1
        if pad_len <= 0:
            return cache_next, prob_history
        # Process each layer's cache.
        for layer in cache_next:
            k, v = layer
            # Create zero padding for remaining steps.
            k_pad = torch.zeros(k.shape[0], k.shape[1], pad_len, k.shape[3], device=k.device)
            v_pad = torch.zeros(v.shape[0], v.shape[1], pad_len, v.shape[3], device=v.device)
            # Concatenate the finished branch cache with padding.
            k_pad = torch.cat([k, k_pad], dim=2)
            v_pad = torch.cat([v, v_pad], dim=2)
            padded_cache.append((k_pad, v_pad))
            
        # Process probability history.
        prob_pad = torch.zeros(prob_history.shape[0], pad_len, prob_history.shape[2], device=prob_history.device)
        padded_prob = torch.cat([prob_history, prob_pad], dim=1)
        return padded_cache, padded_prob
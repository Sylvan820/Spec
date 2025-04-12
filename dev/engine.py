import torch
import transformers
import warnings

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
from .kvcache_draft import KVCacheDraftModel
from .branch_model import BranchModel
from .util import seed_everything, norm_logits, sample, max_fn
from .gamma_predictor import GammaPredictor
import time
import ipdb


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # ! only parallel speculative decoding can use 2 processes
        assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (
                self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "bran_sd", "para_sd_wo_1",
                                                                           "para_sd_wo_1", "rc_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.num_rollback_tokens = 0
        self.branch_acc_tokens = 0
        self.branch_reject_tokens = 0
        self.prob_accept = 0
        self.prob_reject = 0
        self.token_verifed = 0
        self.flag = 1

        # Add prediction accuracy tracking
        self.total_predictions = 0
        self.correct_predictions = 0

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="auto",
                                                                    torch_dtype=torch.bfloat16,
                                                                    trust_remote_code=True).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto",
                                                                     torch_dtype=torch.bfloat16,
                                                                     trust_remote_code=True).eval()
        elif self.args.eval_mode == "sd":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                    torch_dtype=torch.bfloat16,
                                                                    trust_remote_code=True).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model,
                                                                     device_map="balanced_low_0",
                                                                     torch_dtype=torch.bfloat16,
                                                                     trust_remote_code=True).eval()

        elif self.args.eval_mode in ["para_sd", "bran_sd", "para_sd_wo_1", "para_sd_wo_1"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                        torch_dtype=torch.bfloat16,
                                                                        trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model,
                                                                         device_map="balanced_low_0",
                                                                         torch_dtype=torch.bfloat16,
                                                                         trust_remote_code=True).eval()

        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0",
                                                                        torch_dtype=torch.bfloat16,
                                                                        trust_remote_code=True).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(self.args.draft_model,
                                                                          device_map=f"cuda:{torch.cuda.device_count() - 1}",
                                                                          torch_dtype=torch.bfloat16,
                                                                          trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto",
                                                                         torch_dtype=torch.bfloat16,
                                                                         trust_remote_code=True).eval()

        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"

        # for llama models
        self.tokenizer.pad_token_id = 2

    @torch.no_grad()
    def autoregressive_sampling(self, prefix):
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")

        prefix = prefix.to(model.device)

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens

        x = prefix
        past_key_values = None
        while x.shape[1] < max_tokens:
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids.unsqueeze(0)
                outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = model(x)

            if self.accelerator.is_main_process:
                if self.args.eval_mode == "small":
                    self.draft_forward_times += 1
                elif self.args.eval_mode == "large":
                    self.target_forward_times += 1

            last_p = norm_logits(outputs.logits[::, -1, :], self.args.temp, self.args.top_k, self.args.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
        return x

    @torch.no_grad()
    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1

            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (
                approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break

            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]

            approx_model_cache.rollback(n + 1)

            if n < prefix_len + self.args.gamma - 1:
                # reject someone, sample from the pos n
                t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(
                    draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
                target_model_cache.rollback(n + 2)
            prefix = torch.cat((prefix, t), dim=1)
        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma * 2] = x[:, prefix_len - self.args.gamma + 1:prefix_len + self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(
                    torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            draft_ids = all_prob[0, [0], 1:self.args.gamma * 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)

                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                        self.num_rollback_tokens += self.args.gamma
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len - self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    self.num_rollback_tokens += self.args.gamma+self.args.gamma - n - 1
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)

        return prefix

    @torch.no_grad()
    def branch_speculative_decoding(self, prefix):
        gamma = self.args.gamma
        branches = self.args.branches

        # branch speculative decoding
        if self.accelerator.is_main_process:
            model = BranchModel(self.draft_model, 1, 0, 0)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

            # 初始化gamma预测器
            gamma_predictor = GammaPredictor(hidden_dim=16384, embedding_dim=4096)  # 根据模型实际维度设置
            gamma_predictor.load_pretrained('best1.pt')
            gamma_predictor = gamma_predictor.to(device)
            gamma_predictor.eval()

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0
        branch_acc_token = 0
        branch_reject_token = 0

        # 跟踪用于gamma预测的数据
        last_target_hidden = None
        last_selected_token_embedding = None

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)

            if self.accelerator.is_main_process:
                # start_time = time.perf_counter()  # Start timing
                candidate_outputs, invalid_indices = model.generate(input_ids, gamma, branches)
                prob = model.temp_prob[:, prefix_len - gamma - 1:prefix_len, :self.vocab_size].to(torch.float32)
                # transfer the candidate outputs to the prob tensor
                for b in range(branches):
                    prob[b, 0, 0] = -1
                    prob[b, 0, 1:gamma * 2] = candidate_outputs[b, prefix_len - gamma + 1: prefix_len + gamma]
                    prob[b, 0, gamma * 2 + 1] = float(invalid_indices[b])
                self.draft_forward_times += gamma
                flag_tensor = torch.tensor([self.flag], device=device)
                # elapsed = time.perf_counter() - start_time
                # print(f"draft time in {elapsed:.6f} seconds (fast path)")
                # ipdb.set_trace()
            else:
                # start_time = time.perf_counter()  # Start timing
                output = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - gamma - 1: prefix_len, :self.vocab_size].to(
                    torch.float32)
                self.target_forward_times += 1
                prob = prob.repeat(branches, 1, 1)  # repeat the prob tensor for gathering
                # elapsed = time.perf_counter() - start_time
                # print(f"target time in {elapsed:.6f} seconds (fast path)")

            self.accelerator.wait_for_everyone()

            all_prob = self.accelerator.gather(prob).to(device)

            invalid_indices = all_prob[:branches, 0, gamma * 2 + 1].flatten().int()
            draft_ids = all_prob[:branches, 0, 1:gamma * 2].int()
            draft_prob = all_prob[:branches, 1:, :]
            target_prob = all_prob[[branches], 1:, :]

            if cur_mode == True:
                # verification
                first_token = draft_ids[:, -gamma]
                torch.manual_seed(self.seed + prefix_len)
                rand_val = torch.rand(1, device=device)

                ratios = torch.tensor(
                    [target_prob[0, -1, first_token[b]] for b in range(branches)],
                )
                max_ratio, best_branch = torch.max(ratios, dim=0)

                speculative_ratio = max_ratio / (draft_prob[best_branch, -1, first_token[best_branch]] + 1e-8)
                if speculative_ratio >= rand_val:
                    num_acc_token += 1
                    # self.token_verifed = invalid_indices[best_branch]

                    # 在选择分支后，预测下一轮的gamma值（仅非主进程）
                    if not self.accelerator.is_main_process:
                        # 获取target model最后四层的hidden states
                        if hasattr(model, '_last_four_layers') and model._last_four_layers is not None:
                            # 获取最后一个token的hidden states
                            last_target_hidden = torch.cat(
                                [layer[:, -1, :].clone() for layer in model._last_four_layers], dim=-1)

                            # 获取best_branch的第一个token的embedding
                            if hasattr(self.target_model, 'get_input_embeddings'):
                                embedding_layer = self.target_model.get_input_embeddings()
                                first_token_id = first_token[best_branch].view(1, -1)
                                last_selected_token_embedding = embedding_layer(first_token_id).squeeze(1).clone()

                                # 使用预测器预测下一轮的gamma值
                                if last_target_hidden is not None and last_selected_token_embedding is not None:
                                    predicted_gamma = gamma_predictor(last_target_hidden, last_selected_token_embedding)
                                    if predicted_gamma == 1:
                                        self.flag = 0
                                    elif predicted_gamma == 4:
                                        self.flag = 1
                                    else:
                                        self.flag = 6

                                    # 从非主进程传递flag值到主进程
                                    flag_tensor = torch.tensor([self.flag], device=device)

                    gathered_flags = self.accelerator.gather(flag_tensor).to(device)
                    # 所有进程都使用非主进程(index=1)的值
                    self.flag = gathered_flags[1].item()
                    # print('self.flag1', self.flag)

                    if self.flag != 1:
                        invalid_indices[best_branch] = self.flag

                    self.token_verifed = invalid_indices[best_branch]

                    prefix = torch.cat(
                        (input_ids, draft_ids[
                                    [best_branch], gamma - 1:gamma + invalid_indices[best_branch]
                                    ]),
                        dim=1)


                    if self.accelerator.is_main_process:
                        model.select_branch(best_branch)

                        model.rollback(prefix_len + invalid_indices[best_branch] + 1)
                        model.trace_mode = True if invalid_indices[best_branch] < gamma - 1 else False

                        if self.flag == 0:
                            model.invalid_logits[0] = model.first_logit[0]

                    if invalid_indices[best_branch] > 0:
                        cur_mode = False


                else:  # speculative_ratio < rand_val
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[[best_branch], -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    if self.accelerator.is_main_process:
                        model.rollback(prefix_len)
                        self.num_rollback_tokens += gamma
                        model.trace_mode = False

            else:  # cur_mode == False
                n = gamma - 1
                for i in range(gamma - self.token_verifed - 1,
                               gamma - 1):  # from the last verified token to the last token
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - gamma + i)
                    rand_val = torch.rand(1, device=device)
                    ratio = target_prob[0, i, token[0]] / (draft_prob[0, i, token[0]] + 1e-8)

                    if ratio < rand_val:
                        n = i
                        break
                # Evaluate prediction accuracy
                # self.total_predictions += 1
                # if (self.flag == 0 and n == 0) or (self.flag == 6 and n == gamma - 1) or (
                #         self.flag == 1 and n not in [0, gamma - 1]):
                #     self.correct_predictions += 1
                #
                # # Print current accuracy
                # if self.total_predictions % 1 == 0:  # Print every 100 predictions
                #     accuracy = (self.correct_predictions / self.total_predictions) * 100
                #     print(
                #         f"Current Prediction Accuracy: {accuracy:.2f}% ({self.correct_predictions}/{self.total_predictions})")
                #     print(f"Current n: {n}, Current flag: {self.flag}")

                if n == gamma - 1:  # accept all tokens
                    last_token = draft_ids[:, n]
                    torch.manual_seed(self.seed + prefix_len - gamma + n)
                    rand_val = torch.rand(1, device=device)

                    ratios = torch.tensor(
                        [target_prob[0, n, last_token[b]] for b in range(branches)],
                    )
                    max_ratio, best_branch = torch.max(ratios, dim=0)

                    speculative_ratio = max_ratio / (draft_prob[best_branch, n, last_token[best_branch]] + 1e-8)
                    if speculative_ratio >= rand_val:
                        num_acc_token += self.token_verifed + 1

                        # 在选择分支后，预测下一轮的gamma值（仅非主进程）
                        if not self.accelerator.is_main_process:
                            # 获取target model最后四层的hidden states
                            if hasattr(model, '_last_four_layers') and model._last_four_layers is not None:
                                # 获取最后一个token的hidden states
                                last_target_hidden = torch.cat(
                                    [layer[:, -1, :].clone() for layer in model._last_four_layers], dim=-1)

                                # 获取best_branch的第一个token的embedding
                                if hasattr(self.target_model, 'get_input_embeddings'):
                                    embedding_layer = self.target_model.get_input_embeddings()
                                    first_token_id = last_token[best_branch].view(1, -1)
                                    last_selected_token_embedding = embedding_layer(first_token_id).squeeze(1).clone()

                                    # 使用预测器预测下一轮的gamma值
                                    if last_target_hidden is not None and last_selected_token_embedding is not None:
                                        predicted_gamma = gamma_predictor(last_target_hidden,
                                                                          last_selected_token_embedding)
                                        if predicted_gamma == 1:
                                            self.flag = 0
                                        elif predicted_gamma == 4:
                                            self.flag = 1
                                        else:
                                            self.flag = 6

                                        # 从非主进程传递flag值到主进程
                                        flag_tensor = torch.tensor([self.flag], device=device)

                        gathered_flags = self.accelerator.gather(flag_tensor).to(device).to(device)
                        # 所有进程都使用非主进程(index=1)的值
                        self.flag = gathered_flags[1].item()
                        # print('self.flag2', self.flag)
                        if self.flag !=1:
                            invalid_indices[best_branch] = self.flag

                        self.token_verifed = invalid_indices[best_branch]

                        prefix = torch.cat(
                            (input_ids, draft_ids[
                                        [best_branch], gamma - 1:gamma + invalid_indices[best_branch]
                                        ]),
                            dim=1)

                        if self.accelerator.is_main_process:
                            model.select_branch(best_branch)

                            model.rollback(prefix_len + invalid_indices[best_branch] + 1)
                            model.trace_mode = True if invalid_indices[best_branch] < gamma - 1 else False

                            if self.flag == 0:
                                model.invalid_logits[0] = model.first_logit[0]

                        if invalid_indices[best_branch] == 0:
                            cur_mode = True

                    else:  # speculative_ratio < rand_val
                        cur_mode = True
                        t = sample(max_fn(target_prob[:, n, :] - draft_prob[[best_branch], n, :]))
                        prefix = torch.cat((input_ids, t), dim=1)
                        self.num_acc_tokens.append(num_acc_token + self.token_verifed)
                        num_acc_token = 0
                        model.rollback(prefix_len)
                        self.num_rollback_tokens += gamma
                        if self.accelerator.is_main_process:
                            model.trace_mode = False

                else:  # n < gamma - 1
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[[0], n, :]))
                    prefix = torch.cat((input_ids[:, :prefix_len - gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n + self.token_verifed - gamma + 1)
                    num_acc_token = 0
                    model.rollback(prefix_len - gamma + n + 1)
                    self.num_rollback_tokens += gamma+gamma-n-1
                    if self.accelerator.is_main_process:
                        model.trace_mode = False

        return prefix

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")

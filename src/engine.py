import torch
import transformers
import warnings
import ipdb

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
from .kvcache4RC import KVCacheModel as KVCache2Model
from .util import seed_everything, norm_logits, sample, max_fn, make_mask   
import time


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # ! only parallel speculative decoding can use 2 processes
        assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (
                    self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "para_sd_wo_1",
                                                                               "para_sd_wo_1", "rc_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []

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

        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1"]:
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

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

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
    # def speculative_decoding(self, prefix):
    #     max_tokens = prefix.shape[1] + self.args.max_tokens
    #
    #     draft_device = self.draft_model.device
    #     target_device = self.target_model.device
    #
    #     approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
    #     approx_model_cache.vocab_size = self.vocab_size
    #     target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
    #     target_model_cache.vocab_size = self.vocab_size
    #
    #     while prefix.shape[1] < max_tokens:
    #         prefix_len = prefix.shape[1]
    #         x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
    #         _ = target_model_cache.generate(x.to(target_device), 1)
    #         if self.accelerator.is_main_process:
    #             self.draft_forward_times += self.args.gamma
    #             self.target_forward_times += 1
    #
    #         n = prefix_len + self.args.gamma - 1
    #         for i in range(self.args.gamma):
    #             r = torch.rand(1, device=draft_device)
    #             j = x[:, prefix_len + i]
    #
    #             if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
    #                 n = prefix_len + i - 1
    #                 break
    #
    #         self.num_acc_tokens.append(n - prefix_len + 1)
    #
    #         assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
    #         prefix = x[:, :n + 1]
    #
    #         approx_model_cache.rollback(n+1)
    #
    #         if n < prefix_len + self.args.gamma - 1:
    #             # reject someone, sample from the pos n
    #             t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
    #             target_model_cache.rollback(n+1)
    #         else:
    #             # all approx model decoding accepted
    #             t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
    #             target_model_cache.rollback(n+2)
    #         prefix = torch.cat((prefix, t), dim=1)
    #     return prefix

    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size
        # 改进文件读取逻辑
        try:
            with open('token.txt', 'r') as f:
                content = f.read().strip()
                acceptance_stats = list(map(int, content.split(','))) if content else [0] * 17
        except (FileNotFoundError, ValueError):
            acceptance_stats = [0] * 17
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            # print('prefix',prefix)
            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)
            # print('x',x)
            # print('_', _)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1

            n = prefix_len + self.args.gamma - 1
            accepted_count = 0
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]
                if r > (target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]) / (
                approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break
                accepted_count += 1
            # 更新统计数组并保存到文件
            if self.accelerator.is_main_process and accepted_count < 17:
                acceptance_stats[accepted_count] += 1
                with open('token.txt', 'w') as f:
                    f.write(','.join(map(str, acceptance_stats)))

            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            approx_model_cache.rollback(n + 1)

            if n < prefix_len + self.args.gamma - 1:
                t = sample(max_fn(target_model_cache._prob_history[:, n, :self.vocab_size].to(
                    draft_device) - approx_model_cache._prob_history[:, n, :self.vocab_size]))
                target_model_cache.rollback(n + 1)
            else:
                t = sample(target_model_cache._prob_history[:, -1, :self.vocab_size]).to(draft_device)
                target_model_cache.rollback(n + 2)
            prefix = torch.cat((prefix, t), dim=1)
        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        """并行推测解码的另一种实现，支持分支预测

        该方法使用两个模型并行工作：
        - draft model(小模型)用于生成多个分支的候选tokens
        - target model(大模型)用于验证这些候选tokens的质量

        Args:
            prefix (tensor): 输入序列的token ids
        Returns:
            tensor: 完整的生成序列
        """
        # 根据进程类型初始化对应的模型
        if self.accelerator.is_main_process:
            # 主进程初始化小模型(draft model)
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            # 其他进程初始化大模型(target model)
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        # 计算需要生成的总token数
        max_tokens = prefix.shape[1] + self.args.max_tokens

        cur_mode = True
        # 记录当前批次接受的token数量
        num_acc_token = 0

        # 确定分支数量
        num_branches = self.args.top_k if self.args.top_k > 0 else 3

        # 主生成循环
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            # 将输入序列移至对应设备
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:

                ipdb.set_trace()

                # 小模型生成多个分支
                x = model.branch(input_ids, self.args.gamma)

                # 获取概率历史，包括前一个token和新生成的tokens
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]

                # 设置特殊标记(-1)用于后续处理
                prob[:, 0, 0] = -1

                # 存储生成的所有分支的token ids
                branch_tokens = x[0, :, prefix_len:prefix_len + self.args.gamma].reshape(-1)
                prob[:, 0, 1:1 + branch_tokens.numel()] = branch_tokens

                # 更新小模型前向传播次
                self.draft_forward_times += self.args.gamma
            else:
                # 大模型只生成一个token进行验证
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                # 将概率移至cuda:1设备
                prob = prob.to("cuda:1")
                # 更新大模型前向传播次数
                self.target_forward_times += 1

            # 等待所有进程完成ppro
            self.accelerator.wait_for_everyone()

            # 收集并验证所有概率
            all_prob = self.accelerator.gather(prob).to(device)

            # 提取小模型生成的分支token ids
            branch_tokens = all_prob[0, 0, 1:1 + num_branches * self.args.gamma].reshape(num_branches, self.args.gamma).int()
            print('branch_tokens',branch_tokens)

            # 分别获取小模型和大模型的概率
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]


            if cur_mode:
                # 单token验证模式 - 验证每个分支的第一个token
                accepted_branches = []
                accepted_probs = []

                for branch in range(num_branches):
                    first_token = branch_tokens[branch, 0]
                    print('first_token',first_token)
                    # 这块是不是要调整 seed的随机性
                    torch.manual_seed(self.seed + prefix_len)
                    r = torch.rand(1, device=device)

                    # 使用接受/拒绝采样验证当前token
                    if not (r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]):
                        accepted_branches.append(branch)
                        accepted_probs.append(draft_prob[:, -1, first_token].item())
                print('accepted_branches', accepted_branches)
                if len(accepted_branches) == 0:
                    # 所有分支的第一个token都被拒绝，使用大模型重新采样
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)

                    # 记录已接受的token数并重置计数器
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # 拒绝时需要回滚小模型的KV cache
                        model.rollback(prefix_len)
                else:
                    # 至少有一个分支被接受，选择概率最高的
                    if len(accepted_branches) > 1:
                        print('accepted_probs',accepted_probs)
                        selected_idx = torch.argmax(torch.tensor(accepted_probs))
                        selected_branch = accepted_branches[selected_idx]
                    else:
                        selected_branch = accepted_branches[0]
                    print('selected_branch', selected_branch)
                    # 接受第一个token，切换到批量验证模式
                    cur_mode = False
                    print('input_ids',input_ids)
                    print('branch_tokens[selected_branch]', branch_tokens[selected_branch])
                    prefix = torch.cat((input_ids, branch_tokens[selected_branch, 0:1]), dim=1)
                    num_acc_token += 1
            else:
                # 批量验证模式 - 验证剩余tokens
                selected_branch_tokens = branch_tokens[0]  # 使用第一个分支的tokens

                # 找到第一个被拒绝的位置
                n = self.args.gamma

                # 验证除最后一个token以外的所有token
                for i in range(1, self.args.gamma - 1):
                    token = selected_branch_tokens[i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)

                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break

                # 特殊处理最后一个token - 对所有分支的最后一个token进行验证
                if n == self.args.gamma - 1:
                    accepted_branches = []
                    accepted_probs = []

                    # 验证每个分支的最后一个token
                    for branch in range(num_branches):
                        last_token = branch_tokens[branch, -1]
                        torch.manual_seed(self.seed + prefix_len - self.args.gamma + n)
                        r = torch.rand(1, device=device)

                        if not (r > target_prob[:, n, last_token] / draft_prob[:, n, last_token]):
                            accepted_branches.append(branch)
                            accepted_probs.append(draft_prob[:, n, last_token].item())

                    if len(accepted_branches) == 0:
                        # 所有分支的最后一个token都被拒绝
                        t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                        prefix = torch.cat((input_ids, selected_branch_tokens[1:n], t), dim=1)

                        # 更新统计信息
                        self.num_acc_tokens.append(num_acc_token + n)
                        num_acc_token = 0

                        # 回滚模型的KV cache到正确位置
                        model.rollback(prefix_len + n)

                        # 切换回单token验证模式
                        cur_mode = True
                    else:
                        # 至少有一个分支的最后token被接受，选择概率最高的
                        if len(accepted_branches) > 1:
                            selected_idx = torch.argmax(torch.tensor(accepted_probs))
                            selected_branch = accepted_branches[selected_idx]
                        else:
                            selected_branch = accepted_branches[0]

                        # 更新prefix，接受前n个token和选定分支的最后一个token
                        last_token = branch_tokens[selected_branch, -1:]
                        prefix = torch.cat((input_ids, selected_branch_tokens[1:n], last_token), dim=1)
                        num_acc_token += self.args.gamma

                        # 保持批量验证模式
                        cur_mode = False
                elif n < self.args.gamma:
                    # 有token被拒绝
                    cur_mode = True

                    # 为被拒绝的位置采样新token
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    # 更新prefix到被拒绝的位置，并添加新token
                    prefix = torch.cat((input_ids, selected_branch_tokens[1:n], t), dim=1)

                    # 更新统计信息
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0

                    # 回滚模型的KV cache到正确位置
                    model.rollback(prefix_len + n)

        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_RC(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCache2Model(self.draft_model, self.draft_model_2, self.args.temp, self.args.top_k,
                                  self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:0")
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:1")

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma * 2] = x[:, prefix_len - self.args.gamma + 1:prefix_len + self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                # ! the prob of the target model should be moved to a different device of the draft device to avoid deadlock
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
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)

            self.accelerator.wait_for_everyone()

        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_without_strategy_1(self, prefix):
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

        # this flag is used to determine whether to use the strategy 2
        cur_mode = False

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma * 2] = x[:, prefix_len - self.args.gamma + 1:prefix_len + self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            all_prob = self.accelerator.gather(prob)

            assert all_prob[0, 0, 0] == -1
            draft_ids = all_prob[0, [0], 1:self.args.gamma * 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]

            if cur_mode:
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
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = False
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len - self.args.gamma + n + 1], t), dim=1)
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)

            else:
                prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                cur_mode = True

        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_without_strategy_2(self, prefix):
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

        # this flag is used to determine whether to use the strategy 1
        cur_mode = True

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                print()
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma * 2] = x[:, prefix_len - self.args.gamma + 1:prefix_len + self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                print()
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size]
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            all_prob = self.accelerator.gather(prob)

            assert all_prob[0, 0, 0] == -1
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

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)

            else:
                n = self.args.gamma - 1
                for i in range(self.args.gamma - 1):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break

                cur_mode = True
                if n == self.args.gamma - 1:
                    t = sample(target_prob[:, n, :])
                else:
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                prefix = torch.cat((input_ids[:, :prefix_len - self.args.gamma + n + 1], t), dim=1)
                # rollback both the large model and the small model kv cache
                model.rollback(prefix_len - self.args.gamma + n + 1)

        return prefix

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")

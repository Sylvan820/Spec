import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_zoo(args):
    vocab_size = {
        "tinyllamacode-1.1b": 32000,
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        "deepseek-1.3b": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "vicuna-68m": 32000,
        "vicuna-7b": 32000,
        "llama-68m": 32000,
        "llama-160m": 32000,
        "llama-7b": 32000,
        "llama-13b": 32000,
        "llama-30b": 32000,
        "qwen2.5-0.5b": 151643,
        "qwen2.5-1.5b": 151643,
        "qwen2.5-3b": 151643,
        "qwen2.5-7b": 151643,
        "qwen2.5-14b": 151643,
        "qwen2.5-32b": 151643,
        "qwen2.5-72b": 151643,
    }

    zoo = {
        "tinyllamacode-1.1b": "/remote-home/security_shenyuhao/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5",
        "codellama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed",
        "codellama-34b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d",
        "codellama-70b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-70b-Instruct-hf/snapshots/397cae981dffaf5d5c9c90e89a0a75a850528b70",
        "llama-2-7b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "llama-2-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "deepseek-1.3b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-instruct/snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca",
        "deepseek-6.7b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots/e5d64addd26a6a1db0f9b863abf6ee3141936807",
        "deepseek-33b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-33b-instruct/snapshots/61dc97b922b13995e7f83b7c8397701dbf9cfd4c",
        "vicuna-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109",
        "vicuna-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
        "llama-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-68m/snapshots/964a5d77df908b69f8d6476fb70e940425b04cb5",
        "llama-160m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-160m/snapshots/aca9b687d1425f863dcf5de9a4c96e3fe36266dd",
        "llama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
        "llama-13b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba",
        "llama-30b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-30b/snapshots/2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
        "qwen2.5-0.5b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
        "qwen2.5-1.5b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "qwen2.5-3b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "qwen2.5-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "qwen2.5-14b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "qwen2.5-32b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "qwen2.5-72b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
    }

    args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    args.target_model = zoo[args.target_model]


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--data_path', type=str,
                        default="/remote-home/security_shenyuhao/huggingface/hub/datasets--openai--openai_humaneval")
    # parser.add_argument('--data_path', type=str,
    #                     default="/remote-home/security_shenyuhao/huggingface/hub/datasets--openai--gsm8k")
    parser.add_argument('--draft_model', type=str, default="vicuna-68m")
    parser.add_argument('--target_model', type=str, default="vicuna-7b")

    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="small",
                        choices=["small", "large", "sd", "para_sd", "bran_sd", "para_sd_wo_1", "para_sd_wo_2"],
                        help='eval mode.')
    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1,
                        help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234,
                        help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=1, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=10, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    parser.add_argument('--branches', type=int, default=3, help='branchs.')
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    # 批处理优化：检查是否有多个批次
    batch_size = logits.size(0)

    # 快速路径：如果不需要过滤
    if top_k <= 0 and top_p <= 0.0:
        return logits.clone()  # 返回副本以保持一致性

    # 创建logits的副本以避免修改原始数据
    # 使用contiguous()确保内存布局连续，提高后续操作效率
    filtered_logits = logits.clone().contiguous()

    # 批处理优化：对于大批量，使用并行处理
    if batch_size > 1 and filtered_logits.size(-1) > 5000:
        # 并行处理大批量
        # 这里我们使用向量化操作，而不是循环处理每个批次

        # 应用top-k过滤（如果需要）
        if top_k > 0:
            # 对每个批次并行应用top-k
            k = min(top_k, filtered_logits.size(-1))
            values, _ = torch.topk(filtered_logits, k, dim=-1)
            # 获取每个批次的阈值
            min_values = values[:, -1].unsqueeze(-1)
            # 使用广播和原地操作设置掩码
            filtered_logits = torch.where(
                filtered_logits < min_values.expand_as(filtered_logits),
                torch.full_like(filtered_logits, float('-inf')),
                filtered_logits
            )

        # 应用top-p过滤（如果需要）
        if top_p > 0.0:
            # 优化：如果top_k已经应用且k很小，可以跳过top_p
            if top_k > 0 and top_k < 50:
                return filtered_logits

            # 对每个批次并行应用top-p
            # 注意：这部分难以完全并行化，但我们可以优化计算

            # 对每个批次排序
            sorted_logits, sorted_indices = torch.sort(filtered_logits, dim=-1, descending=True)

            # 计算softmax和累积概率（并行）
            max_logits = torch.max(sorted_logits, dim=-1, keepdim=True)[0]
            exp_logits = torch.exp(sorted_logits - max_logits)
            exp_sum = torch.cumsum(exp_logits, dim=-1)
            exp_total = exp_sum[:, -1].unsqueeze(-1)
            cumulative_probs = exp_sum / exp_total

            # 创建要移除的token的掩码
            remove_tokens = cumulative_probs > top_p

            # 确保至少保留一个token
            if remove_tokens.size(1) > 1:
                remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()
            remove_tokens[:, 0] = 0

            # 为每个批次创建掩码并应用
            for i in range(batch_size):
                indices_to_remove = torch.zeros_like(filtered_logits[i], dtype=torch.bool)
                indices_to_remove.scatter_(0, sorted_indices[i], remove_tokens[i])
                filtered_logits[i].masked_fill_(indices_to_remove, float('-inf'))

            return filtered_logits

    # 对于小批量或小词汇表，使用标准处理
    # 应用top-k过滤（如果需要）
    if top_k > 0:
        # 使用torch.topk更高效，避免对整个张量排序
        k = min(top_k, filtered_logits.size(-1))

        # 优化：对于大词汇表，使用更高效的方法
        if filtered_logits.size(-1) > 10000:  # 对于大词汇表
            # 只获取top-k值，不需要索引
            values = torch.topk(filtered_logits, k)[0]
            # 获取阈值（每批次中第k个最大值）
            min_values = values[:, -1].unsqueeze(-1)
            # 使用原地操作设置掩码
            filtered_logits.masked_fill_(filtered_logits < min_values, float('-inf'))
        else:
            # 对于小词汇表，直接使用标准方法
            values = torch.topk(filtered_logits, k)[0]
            min_values = values[:, -1].unsqueeze(-1)
            filtered_logits.masked_fill_(filtered_logits < min_values, float('-inf'))

    # 应用top-p（核采样）过滤（如果需要）
    if top_p > 0.0:
        # 优化：如果top_k已经应用且k很小，可以跳过top_p
        if top_k > 0 and top_k < 50:  # 如果已经过滤到很少的token
            return filtered_logits

        # 只对需要应用top-p过滤的情况进行排序
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)

        # 计算softmax和累积概率
        # 使用更高效的方法计算softmax
        max_logits = torch.max(sorted_logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(sorted_logits - max_logits)
        exp_sum = torch.cumsum(exp_logits, dim=-1)
        exp_total = exp_sum[:, -1].unsqueeze(-1)
        cumulative_probs = exp_sum / exp_total

        # 创建要移除的token的掩码
        remove_tokens = cumulative_probs > top_p

        # 确保至少保留一个token
        if remove_tokens.size(1) > 1:
            remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()
        remove_tokens[:, 0] = 0

        # 应用掩码到排序后的logits
        # 使用更高效的方法应用掩码
        indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, remove_tokens)
        filtered_logits.masked_fill_(indices_to_remove, float('-inf'))

    return filtered_logits


# 添加缓存字典和缓存大小限制
_logits_cache = {}
_CACHE_SIZE_LIMIT = 10  # 限制缓存大小，避免内存泄漏


def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    start_time = time.perf_counter()  # Start timing

    # 检查缓存
    # 使用参数和logits的哈希作为缓存键
    # 注意：我们只对小批量的logits使用缓存，以避免哈希计算开销超过收益
    if logits.size(0) == 1 and logits.size(1) < 10000:
        cache_key = (temperature, top_k, top_p, logits.sum().item(), logits[0, 0].item())
        if cache_key in _logits_cache:
            probs, confidence = _logits_cache[cache_key]
            elapsed = time.perf_counter() - start_time
            print(f"norm_logits executed in {elapsed:.6f} seconds (cache hit)")
            return probs, confidence

    # 超快路径：如果不需要过滤且温度为1.0
    if top_k <= 0 and top_p <= 0.0 and abs(temperature - 1.0) < 1e-6:
        # 直接计算softmax，避免不必要的复制和过滤
        with torch.no_grad():  # 使用no_grad加速计算
            probs = F.softmax(logits, dim=1)
        elapsed = time.perf_counter() - start_time
        # print(f"norm_logits executed in {elapsed:.6f} seconds (fast path)")

        # 对于小批量，更新缓存
        if logits.size(0) == 1 and logits.size(1) < 10000:
            # 限制缓存大小
            if len(_logits_cache) >= _CACHE_SIZE_LIMIT:
                # 移除随机键以避免缓存过大
                _logits_cache.pop(next(iter(_logits_cache.keys())))
            _logits_cache[cache_key] = (probs, probs)

        return probs, probs

    # 处理零温度
    if temperature == 0:
        temperature = 1e-8

    # 使用torch.no_grad()加速计算，因为我们不需要梯度
    with torch.no_grad():
        # 应用top_k和top_p过滤
        filtered_logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

        # 根据温度是否为1.0优化计算
        if abs(temperature - 1.0) < 1e-6:
            # 如果温度接近1.0，我们可以计算一次并重用
            probs = F.softmax(filtered_logits, dim=1)
            confidence = probs  # 直接引用，不创建新张量
        else:
            # 需要温度缩放和单独计算
            # 先计算confidence（使用未缩放的logits）
            confidence = F.softmax(filtered_logits, dim=1)
            # 然后计算probs（使用缩放后的logits）
            # 使用原地操作进行温度缩放
            scaled_filtered_logits = filtered_logits / temperature
            probs = F.softmax(scaled_filtered_logits, dim=1)

    # elapsed = time.perf_counter() - start_time
    # print(f"norm_logits executed in {elapsed:.6f} seconds")

    # 对于小批量，更新缓存
    if logits.size(0) == 1 and logits.size(1) < 10000:
        # 限制缓存大小
        if len(_logits_cache) >= _CACHE_SIZE_LIMIT:
            # 移除随机键以避免缓存过大
            _logits_cache.pop(next(iter(_logits_cache.keys())))
        _logits_cache[cache_key] = (probs, confidence)

    return probs, confidence


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def top_k_top_p_filter1(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits1(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        tuple: (probs, confidence) where probs is the probability distribution and confidence is the confidence score
    """
    assert logits.dim() == 2
    if temperature == 0:
        temperature += 1e-8
    confidence = logits.clone()
    logits = logits / temperature
    # confidence = top_k_top_p_filter1(confidence, top_k=top_k, top_p=top_p)
    # confidence = F.softmax(confidence, dim=1)
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs, confidence

# def make_mask(past_len, seq_len, device, dtype=torch.Bfloat16):
#     """创建因果掩码，用于控制模型在生成时可以看到的token
#
#     Args:
#         past_len (int): 已经生成的token长度
#         seq_len (int): 序列长度
#         device: 设备
#         dtype: 数据类型，默认为torch.float32
#
#     Returns:
#         torch.Tensor: 因果掩码
#     """
#     min_dtype = torch.finfo(dtype).min
#     causal_mask = torch.full((seq_len, past_len + seq_len), fill_value=min_dtype, dtype=dtype, device=device) # full of min
#     causal_mask[:, :past_len+1] = 0 # set 0 for past seen tokens and the first ground truth token
#     diag_idx = torch.arange(seq_len, device=device) # generate diag index
#     causal_mask[diag_idx, past_len + diag_idx] *= False # set 0 for draft tokens
#     causal_mask = causal_mask[None,None,:,:].expand(1,1,-1,-1)
#     return causal_maskimport os
# import random
# import argparse
# import torch
# import torch.nn.functional as F
# import numpy as np
# import time
#
#
# def seed_everything(seed: int):
#     "set all random seed for reproducible results."
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#
#
# def model_zoo(args):
#     vocab_size = {
#         "tinyllamacode-1.1b": 32000,
#         "codellama-7b": 32000,
#         "codellama-34b": 32000,
#         "codellama-70b": 32000,
#         "llama-2-7b": 32000,
#         "llama-2-70b": 32000,
#         "deepseek-1.3b": 32256,
#         "deepseek-6.7b": 32256,
#         "deepseek-33b": 32256,
#         "vicuna-68m": 32000,
#         "vicuna-7b": 32000,
#         "llama-68m": 32000,
#         "llama-160m": 32000,
#         "llama-7b": 32000,
#         "llama-13b": 32000,
#         "llama-30b": 32000,
#         "qwen2.5-0.5b": 151643,
#         "qwen2.5-1.5b": 151643,
#         "qwen2.5-3b": 151643,
#         "qwen2.5-7b": 151643,
#         "qwen2.5-14b": 151643,
#         "qwen2.5-32b": 151643,
#         "qwen2.5-72b": 151643,
#     }
#
#     zoo = {
#         "tinyllamacode-1.1b": "/remote-home/security_shenyuhao/huggingface/hub/models--TinyLlama--TinyLlama_v1.1_math_code/snapshots/698ef988e06730a38eca552cdf86e99c08118df5",
#         "codellama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed",
#         "codellama-34b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/d4c1c474abcacd32d2a6eda45f9811d38c83e93d",
#         "codellama-70b": "/remote-home/security_shenyuhao/huggingface/hub/models--codellama--CodeLlama-70b-Instruct-hf/snapshots/397cae981dffaf5d5c9c90e89a0a75a850528b70",
#         "llama-2-7b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
#         "llama-2-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
#         "deepseek-1.3b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-instruct/snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca",
#         "deepseek-6.7b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-6.7b-instruct/snapshots/e5d64addd26a6a1db0f9b863abf6ee3141936807",
#         "deepseek-33b": "/remote-home/security_shenyuhao/huggingface/hub/models--deepseek-ai--deepseek-coder-33b-instruct/snapshots/61dc97b922b13995e7f83b7c8397701dbf9cfd4c",
#         "vicuna-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109",
#         "vicuna-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
#         "llama-68m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-68m/snapshots/964a5d77df908b69f8d6476fb70e940425b04cb5",
#         "llama-160m": "/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-160m/snapshots/aca9b687d1425f863dcf5de9a4c96e3fe36266dd",
#         "llama-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",
#         "llama-13b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba",
#         "llama-30b": "/remote-home/security_shenyuhao/huggingface/hub/models--huggyllama--llama-30b/snapshots/2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
#         "qwen2.5-0.5b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
#         "qwen2.5-1.5b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
#         "qwen2.5-3b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
#         "qwen2.5-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
#         "qwen2.5-14b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
#         "qwen2.5-32b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
#         "qwen2.5-72b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
#     }
#
#     args.vocab_size = vocab_size[args.draft_model]
#     args.draft_model = zoo[args.draft_model]
#     args.target_model = zoo[args.target_model]
#
#
# def parse_arguments():
#     """Specified arguments for running scripts."""
#     parser = argparse.ArgumentParser(description='args for this file')
#
#     parser.add_argument('--data_path', type=str, default="/remote-home/security_shenyuhao/huggingface/hub/datasets--openai--openai_humaneval")
#     # parser.add_argument('--data_path', type=str,
#     #                     default="/remote-home/security_shenyuhao/huggingface/hub/datasets--openai--gsm8k")
#     parser.add_argument('--draft_model', type=str, default="vicuna-68m")
#     parser.add_argument('--target_model', type=str, default="vicuna-7b")
#
#     parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
#     parser.add_argument('--eval_mode', type=str, default="small",
#                         choices=["small", "large", "sd", "para_sd", "bran_sd", "para_sd_wo_1", "para_sd_wo_2"], help='eval mode.')
#     parser.add_argument('--num_samples_per_task', '-n', type=int, default=1,
#                         help='num_samples for a task (prompt) in humaneval dataset.')
#     parser.add_argument('--seed', '-s', type=int, default=1234,
#                         help='set a random seed, which can makes the result reproducible')
#     parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
#     parser.add_argument('--temp', type=float, default=1, help='temperature for generating new tokens.')
#     parser.add_argument('--top_k', type=int, default=10, help='top_k for ungreedy sampling strategy.')
#     parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
#     parser.add_argument('--gamma', type=int, default=4, help='guess time.')
#     parser.add_argument('--branches', type=int, default=3,help='branchs.')
#     args = parser.parse_args()
#     args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
#     os.makedirs(args.exp_name, exist_ok=True)
#     model_zoo(args)
#     return args
#
#
# def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
#     """
#     Args:
#         logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
#         top_k (int, optional): top_k. Defaults to 0.
#         top_p (float, optional): top_p. Defaults to 0.0.
#
#     Returns:
#         torch.Tensor: a renormalized logits
#     """
#     # 批处理优化：检查是否有多个批次
#     batch_size = logits.size(0)
#
#     # 快速路径：如果不需要过滤
#     if top_k <= 0 and top_p <= 0.0:
#         return logits.clone()  # 返回副本以保持一致性
#
#     # 创建logits的副本以避免修改原始数据
#     # 使用contiguous()确保内存布局连续，提高后续操作效率
#     filtered_logits = logits.clone().contiguous()
#
#     # 批处理优化：对于大批量，使用并行处理
#     if batch_size > 1 and filtered_logits.size(-1) > 5000:
#         # 并行处理大批量
#         # 这里我们使用向量化操作，而不是循环处理每个批次
#
#         # 应用top-k过滤（如果需要）
#         if top_k > 0:
#             # 对每个批次并行应用top-k
#             k = min(top_k, filtered_logits.size(-1))
#             values, _ = torch.topk(filtered_logits, k, dim=-1)
#             # 获取每个批次的阈值
#             min_values = values[:, -1].unsqueeze(-1)
#             # 使用广播和原地操作设置掩码
#             filtered_logits = torch.where(
#                 filtered_logits < min_values.expand_as(filtered_logits),
#                 torch.full_like(filtered_logits, float('-inf')),
#                 filtered_logits
#             )
#
#         # 应用top-p过滤（如果需要）
#         if top_p > 0.0:
#             # 优化：如果top_k已经应用且k很小，可以跳过top_p
#             if top_k > 0 and top_k < 50:
#                 return filtered_logits
#
#             # 对每个批次并行应用top-p
#             # 注意：这部分难以完全并行化，但我们可以优化计算
#
#             # 对每个批次排序
#             sorted_logits, sorted_indices = torch.sort(filtered_logits, dim=-1, descending=True)
#
#             # 计算softmax和累积概率（并行）
#             max_logits = torch.max(sorted_logits, dim=-1, keepdim=True)[0]
#             exp_logits = torch.exp(sorted_logits - max_logits)
#             exp_sum = torch.cumsum(exp_logits, dim=-1)
#             exp_total = exp_sum[:, -1].unsqueeze(-1)
#             cumulative_probs = exp_sum / exp_total
#
#             # 创建要移除的token的掩码
#             remove_tokens = cumulative_probs > top_p
#
#             # 确保至少保留一个token
#             if remove_tokens.size(1) > 1:
#                 remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()
#             remove_tokens[:, 0] = 0
#
#             # 为每个批次创建掩码并应用
#             for i in range(batch_size):
#                 indices_to_remove = torch.zeros_like(filtered_logits[i], dtype=torch.bool)
#                 indices_to_remove.scatter_(0, sorted_indices[i], remove_tokens[i])
#                 filtered_logits[i].masked_fill_(indices_to_remove, float('-inf'))
#
#             return filtered_logits
#
#     # 对于小批量或小词汇表，使用标准处理
#     # 应用top-k过滤（如果需要）
#     if top_k > 0:
#         # 使用torch.topk更高效，避免对整个张量排序
#         k = min(top_k, filtered_logits.size(-1))
#
#         # 优化：对于大词汇表，使用更高效的方法
#         if filtered_logits.size(-1) > 10000:  # 对于大词汇表
#             # 只获取top-k值，不需要索引
#             values = torch.topk(filtered_logits, k)[0]
#             # 获取阈值（每批次中第k个最大值）
#             min_values = values[:, -1].unsqueeze(-1)
#             # 使用原地操作设置掩码
#             filtered_logits.masked_fill_(filtered_logits < min_values, float('-inf'))
#         else:
#             # 对于小词汇表，直接使用标准方法
#             values = torch.topk(filtered_logits, k)[0]
#             min_values = values[:, -1].unsqueeze(-1)
#             filtered_logits.masked_fill_(filtered_logits < min_values, float('-inf'))
#
#     # 应用top-p（核采样）过滤（如果需要）
#     if top_p > 0.0:
#         # 优化：如果top_k已经应用且k很小，可以跳过top_p
#         if top_k > 0 and top_k < 50:  # 如果已经过滤到很少的token
#             return filtered_logits
#
#         # 只对需要应用top-p过滤的情况进行排序
#         sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
#
#         # 计算softmax和累积概率
#         # 使用更高效的方法计算softmax
#         max_logits = torch.max(sorted_logits, dim=-1, keepdim=True)[0]
#         exp_logits = torch.exp(sorted_logits - max_logits)
#         exp_sum = torch.cumsum(exp_logits, dim=-1)
#         exp_total = exp_sum[:, -1].unsqueeze(-1)
#         cumulative_probs = exp_sum / exp_total
#
#         # 创建要移除的token的掩码
#         remove_tokens = cumulative_probs > top_p
#
#         # 确保至少保留一个token
#         if remove_tokens.size(1) > 1:
#             remove_tokens[:, 1:] = remove_tokens[:, :-1].clone()
#         remove_tokens[:, 0] = 0
#
#         # 应用掩码到排序后的logits
#         # 使用更高效的方法应用掩码
#         indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
#         indices_to_remove.scatter_(1, sorted_indices, remove_tokens)
#         filtered_logits.masked_fill_(indices_to_remove, float('-inf'))
#
#     return filtered_logits
#
#
# # 添加缓存字典和缓存大小限制
# _logits_cache = {}
# _CACHE_SIZE_LIMIT = 10  # 限制缓存大小，避免内存泄漏
#
# def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
#     """
#     Args:
#         logits (torch.Tensor): shape (1, vocab)
#         temperature (float): temperature
#         top_k (float): top_k
#         top_p (float): top_p
#
#     Returns:
#         torch.Tensor: next token with shape as (batch,  1)
#     """
#     assert logits.dim() == 2
#     start_time = time.perf_counter()  # Start timing
#
#     # 检查缓存
#     # 使用参数和logits的哈希作为缓存键
#     # 注意：我们只对小批量的logits使用缓存，以避免哈希计算开销超过收益
#     if logits.size(0) == 1 and logits.size(1) < 10000:
#         cache_key = (temperature, top_k, top_p, logits.sum().item(), logits[0, 0].item())
#         if cache_key in _logits_cache:
#             probs, confidence = _logits_cache[cache_key]
#             elapsed = time.perf_counter() - start_time
#             print(f"norm_logits executed in {elapsed:.6f} seconds (cache hit)")
#             return probs, confidence
#
#     # 超快路径：如果不需要过滤且温度为1.0
#     if top_k <= 0 and top_p <= 0.0 and abs(temperature - 1.0) < 1e-6:
#         # 直接计算softmax，避免不必要的复制和过滤
#         with torch.no_grad():  # 使用no_grad加速计算
#             probs = F.softmax(logits, dim=1)
#         elapsed = time.perf_counter() - start_time
#         print(f"norm_logits executed in {elapsed:.6f} seconds (fast path)")
#
#         # 对于小批量，更新缓存
#         if logits.size(0) == 1 and logits.size(1) < 10000:
#             # 限制缓存大小
#             if len(_logits_cache) >= _CACHE_SIZE_LIMIT:
#                 # 移除随机键以避免缓存过大
#                 _logits_cache.pop(next(iter(_logits_cache.keys())))
#             _logits_cache[cache_key] = (probs, probs)
#
#         return probs, probs
#
#     # 处理零温度
#     if temperature == 0:
#         temperature = 1e-8
#
#     # 使用torch.no_grad()加速计算，因为我们不需要梯度
#     with torch.no_grad():
#         # 应用top_k和top_p过滤
#         filtered_logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
#
#         # 根据温度是否为1.0优化计算
#         if abs(temperature - 1.0) < 1e-6:
#             # 如果温度接近1.0，我们可以计算一次并重用
#             probs = F.softmax(filtered_logits, dim=1)
#             confidence = probs  # 直接引用，不创建新张量
#         else:
#             # 需要温度缩放和单独计算
#             # 先计算confidence（使用未缩放的logits）
#             confidence = F.softmax(filtered_logits, dim=1)
#             # 然后计算probs（使用缩放后的logits）
#             # 使用原地操作进行温度缩放
#             scaled_filtered_logits = filtered_logits / temperature
#             probs = F.softmax(scaled_filtered_logits, dim=1)
#
#     elapsed = time.perf_counter() - start_time
#     print(f"norm_logits executed in {elapsed:.6f} seconds")
#
#     # 对于小批量，更新缓存
#     if logits.size(0) == 1 and logits.size(1) < 10000:
#         # 限制缓存大小
#         if len(_logits_cache) >= _CACHE_SIZE_LIMIT:
#             # 移除随机键以避免缓存过大
#             _logits_cache.pop(next(iter(_logits_cache.keys())))
#         _logits_cache[cache_key] = (probs, confidence)
#
#     return probs, confidence
#
#
# def sample(probs: torch.Tensor, num_samples: int = 1):
#     idx_next = torch.multinomial(probs, num_samples=num_samples)
#     return idx_next
#
#
# def max_fn(x):
#     """
#         norm(max (x, 0))
#     """
#     x_max = torch.where(x > 0, x, torch.zeros_like(x))
#     x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
#     return x_max / x_max_sum
#
# # def make_mask(past_len, seq_len, device, dtype=torch.Bfloat16):
# #     """创建因果掩码，用于控制模型在生成时可以看到的token
# #
# #     Args:
# #         past_len (int): 已经生成的token长度
# #         seq_len (int): 序列长度
# #         device: 设备
# #         dtype: 数据类型，默认为torch.float32
# #
# #     Returns:
# #         torch.Tensor: 因果掩码
# #     """
# #     min_dtype = torch.finfo(dtype).min
# #     causal_mask = torch.full((seq_len, past_len + seq_len), fill_value=min_dtype, dtype=dtype, device=device) # full of min
# #     causal_mask[:, :past_len+1] = 0 # set 0 for past seen tokens and the first ground truth token
# #     diag_idx = torch.arange(seq_len, device=device) # generate diag index
# #     causal_mask[diag_idx, past_len + diag_idx] *= False # set 0 for draft tokens
# #     causal_mask = causal_mask[None,None,:,:].expand(1,1,-1,-1)
# #     return causal_mask

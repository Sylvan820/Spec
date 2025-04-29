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
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-3b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "qwen2.5-7b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "qwen2.5-14b": "/remote-home/security_shenyuhao/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5-72b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
    }

    args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--data_path', type=str,
                        default="data")
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
    parser.add_argument('--top_k', type=int, default=3, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    parser.add_argument('--branches', type=int, default=3, help='branchs.')
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
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

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
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
    if temperature == 0:
        idx = torch.argmax(logits, dim=-1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        for i in range(logits.size(0)):
             new_logits[i, idx[i]] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample_greedy(probs: torch.Tensor, num_samples: int = 1):
    # 使用topk获取概率最高的前k个值的索引
    _, idx_next = torch.topk(probs, k=num_samples, dim=-1)
    if num_samples > 1:
        if len(idx_next) > num_samples:
            idx_next = idx_next[:num_samples]
    return idx_next

def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next



def calculate_processed_entropy(probs):
    # 确保概率非零且归一化
    # probs = probs / probs.sum()
    entropy = torch.sum(-(probs * torch.log(probs)),dim=-1)
    # 处理熵值：1 - sqrt(0.15 * entropy)
    entropy = 1 - torch.sqrt(0.2 * entropy)
    return entropy

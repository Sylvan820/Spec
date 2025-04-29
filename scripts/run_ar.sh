#Qwen 1.5b 32b Vanilla

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_qwen_1532 --draft_model qwen2.5-32b --target_model qwen2.5-32b --max_tokens 512 --temp 0 

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode small -n 1  -e g_AR_qwen_1532 --draft_model qwen2.5-32b --target_model qwen2.5-32b --max_tokens 512 --temp 0    

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_cnndm.py --eval_mode small -n 1  -e cn_AR_qwen_1532 --draft_model qwen2.5-32b --target_model qwen2.5-32b --max_tokens 512 --temp 0

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_specbench.py --eval_mode small -n 1  -e sp_AR_qwen_1532 --draft_model qwen2.5-32b --target_model qwen2.5-32b --max_tokens 512 --temp 0



# # Vicuna 68m 13b Vanilla

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode small -n 1  -e H_AR_vicuna_13b --draft_model vicuna-13b --target_model vicuna-13b --max_tokens 512 --temp 0 

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode small -n 1  -e g_AR_vicuna_13b --draft_model vicuna-13b --target_model vicuna-13b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode small -n 1  -e mt_AR_vicuna_13b --draft_model vicuna-13b --target_model vicuna-13b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_specbench.py --eval_mode small -n 1  -e sp_AR_vicuna_13b --draft_model vicuna-13b --target_model vicuna-13b --max_tokens 512 --temp 0

# LLama 68m7b Vanilla

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_mt_bench.py --eval_mode small -n 1  -e mt_AR_llama_7b --draft_model llama-7b --target_model llama-7b --max_tokens 512 --temp 0 

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 benchmark/eval_mgsm.py --eval_mode small -n 1  -e mg_AR_llama_7b --draft_model llama-7b --target_model llama-7b --max_tokens 512 --temp 0 

# Qwen 1.5b 32b speculative decoding

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 4 -n 1  -e H_SD_qwen_1532 --draft_model qwen2.5-1.5b --target_model qwen2.5-32b --max_tokens 512 --temp 0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode sd --gamma 4 -n 1  -e g_SD_qwen_1532 --draft_model qwen2.5-1.5b --target_model qwen2.5-32b --max_tokens 512 --temp 0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_specbench.py --eval_mode sd --gamma 4 -n 1  -e sp_SD_qwen_1532 --draft_model qwen2.5-1.5b --target_model qwen2.5-32b --max_tokens 512 --temp 0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_cnndm.py --eval_mode sd --gamma 4 -n 1  -e cn_SD_qwen_1532 --draft_model qwen2.5-1.5b --target_model qwen2.5-32b --max_tokens 512 --temp 0






# # # Vicuna 68m 13b speculative decoding
# # CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 8 -n 1  -e H_SD_vicuna_68m13b --draft_model vicuna-68m --target_model vicuna-13b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode sd --gamma 8 -n 1 -e G_SD_vicuna_68m13b --draft_model vicuna-68m --target_model vicuna-13b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_specbench.py --eval_mode sd --gamma 8 -n 1  -e sp_SD_vicuna_68m13b --draft_model vicuna-68m --target_model vicuna-13b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_mgsm.py --eval_mode sd --gamma 8 -n 1  -e Mg_SD_vicuna_68m13b --draft_model vicuna-68m --target_model vicuna-13b --max_tokens 512 --temp 0

# # LlaMa 68m 7b speculative decoding

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_humaneval.py --eval_mode sd --gamma 7 -n 1  -e H_SD_llamma_68m7b --draft_model llama-68m --target_model llama-7b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_gsm8k.py --eval_mode sd --gamma 7 -n 1  -e g_SD_llamma_68m7b --draft_model llama-68m --target_model llama-7b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_specbench.py --eval_mode sd --gamma 7 -n 1  -e sp_SD_llamma_68m7b --draft_model llama-68m --target_model llama-7b --max_tokens 512 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 benchmark/eval_cnndm.py --eval_mode sd --gamma 7 -n 1  -e cn_SD_llamma_68m7b --draft_model llama-68m --target_model llama-7b --max_tokens 512 --temp 0

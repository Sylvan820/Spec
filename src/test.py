from src.engine import Decoding
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--draft_model', type=str, default='/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-68m/snapshots/964a5d77df908b69f8d6476fb70e940425b04cb5')
argparser.add_argument('--target_model', type=str, default='/remote-home/security_shenyuhao/huggingface/hub/models--JackFram--llama-160m/snapshots/aca9b687d1425f863dcf5de9a4c96e3fe36266dd')
argparser.add_argument('--eval_mode', type=str, default='para_sd')
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--temp', type=float, default=0.6)
argparser.add_argument('--top_k', type=int, default=10)
argparser.add_argument('--top_p', type=float, default=0.9)
argparser.add_argument('--max_tokens', type=int, default=128)
argparser.add_argument('--gamma', type=int, default=5)
argparser.add_argument('--vocab_size', type=int, default=128256)
argparser.add_argument('--branches', type=int, default=3)
args = argparser.parse_args()

decoding = Decoding(args)
decoding.load_model()
decoding.load_tokenizer()

prompt = "What is the meaning of life?"
input_ids = decoding.tokenizer(prompt, return_tensors='pt').input_ids
output_ids = decoding.branch_speculative_decoding(input_ids)
output = decoding.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)
# Output: "The meaning of life is 42."
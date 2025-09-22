from memDec import MemoryDecoder

import transformers
from transformers import AutoModelForCausalLM
from loguru import logger

base_lm_path = "/fs-computility/plm/shared/jqcao/models/gpt2/gpt2-xl"
knn_generator_path = "/fs-computility/plm/shared/jqcao/projects/MemoryDecoder/checkpoint/memdec-gpt2-small"

tokenizer = transformers.AutoTokenizer.from_pretrained(base_lm_path)
base_lm = AutoModelForCausalLM.from_pretrained(base_lm_path)
knn_generator = AutoModelForCausalLM.from_pretrained(knn_generator_path)

base_lm.resize_token_embeddings(len(tokenizer))
knn_generator.resize_token_embeddings(len(tokenizer))
base_lm.eval()
knn_generator.eval()

joint = MemoryDecoder(base_lm, knn_generator, lmbda=0.55, knn_temp=1.0).to("cuda")

prompt = f"As with previous Valkyira Chronicles games , Valkyria Chronicles III is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

out_ids = joint.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Memory Decoder output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected output: As with previous Valkyira Chronicles games , Valkyria Chronicles III is a role @-@ playing video game developed by Sega and published by Sega for the PlayStation 2 .

out_ids = base_lm.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Base Model output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected output: As with previous Valkyira Chronicles games , Valkyria Chronicles III is a turn-based strategy game. The player takes control of a squad of Valkyria soldiers,
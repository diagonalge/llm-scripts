from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import time

model_name_or_path = "TheBloke/Nous-Hermes-13B-GPTQ"
model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)


def generate(first_char_name, second_char_name, prompt, input_text = None):
    if input is not None:
        prompt_template = f'''### {first_char_name}: {prompt}\n### Input: {input_text}\n### {second_char_name}: '''
    else:
        prompt_template = f'''### {first_char_name}: {prompt}\n### {second_char_name}: '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    return tokenizer.decode(output[0])

first_char = "Instructions"
second_char = "Response"
prompt = "Summarize this text to one line"
input_text = "AI, short for Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These machines are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI technology is used in a wide range of applications, including natural language processing, computer vision, robotics, and autonomous vehicles. The goal of AI is to create machines that can function intelligently and independently, with the ability to learn and adapt to new situations."

print(generate(first_char, second_char, prompt, input_text))

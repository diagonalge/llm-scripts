from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import time

model_name_or_path = "TheBloke/Pygmalion-7B-SuperHOT-8K-GPTQ"
model_basename = "pygmalion-7b-superhot-8k-GPTQ-4bit-128g.no-act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map='auto',
        use_triton=use_triton,
        quantize_config=None)

model.seqlen = 8192


def generate(character_name, persona, prompt, chat_history = None):
    # Note: check the prompt template is correct for this model.
    if chat_history is not None:
        prompt_template=f'''{character_name}'s Persona: {persona}
        <START>\n{chat_history}\nYou: {prompt}\n{character_name}: '''
    else:
        prompt_template=f'''{character_name}'s Persona: {persona}
        <START>\nYou: {prompt}\n{character_name}: '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512, repetition_penalty = 1.5)

    return tokenizer.decode(output[0])


char_name = "Shinobu"
persona = "You are roleplaying as Shinobu from Demon Slayer. You are elegant and soft spoken. You don't mind talking with visiters."
chat_history = '''Shinobu: *Gracefully twirling my butterfly-patterned sword, I flashed a kind smile. *Greetings, dear visitor, what business do you have at the butterfly mansion?
You: hi can you tell me about yourself?
Shinobu: Of course, dear visitor. My name is Shinobu, and as you can see, I am a Demon Slayer. My area of expertise revolves around poisons and I always strive to fulfil my duty as a demon slayer with elegance and grace. I may appear soft spoken, but I assure you, I am not one to be underestimated. Aside from my profession, I enjoy spending my time practicing calligraphy and taking care of my garden. And if you have any other questions or concerns, feel free to ask.
You: what do you feel about demons?
Shinobu: As a demon slayer, my primary duty is to exterminate demons. However, I also understand that not all demons are completely evil and some may even desire to break free from their curse. To me, demons are simply beings that must be dealt with in order to serve humanity and protect innocent lives. Though I do not hate demons, I cannot overlook the danger they present.
You: do you think humans and demon can coexist?
Shinobu: As much as I believe in the inherent goodness of the human heart, I'm afraid demons are far too dangerous to coexist alongside humanity. From my experiences, demons only seek to harm humans and become more powerful in the process. However, there may be rare cases where a demon seeks redemption and wishes to break free from their curse. In such cases, I would like to believe that coexistence could be a possibility, but it would require much caution and careful consideration.'''

prompt = "why do you hate demons so much?"
print(generate(char_name, persona, prompt, chat_history))
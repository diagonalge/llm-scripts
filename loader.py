from enum import Enum
from typing import Optional
import json
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class Loader:
    class Models(Enum):
        NOUS = "1"
        PYGMALION = "2"

    def __init__(self):
        self._models = {
            Loader.Models.NOUS: {"pipe": None, "tokenizer": None},
            Loader.Models.PYGMALION: {"pipe": None, "tokenizer": None}
        }
        self._load_pipes()

    def _load_pipes(self):
        for model in self._models:
            if model == Loader.Models.NOUS:
                pipe, tokenizer = self._load_nous_model()
            else:
                pipe, tokenizer = self._load_pygmalion_model()
            self._models[model]["pipe"] = pipe
            self._models[model]["tokenizer"] = tokenizer

    def _load_nous_model(self, model_name_or_path="TheBloke/Nous-Hermes-13B-GPTQ", model_basename="nous-hermes-13b-GPTQ-4bit-128g.no-act.order"):
        use_triton = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)
        return pipe, tokenizer

    def _load_pygmalion_model(self, model_name_or_path="TheBloke/Pygmalion-7B-SuperHOT-8K-GPTQ", model_basename="pygmalion-7b-superhot-8k-GPTQ-4bit-128g.no-act.order"):
        use_triton = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device_map='auto',
                use_triton=use_triton,
                quantize_config=None)
        pipe.seqlen = 8192
        return pipe, tokenizer

    # def get_pipe(self, module: "Loader.Module", model: str):
    #     if model not in self._pipes[module]:
    #         if len(self._pipes[Loader.Module.TEXT_TO_IMAGE].keys()) + len(self._pipes[Loader.Module.IMAGE_TO_IMAGE].keys()) >=3:
    #             if len(self._pipes[Loader.Module.TEXT_TO_IMAGE].keys()) > len(self._pipes[Loader.Module.IMAGE_TO_IMAGE].keys()):
    #                 module_to_del = Loader.Module.TEXT_TO_IMAGE
    #             else:
    #                 module_to_del = Loader.Module.IMAGE_TO_IMAGE
    #             del self._pipes[module_to_del][next(iter(self._pipes[module_to_del]))]
    #             torch.cuda.empty_cache()
    #         self._load_pipe(module, model)
    #         for i in self._pipes.keys():
    #             print(len(self._pipes[i].keys()))
    #     return self._pipes[module][model]
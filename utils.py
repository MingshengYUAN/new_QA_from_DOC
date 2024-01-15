import json
import os.path as osp
from typing import Union
from sentence_transformers import SentenceTransformer
    
embedding_function = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda:0")

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_path: str = "", verbose: bool = False):
        self._verbose = verbose
        if not osp.exists(template_path):
            raise ValueError(f"Can't read {template_path}")
        
        with open(template_path) as fp:
            self.template = json.load(fp)

    def generate_prompt(self, question: str, context: str, prompt_serie) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template[prompt_serie].format(question=question, context=context)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()



import json
import os.path as osp
import xlrd
from typing import Union
from sentence_transformers import SentenceTransformer
from share_args import ShareArgs
from tqdm import tqdm
import configparser
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')
    
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

def add_qa_pairs(fragements, filename):
    # print(filename)
    # print(filename == "THE LINE - Fatigue Management Guideline")
    filelist = ["THE LINE Adverse Weather Working Plan", "THE LINE - HSW Delivery Plan", "THE LINE - Reward and Recognition Guideline", "THE LINE - Fatigue Management Guideline",
                "THE LINE - Worker Welfare Plan", "THE LINE Adverse Weather Working Plan", "THE LINE OH&H plan draft", "NEOM-NPR-STD-001_01.00 Projects Health and Safety Assurance Standard_Jan24"]
    filename = filename.strip('.txt').strip('  ').strip(' ')
    if filename in filelist:
        excel_path = f"./qa_pairs/{conf['application']['name']}/{filename}.xlsx"
        # excel_path = f"./qa_pairs/the_line/{filename}.xlsx"
    else:
        return fragements
    try:
        wb = xlrd.open_workbook(excel_path)
        qa_sheet = wb.sheets()[0]
        all_rows = qa_sheet.nrows
        for i in tqdm(range(all_rows), desc='ADD_QA'):
            if not i:
                continue
            if qa_sheet.cell(i, 4).value == 'N' or len(qa_sheet.cell(i, 4).value) > 2:
                continue
            tmp_fragement = {'fragement':f"{qa_sheet.cell(i, 0).value}|___|{qa_sheet.cell(i, 2).value}",
                            'searchable_text': qa_sheet.cell(i, 1).value,
                            'searchable_text_type': 'QA_pairs'}
            fragements.append(tmp_fragement)
    except:
        pass
    return fragements

# a = add_qa_pairs([], "THE LINE - Reward and Recognition Guideline")
# print(a[0])
import json
import os.path as osp
import xlrd
from typing import Union
from sentence_transformers import SentenceTransformer
from share_args import ShareArgs
from tqdm import tqdm
import six
from google.cloud import translate_v2 as translate
import langid
import configparser
import redis

conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

redis_pool = redis.ConnectionPool(
    host=str(conf['redis']['host']),
    port=3379,
    password='vitonguE@1@1',
    db=int(conf['redis']['db']),
    decode_responses=True)

redis_conn = redis.Redis(connection_pool=redis_pool)


embedding_function = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda:0")

#########################

langid.set_languages(['en', 'ar'])
def check_lang_id(text):
	if langid.rank(text)[0][0] == 'ar':
		return 'ar'
	elif langid.rank(text)[0][0] == 'en':
		return 'en'

#########################

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
    
    def generate_prompt_with_answer(self, question: str, context: str, answer: str, prompt_serie) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template[prompt_serie].format(question=question, context=context, answer=answer)
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

#######################  Google translate
 
def translate_text(target, text):
    """Translates text into the target language.
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
   
 
    translate_client = translate.Client()
 
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
 
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    
    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language:
    return result["translatedText"]

def save_redis(chatId, msgId, response, ifend):
    if ifend:
        message = {"chatId": chatId,"msgId": msgId, "response": "[\FINAL\]"}
    else:
        message = {"chatId": chatId,"msgId": msgId, "response": response}
    stream_name = f"momrah:sse:chat:{msgId}"
    message1_id = redis_conn.xadd(stream_name, message)
    # if ifend:
    #     redis_conn.expire(stream_name, 3600)
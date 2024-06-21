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
import numpy as np
import time
from FlagEmbedding import FlagReranker, BGEM3FlagModel
from log_info import logger
import requests
import numpy as np


conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

redis_pool = redis.ConnectionPool(
    host=str(conf['redis']['host']),
    port=3379,
    password='vitonguE@1@1',
    db=int(conf['redis']['db']),
    decode_responses=True)

redis_conn = redis.Redis(connection_pool=redis_pool)

# bge_m3_embedding_function = SentenceTransformer('BAAI/bge-m3', device="cuda:0")
embedding_function = SentenceTransformer(model_name_or_path="all-mpnet-base-v2")
# reranker = FlagReranker('BAAI/bge-reranker-v2-m3')
# reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
# embedding_function = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda:0")
# bge_m3_embedding_function = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

#########################

# Agent service to determine if the fragment include the answer for question
def llm_relative_determine(fragement_candidates, question):
    if isinstance(fragement_candidates[0], list):
        fragement_candidates = fragement_candidates[0]
    logger.info(f"llm_relative_determine: {fragement_candidates}")

    fragement_candidates_res = []
    for i in fragement_candidates:
        messages = [{"role":"system", "content": """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. Do not give the reason."""},
                    {"role":"user", "content":f"{i} \n\nUser question: {question}"}]
        response = requests.post(
	'http://192.168.0.69:3070/v1/chat/completions',   
	    json = {'messages': messages, "model": "../../../Meta-Llama-3-70B-Instruct-hf/", "max_tokens": 32, "stream": False, "temperature": 0.0, "stop_token_ids": [128009]}).json()
        try:
            res = response['choices'][0]['message']['content']
            if res == 'yes' or 'yes' in res:
                logger.info(f"YES!")
                fragement_candidates_res.append(i)
            else:
                logger.info(f"Question: {question} \nUnrelated Fragment: {res}")
        except:
            logger.info(f"Determine ERROR: {response}")
            continue
    return fragement_candidates_res

# Reranker API, can be set in this project if the cuda version has been updated 
def retrieve_top_fragment(fragement_candidates, question, top_k=1):
    # print(f"fragement_candidates:{fragement_candidates},question:{question}, top_k:{top_k}")
    if isinstance(fragement_candidates[0], str):
        fragement_candidates = [fragement_candidates]
    
    tmp_res = requests.post('http://192.168.0.151:3090/retrieve_top_fragment', json={"fragement_candidates":fragement_candidates[0],"question":question, "top_k":top_k}).json()
    retrieve_use_time = tmp_res['retrieve_use_time']
    
    res = tmp_res['res']
    top_index = tmp_res['top_index']
    top_score = tmp_res['top_score']
    logger.info(f"Retrieve use time: {retrieve_use_time}")
    return res, top_index, top_score

# Embedding API, can be set in this project if the cuda version has been updated 
def bge_m3_embedding_function(texts):
    embedding_vectors = requests.post('http://192.168.0.151:3090/bge_m3_embedding', json={"texts":texts})
    return embedding_vectors

# def retrieve_top_fragment(fragement_candidates, question, top_k=1):
#     retrieve_start = time.time()
#     fragement_question_pairs = [[i[0], question] for i in fragement_candidates]
#     print(f"fragement_question_pairs: {fragement_question_pairs}")
#     scores = reranker.compute_score(fragement_question_pairs, normalize=True)
#     top_index = np.argsort(-np.array(scores))
#     res = [fragement_candidates[top_index[i]] for i in range(top_k)]
#     retrieve_use_time = time.time() - retrieve_start
#     logger.info(f"Retrieve use time: {retrieve_use_time}")
#     return res

#########################

langid.set_languages(['en', 'ar'])
def check_lang_id(text):
	if langid.rank(text)[0][0] == 'ar':
		return 'ar'
	elif langid.rank(text)[0][0] == 'en':
		return 'en'

#########################

# Generate the prompt in selected prompt form
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

# Old FAQ functions to save FAQ in this repo, Has replaced by the standard semantic search API
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
 
# Frank translate API
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
    
    return result["translatedText"]

# old stream flow, save message to redis, not in use
def save_redis(chatId, msgId, response, ifend):
    if ifend:
        message = {"chatId": chatId,"msgId": msgId, "response": "[\FINAL\]"}
    else:
        message = {"chatId": chatId,"msgId": msgId, "response": response}
    stream_name = f"momrah:sse:chat:{msgId}"
    message1_id = redis_conn.xadd(stream_name, message)
    # if ifend:
    #     redis_conn.expire(stream_name, 3600)

# Create OpenAI batch form prompt for LLM API
def create_mixtral_messages_prompt(messages, question):

    prompter = Prompter('./prompt.json')
    final_prompt = ''
    if len(messages) == 0:
        prompt = '<s> [INST] ' + prompter.generate_prompt_with_answer(question=question, context='', answer='', prompt_serie='chat_standard') + '[/INST]'
        final_prompt += prompt
    else:
        user_former_input, assistant_former_answer = '', ''
        for num, i in enumerate(messages):
            if i['role'] == 'user':
                user_former_input = i['content']
            else:
                assistant_former_answer = i['content']
            if user_former_input != '' and assistant_former_answer !='':
                ## 1st round with system prompt
                if num - 1 == 0:
                    prompt = '<s> [INST] ' + prompter.generate_prompt_with_answer(question=user_former_input, context='', answer=assistant_former_answer, prompt_serie='chat_standard') + '[/INST]'
                    final_prompt += prompt
                ## After 1st round no system prompt needed, just question + context + 'Answer:'
                else:
                    prompt = '[INST]' + prompter.generate_prompt_with_answer(question=user_former_input, context='', answer=assistant_former_answer, prompt_serie='chat_standard') + '[/INST]'
                    final_prompt += prompt
                user_former_input, assistant_former_answer = '', ''
                ## Add final symbol '<\s>' if ended or add '\n' 
            if num == len(messages) - 1:
                final_prompt += '<\s>'
            else:
                final_prompt += '\n'
        prompt = prompter.generate_prompt_with_answer(question=question, context='', answer='', prompt_serie='chat')
        final_prompt += prompt
    return final_prompt

# Create OpenAI form messages for LLM workflow
def create_mixtral_messages(messages, question):
    prompter = Prompter('./prompt.json')
    llm_messages = []
    if len(messages) == 0:
        prompt = prompter.generate_prompt(question=question, context='', prompt_serie='chat_standard')
        llm_messages.append({"role": "user", "content":prompt})
    else:
        for num, i in enumerate(messages):
            if i['role'] == 'user':
                user_former_input = i['content']
            else:
                assistant_former_answer = i['content']
            if num % 2:
                prompt = prompter.generate_prompt(question=user_former_input, context='', prompt_serie='chat_standard')
                llm_messages.append({"role": "user", "content":prompt})
        llm_messages.append({"role": "assistant", "content":assistant_former_answer})

    return llm_messages
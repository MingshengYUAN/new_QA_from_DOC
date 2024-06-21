import time
import re 
import os
from obs import GetObjectHeader
from obs import ObsClient
import traceback
import io
import copy
import requests
import logging
import argparse
import configparser
import numpy as np
from datetime import timedelta
from flask import Flask, request
from flasgger import Swagger
from flasgger.utils import swag_from
from swagger_template import template
from log_info import *
from share_args import ShareArgs, LLMArgs
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# headers = {
#   "Content-Type": "application/json",
#   "Authorization": f"Bearer sk-uHLvvd0Mqyo7Cu3M05DeA4Ca0e6d49A49025197100AfF7Ff"
# }

ak = "UERNBNVSGD0C638VBTY5"
sk = "7jVUYOwxvZEoRNtoo0Oq6uLwfXBTuxT8BBpeBqon"
server = "obs://aramus-llm/aramus-qa/upload/default/"
bucketName="examplebucket"
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)

executor = ThreadPoolExecutor(max_workers=20)

parser = argparse.ArgumentParser()
parser.add_argument('--port', default=3010)
parser.add_argument('--config_path', default='./conf/config_gov.ini')
args = parser.parse_args()
args_default = vars(args)
ShareArgs.update(args_default)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./conf/google_translate_frank.json"

from utils import translate_text, check_lang_id, save_redis, redis_conn, create_mixtral_messages_prompt, create_mixtral_messages
from document_qa_new import build_rag_chain_from_text, answer_from_doc, empty_collection, del_select_collection
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

app = Flask(__name__)
swagger = Swagger(app, template=template)

log_path = f"./log/qa_from_doc_{conf['application']['name']}.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(fh)

idk_threshold = float(conf['threshold']['idk']) if 'threshold' in conf else 0.4
# print(f"idk_threshold: {idk_threshold}")
save_folder = 'uploaded'

try:
	os.system(f"mkdir {save_folder}")
except:
	pass
try:
	os.system(f"mkdir {save_folder}/{conf['application']['name']}")
except:
	pass

save_folder = os.path.join(save_folder, conf['application']['name'])

#####################

@app.route('/empty_collection', methods=['POST'])
def do_empty_collection():
    start = time.time()
    data = request.get_json()
    collection_name = data['collection_name']
    
    name_list = empty_collection(collection_name)
    return {"response": f"Success delete following collection : {name_list}", "status":'success', "running_time": float(time.time() - start)}

#####################

def exec_doc_input(data):
    # File upload status : START
    stream_name = f"file_upload_status"
    # try:
    if '.txt' not in data['filename']:
        filename = data['filename']+".txt"
    save_path = os.path.join(save_folder, data['filename'])
    try:
        # data['text'] is the file_path in the OBS
        with open(save_path, mode='w') as w:
            w.write(data['text'])
        logger.info(f"Save text status: Save Success")
        logger.info(f"Save OBS path: {save_path}")
    except Exception as e:
        logger.info(f"Save text Error: {e}")

    if 'level' in data:
        rag_chain =  build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename = filename, level=level, file_path = data['file_path'] if 'file_path' in data.keys() else 'None', file_type = data['file_type'] if 'file_type' in data.keys() else 'None')
    else:
        rag_chain =  build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename = filename, file_path = data['file_path'] if 'file_path' in data.keys() else 'None', file_type = data['file_type'] if 'file_type' in data.keys() else 'None')

    # File upload status : Success
    message1_id = redis_conn.xadd(stream_name, {"token_name":data['token_name'], "status":'Success'})
    
    logger.info(f"Save chromadb status: Save Success")
    logger.info(f"Save name: {data['filename']}")
    logger.info(f"Save unique token name: {data['token_name']}")

    # except Exception as e:
    #     # File upload status : Fail
    #     message1_id = redis_conn.xadd(stream_name, {"token_name":data['token_name'], "status":'Fail'})


    #     logger.info(f"Save chromadb Error: {e}")


@app.route("/doc_input", methods=['POST'])
def doc_input():
    start = time.time()
    data = request.get_json()
    executor.submit(exec_doc_input, data=data)
    return {"response": "Start processing", "status": "Start", "running_time": float(time.time() - start)}


###########################

@app.route("/doc_delete", methods=['POST'])
def doc_delete():
    start = time.time()
    token_name = request.form.get('token_name')
    response = del_select_collection(token_name)
    return {"response": response, "status": "Success!", "running_time": float(time.time() - start)}

##########################

@app.route("/qa_from_doc", methods=['POST'])
def qa_from_doc():
    start = time.time()
    data = request.get_json()
    try:
        history_qa = data['messages']
    except:
        history_qa = []
    new_question = data['question']

    # stream parameter 
    stream = True if not data.get('stream') else data.get('stream')
    chat_id = 0 if not data.get('id') else data.get('id')
    msg_id = 0 if not data.get('msg_id') else data.get('msg_id')
    logger.info(f"Stream: {stream} | Chat_id: {chat_id} | Msg_id: {msg_id}")

    # condense question
    condense_messages = []    
    if not history_qa:
        condense_messages = [{"role":"system", "content": "Do NOT answer the question. Accroding to the chat history, simply reformulate the latest question if needed or just return itself."}] 
    else: 
        condense_messages = copy.deepcopy(history_qa)
        condense_messages.insert(0, {"role":"system", "content": "Do NOT answer the question. Accroding to the chat history, simply reformulate the latest question if needed or just return itself."})
    condense_messages.append({"role":"user", "content":new_question})
    # print(f"condense_messages: !!!! {condense_messages} !!!!!!")
			
    response = requests.post(
        'http://192.168.0.69:3070/v1/chat/completions',   
        json = {'messages': condense_messages, "model": "../../../Meta-Llama-3-70B-Instruct-hf/", "max_tokens": 512, "stream": False, "temperature": 0.0,"stop_token_ids": [128009]})
    condense_question = response.json()['choices'][0]['message']['content']

    # FAQ, QA from DOC switches
    use_FAQ = True if not data.get('use_FAQ') else data.get('use_FAQ')
    use_QA_from_DOC = True if not data.get('use_QA_from_DOC') else data.get('use_QA_from_DOC')

    logger.info(f"Condense Question: {condense_question}")

    text_name = data['filename']
    logger.info(f"File token name: {text_name}")
    
    gather_question = ''
    if len(history_qa) > 3:
        history_qa = history_qa[len(history_qa)-4:-1]
    q_num = 0
    for i in history_qa:
        if i['role'] == 'user':
            if check_lang_id(i['content']) == 'ar' and conf['application']['name'] == 'gov':
                tmp = translate_text('en', i['content'])
            else:
                tmp = i['content']
            gather_question += f"Q{q_num + 1}: "+ tmp + '  '
            q_num += 1
    gather_question += "Last question: " + new_question


    # check QA pairs list
    # check exist
    if use_FAQ:
        exist_flag = requests.post(
                'http://192.168.0.16:3020/check_collection_exist',
                json = {
                "application_name":conf['application']['name'],
                }
            )
        if exist_flag:
            qa_pair_response = requests.post(
                    'http://192.168.0.178:3090/generate',
                    json = {
                    "question":question,
                    "application_name":conf['application']['name'],
                    "threshold_score":0.9,
                    }
                )
            if qa_pair_response['response'] != "Score not meet the threshold":
                save_redis(chat_id, msg_id, qa_pair_response['response'], 0)
                save_redis(chat_id, msg_id, '', 1)
                return {"response":'' , "fragment": '', "score":1.0, "document_name": '' , "status": "Success!", "running_time": float(time.time() - start)}
            
    logger.info(f"Question: {new_question}")

    pre_prompt = '' if not data.get('pre_prompt') else data.get('pre_prompt')
    logger.info(f"pre_prompt: {pre_prompt}")

    if use_QA_from_DOC:
        if 'level' in data.keys() and data['level']:
            # Get response
            response, fragment, score, document_name, llm_messages = answer_from_doc(token_name=text_name, gather_question=gather_question, question=new_question,
                                                                        msg_id=msg_id, chat_id=chat_id, condense_question=condense_question, pre_prompt=pre_prompt,
                                                                        messages=history_qa, stream=stream, level=data['level'])
        else:
            response, fragment, score, document_name, llm_messages = answer_from_doc(token_name=text_name, gather_question=gather_question, question=new_question,
                                                                        msg_id=msg_id, chat_id=chat_id, condense_question=condense_question, pre_prompt=pre_prompt,
                                                                        stream=stream, messages=history_qa)
        logger.info(f"Response mode: {response}")

        if document_name == '':
            try:
                document_name = fragment.split('|___|')[1].replace('.txt', '')
            except:
                pass
        # Normal generate mode if score < threshold
        if score and score < idk_threshold:
            response = "Workflow" 
            messages = create_mixtral_messages(history_qa, new_question)
            score = 0.0
            fragment = ''
            document_name = ''

        llm_args = LLMArgs.args
        if response != 'Workflow':
            # For redis form LLM QA generate, current not use
            return {"response": response, "fragment": fragment.split('|___|')[0], "score":score, "document_name": document_name , "messages": [], "llm_args": {}, "status": "Success!", "running_time": float(time.time() - start)}
        else:
            return {"response": '', "fragment": fragment.split('|___|')[0], "score":score, "document_name": document_name , "messages": llm_messages, "llm_args": llm_args, "status": "Success!", "running_time": float(time.time() - start)}

    else:
        messages = create_mixtral_messages(history_qa, new_question)
        return {"response": '', "fragment": '', "score":0.0, "document_name": '' , "messages": llm_messages, "llm_args": llm_args, "status": "Success!", "running_time": float(time.time() - start)}


#######################
if __name__=="__main__":
    app.run(port=args.port, host="0.0.0.0", debug=False)
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

executor = ThreadPoolExecutor()

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
    try:
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
            rag_chain =  build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename = filename, level=level, file_path = data['file_path'], file_type = data['file_type'])
        else:
            rag_chain =  build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename = filename, file_path = data['file_path'], file_type = data['file_type'])

        # File upload status : Success
        message1_id = redis_conn.xadd(stream_name, {"token_name":data['token_name'], "status":'Success'})
        
        logger.info(f"Save chromadb status: Save Success")
        logger.info(f"Save name: {data['filename']}")
        logger.info(f"Save unique token name: {data['token_name']}")

    except Exception as e:
        # File upload status : Fail
        message1_id = redis_conn.xadd(stream_name, {"token_name":data['token_name'], "status":'Fail'})


        logger.info(f"Save chromadb Error: {e}")


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
    print(f"messages: {data['messages']}")
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
    print(f"condense_messages: !!!! {condense_messages} !!!!!!")
			
    response = requests.post(
        'http://192.168.0.69:3070/v1/chat/completions',   
        json = {'messages': condense_messages, "model": "../../../Meta-Llama-3-70B-Instruct-hf/", "max_tokens": 512, "stream": False, "temperature": 0.0,"stop_token_ids": [128009]})
    condense_question = response.json()['choices'][0]['message']['content']

    # FAQ, QA from DOC switches
    use_FAQ = True if not data.get('use_FAQ') else data.get('use_FAQ')
    use_QA_from_DOC = True if not data.get('use_QA_from_DOC') else data.get('use_QA_from_DOC')

    # GPT4V parameter
    
    # image_input = 0 if not data.get('image_input') else data.get('image_input')
    # image_qa = 0
    # for i in history_qa:
    #     if type(i['content']) is not str:
    #         image_qa = 1
    # if image_input or image_qa:
    #     images_messages = history_qa.append({"role": "user", "content": new_question})
    #     payload = {"model": "gpt-4-vision-preview",
    #                 "messages": images_messages,
    #                 "max_tokens": 1024}
    #     response = requests.post("https://gpt.datapipe.top/v1/chta/completions", headers = headers, json=payload)

    #     return {"response":response.choices[0].message.content , "fragment": '', "score":0.0, "document_name": '' , "status": "Success!", "running_time": float(time.time() - start)}

    # Get condense question for retrieve fragment
    condense_question = ''
    condense_question_response = requests.post(
					# 'http://192.168.0.91:3072/generate',
					'http://localhost:3024/condense_question',
					json = {"user_input": data['question'], "messages": data['messages']}).json()
    condense_question = condense_question_response['response']
    logger.info(f"Condense Question: {condense_question} | Using time: {condense_question_response['running_time']}")

    # For <Gov> translate the arabic to en
    input_lang = 'en'
    
    if conf['application']['name'] == 'gov' :
        input_lang = check_lang_id(new_question)
        if input_lang == 'ar':
            new_question = translate_text("en", new_question)

    text_name = data['filename']
    logger.info(f"File token: {text_name}")
    
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
    # try:
    if use_QA_from_DOC:
        if 'level' in data.keys() and data['level']:
            response, fragment, score, document_name, llm_messages = answer_from_doc(token_name=text_name, gather_question=gather_question, question=new_question,
                                                                        msg_id=msg_id, chat_id=chat_id, condense_question=condense_question, 
                                                                        messages=history_qa, stream=stream, level=data['level'])
        else:
            response, fragment, score, document_name, llm_messages = answer_from_doc(token_name=text_name, gather_question=gather_question, question=new_question,
                                                                        msg_id=msg_id, chat_id=chat_id, condense_question=condense_question,
                                                                        stream=stream, messages=history_qa)
        logger.info(f"Response mode: {response}")
        
        
        # print(f"{response} | {fragment} | {score} | {document_name}")
        if document_name == '':
            try:
                document_name = fragment.split('|___|')[1].replace('.txt', '')
            except:
                pass

        if score and score < idk_threshold:
            response = "Workflow" 
            messages = create_mixtral_messages(history_qa, new_question)
            score = 0.0
            fragment = ''
            document_name = ''

        # if score and (response == "I don't know" or ("I don't know" in response and len(response) < 17) or score < idk_threshold):
        #     response = "I’m sorry I currently do not have an answer to that question, please rephrase or ask me another question." 
        #     score = 0.0
        #     fragment = ''
        #     document_name = ''

        
        # Gov translation part : NEED TO BE TRANSRORMED TO OPENAI FORM RETURN
        if conf['application']['name'] == 'gov' and input_lang == 'ar':
            response = translate_text('ar', response)
            fragment = translate_text('ar', fragment)
        # if len(response) > 5:
        #     save_redis(chat_id, msg_id, response, 0)
        #     save_redis(chat_id, msg_id, '', 1)
        #     response = ''
        llm_args = LLMArgs.args
        if response != 'Workflow':
            return {"response": response, "fragment": fragment.split('|___|')[0], "score":score, "document_name": document_name , "messages": [], "llm_args": {}, "status": "Success!", "running_time": float(time.time() - start)}
        else:
            return {"response": '', "fragment": fragment.split('|___|')[0], "score":score, "document_name": document_name , "messages": llm_messages, "llm_args": llm_args, "status": "Success!", "running_time": float(time.time() - start)}

    else:
        messages = create_mixtral_messages(history_qa, new_question)
        # response = requests.post(
		# 			# 'http://192.168.0.91:3072/generate',
		# 			'http://192.168.0.223:3074/generate',
		# 			json = {'prompt': prompt, 'max_tokens': 1024, 'temperature': 0.0, 'stream': stream, 'msg_id': msg_id, 'id':chat_id, 'application':conf['application']['name']}
		# 		).status_code
        # if response == 200:
        #     response = ''
        return {"response": '', "fragment": '', "score":0.0, "document_name": '' , "messages": llm_messages, "llm_args": llm_args, "status": "Success!", "running_time": float(time.time() - start)}


#######################
if __name__=="__main__":
    app.run(port=args.port, host="0.0.0.0", debug=False)
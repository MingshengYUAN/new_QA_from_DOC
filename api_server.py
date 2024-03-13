import time
import re
import os
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
from share_args import ShareArgs
from utils import translate_text, check_lang_id

parser = argparse.ArgumentParser()
parser.add_argument('--port', default=3010)
parser.add_argument('--config_path', default='./conf/config_gov.ini')
args = parser.parse_args()
args_default = vars(args)
ShareArgs.update(args_default)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./conf/google_translate_frank.json"
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
    return {"response": f"Success delete following collection : {name_list}", "status": 'success',
            "running_time": float(time.time() - start)}


#####################

@app.route("/doc_input", methods=['POST'])
def doc_input():
    start = time.time()
    data = request.get_json()
    if '.txt' not in data['filename']:
        filename = data['filename'] + ".txt"
    save_path = os.path.join(save_folder, data['filename'])
    try:
        with open(save_path, mode='w') as w:
            w.write(data['text'])
        logger.info(f"Save text status: Save Success")
        logger.info(f"Save path: {save_path}")
    except Exception as e:
        logger.info(f"Save text Error: {e}")

    # try:
    if 'level' in data:
        rag_chain = build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename=filename,
                                              level=level)
    else:
        rag_chain = build_rag_chain_from_text(token_name=data['token_name'], text=data['text'], filename=filename)

    logger.info(f"Save chromadb status: Save Success")
    logger.info(f"Save name: {data['filename']}")
    logger.info(f"Save unique token name: {data['token_name']}")
    return {"response": 'Save Success', "status": "Success!", "running_time": float(time.time() - start)}
    # except Exception as e:
    #     logger.info(f"Save chromadb Error: {e}")
    #     return {"response": "Save Error!", "status": "Fail!", "running_time": float(time.time() - start)}


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

    # print(f"data: {data}")
    input_lang = 'en'
    new_question = data['question']
    if conf['application']['name'] == 'gov':
        input_lang = check_lang_id(new_question)
        if input_lang == 'ar':
            new_question = translate_text("en", new_question)
    text_name = data['filename']
    logger.info(f"File token: {text_name}")
    try:
        history_qa = data['messages']
    except:
        history_qa = []

    # Streaming parameters
    stream = False if not data.get('stream') else data.get('stream')
    chat_id = 0 if not data.get('id') else data.get('id')
    msg_id = 0 if not data.get('msg_id') else data.get('msg_id')

    question = ''
    if len(history_qa) > 3:
        history_qa = history_qa[len(history_qa) - 4:-1]
    q_num = 0
    for i in history_qa:
        if i['role'] == 'user':
            if check_lang_id(i['content']) == 'ar' and conf['application']['name'] == 'gov':
                tmp = translate_text('en', i['content'])
            else:
                tmp = i['content']
            question += f"Q{q_num + 1}: " + tmp + '  '
            q_num += 1
    question += "Last question: " + new_question

    # check QA pairs list
    # check exist
    exist_flag = requests.post(
        'http://192.168.0.16:3020/check_collection_exist',  # mixtral 8x7B 15
        json={
            "application_name": conf['application']['name'],
        }
    )
    if exist_flag:
        qa_pair_response = requests.post(
            'http://192.168.0.178:3090/generate',  # mixtral 8x7B 15
            json={
                "question": question,
                "application_name": conf['application']['name'],
                "threshold_score": 0.9,
            }
        )
        if qa_pair_response['response'] != "Score not meet the threshold":
            return {
                "response": qa_pair_response['response'],
                "fragment": '',
                "score": 1.0,
                "document_name": '',
                "status": "Success!",
                "running_time": float(time.time() - start),
                "answer_origin": "qa_pair",
                "stream": False,
            }

    logger.info(f"Question: {question}")
    # try:
    if 'level' in data.keys():
        response, fragment, score, document_name, _origin, _stream = answer_from_doc(text_name, question, level)
    else:
        response, fragment, score, document_name, _origin, _stream = answer_from_doc(text_name, question)
    logger.info(f"Question Response: {response}")

    # print(f"{response} | {fragment} | {score} | {document_name}")
    if document_name == '':
        try:
            document_name = fragment.split('|___|')[1].replace('.txt', '')
        except:
            pass
    # print(f"Score : {score}")
    # if score:
    if score and (
            response == "I don't know" or ("I don't know" in response and len(response) < 17) or score < idk_threshold):
        response = "I’m sorry I currently do not have an answer to that question, please rephrase or ask me another question."
        score = 0.0
        fragment = ''
        document_name = ''
    # print(f"Score : {score}")
    if conf['application']['name'] == 'gov' and input_lang == 'ar':
        response = translate_text('ar', response)
        fragment = translate_text('ar', fragment)
    return {
        "response": response,
        "fragment": fragment.split('|___|')[0],
        "score": score,
        "document_name": document_name,
        "status": "Success!",
        "running_time": float(time.time() - start),
        "answer_origin": _origin,
        "stream": _stream,
    }
    # except Exception as e:
    #     logger.info(f"Answer question Error: {e}")
    #     return {"response": f"Error: {e}", "fragment": "", "status": "Fail!", "running_time": float(time.time() - start)}


#######################
if __name__ == "__main__":
    app.run(port=args.port, host="0.0.0.0", debug=False)

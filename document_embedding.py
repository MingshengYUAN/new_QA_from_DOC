import re 
import os 
import pandas as pd
import numpy as np
import requests
import stanza
from utils import embedding_function, bge_m3_embedding_function, check_lang_id
import torch
from tqdm import tqdm
from log_info import logger
from numpy.linalg import norm
import configparser
from share_args import ShareArgs
import xlrd
from obs import GetObjectHeader
from obs import ObsClient
import traceback
from io import StringIO
import csv
from tokenizers import Tokenizer
csv.field_size_limit(13107200)
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

torch.backends.cudnn.enabled = False

tokenizer = Tokenizer.from_file("/home/mingsheng.yuan/yma/LLM/LLAMA3/Meta-Llama-3-70B-Instruct-hf/tokenizer.json")

# stanza.download('ar')
nlp = stanza.Pipeline('en',processors='tokenize',device = "cuda:0", download_method=None)
nlp_ar = stanza.Pipeline('ar',processors='tokenize',device = "cuda:0", download_method=None)


ak = "UERNBNVSGD0C638VBTY5"
sk = "7jVUYOwxvZEoRNtoo0Oq6uLwfXBTuxT8BBpeBqon"
bucketName = "aramus-llm"
server = "http://obs.me-east-212.managedcognitivecloud.com"
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)


###########

def document_split(
	document_content,
	filename,
	fragment_window_size = 40,
	fragment_step_size = 4,
	sentence_left_context_size = 2,
	sentence_right_context_size = 5,
	file_type = '',
	file_path = ''
	):

	# filename
	filename = filename.replace('.txt', '').strip('  ').strip(' ')
	logger.info(f"filename: {filename} ｜ file_type: {file_type} | file_path: {file_path}")
	# print(file_type == '')

	max_tokens = int(conf.get("llm", "max_tokens"))
	temperature = float(conf.get("llm", "temperature"))
	if file_type == '':
		if '_ar' in filename or check_lang_id("\n".join(document_content.split('\n')[0:10])) == 'ar':
			logger.info("Stanza ar part start!")
			doc = nlp_ar(document_content)
		else:
			logger.info("Stanza en part start!")
			doc = nlp(document_content)
		logger.info("Stanza part finished!")
		sentences = [s.text for s in doc.sentences]
	elif file_type == 'pdf':
		sentences = []
		data = obsClient.getObject(bucketName, file_path, loadStreamInMemory=True).body.buffer
		data_io = StringIO(data.decode('utf-8'))
		csv_reader = csv.reader(data_io)
		for i in csv_reader:
			tmp_text = ''
			if i[1] != 'None':
				tmp_text += i[1]
			if i[2] != 'None':
				tmp_text += i[2]
			if i[3] != 'None':
				tmp_text += i[3]
			tmp_text += i[4]
			sentences.append(tmp_text)
		fragment_window_size = 4
		fragment_step_size = 2
		logger.info("PDF analyse finished!")
	elif file_type == 'word':
		sentences = []
		data = obsClient.getObject(bucketName, file_path, loadStreamInMemory=True).body.buffer
		data_io = StringIO(data.decode('utf-8'))
		csv_reader = csv.reader(data_io)
		for i in csv_reader:
			if i[1] == '' and i[2] == '' and i[3] == '' and i[4] == '':
				continue
			if i[5] == '':
				continue
			tmp_sentence = ''
			for j in range(1,5):
				if i[j] != '':
					tmp_sentence += i[j] + '\n'
			sentences.append(tmp_sentence)
		fragment_window_size = 4
		fragment_step_size = 2
		logger.info("WORD analyse finished!")
	elif file_type == 'excel':
		sentences = []
		logger.info("EXCEL analyse finished!")
		pass


	# prompts to questions
	output = []

	## prompts to get questions
	prompts = []

	# setence to fragements
	fragments = []


	start_sentence_idx = 0
	print(f"Sentence len : {len(sentences)}")
	if '_ar' in filename or check_lang_id("\n".join(document_content.split('\n')[0:10])) == 'ar':
		fragment_window_size = 20
	for i in tqdm(range(len(sentences)//fragment_step_size), desc='sentence level add'):
		tmp_ragment_window_size = fragment_window_size
		flag = True
		while flag:
			start_idx = i * fragment_step_size
			end_idx = i * fragment_step_size + tmp_ragment_window_size
			fragment = sentences[start_idx:end_idx]
			fragment = ' '.join(fragment)
			if len(tokenizer.encode(fragment).ids) > 5000:
				tmp_ragment_window_size -= 1
			else:
				flag = False
		fragments.append(fragment)
	flag = True
	final_sentences = len(sentences) - 1
	while flag:
		fragment = sentences[len(sentences)//fragment_window_size * fragment_step_size:final_sentences]
		fragment = ' '.join(fragment)
		if len(tokenizer.encode(fragment).ids) > 5000:
			final_sentences -= 1
		else:
			flag = False
	fragments.append(fragment)
	if final_sentences < len(sentences) -1:
		fragment = sentences[final_sentences:len(sentences) -1]
		fragment = ' '.join(fragment)
		fragments.append(fragment)

	print(f"fragements_len: {len(fragments)}")


	# fragements to prompts
	for f in fragments:
		if check_lang_id(f) == 'en':
			prompt = f"""
		<<SYS>> Read the following document and generate questions from its content. The question should be self-contained and complete. The generated questions are in the format of:

		Q:
		Q:
		Q:
		...

		Do not generate answers. Do not include the phrase "according to the document..." or "according to the author" in the question. Each question should be shorter than 50 words. If the document is in arabic, you need to generate questions in arabic too.

		<</SYS>>

		document: {f}
		[/INST] Sure. Here are the generated questions:
			"""
		else:
			prompt = f"""
		<<SYS>> اقرأ الوثيقة التالية وقم بطرح الأسئلة من محتواها. يجب أن يكون السؤال قائما بذاته وكاملا. الأسئلة التي تم إنشاؤها هي في شكل:

     س:
     س:
     س:
     ...

     لا تولد إجابات. لا تدرج عبارة "حسب الوثيقة..." أو "حسب المؤلف" في السؤال. إذا كانت لغة المستند هي اللغة العربية، فستحتاج إلى إنشاء أسئلة باللغة العربية أيضًا. يجب أن يكون كل سؤال أقل من 50 كلمة.
	 <</SYS>>

		document: {f}
		[/INST] Sure. Here are the generated questions:
			"""
		# print(f"Tokenizer len :{len(tokenizer.encode(prompt).ids)}")
		prompts.append(prompt.strip())

	print(f"Len: {len(prompts)}")

	# Prevent batches from being too large
	if len(prompts)> 4100:
		stack_num = len(prompts) // 4009
		for i in range(stack_num):
			start = i * 4009
			end = (i + 1) * 4009 if (i+1)!=stack_num else -1
			response = requests.post(
			'http://192.168.0.69:3070/v1/completions',   
			json = {'prompt': prompts[start:end], 
					"model": "../../../Meta-Llama-3-70B-Instruct-hf/", 
					"max_tokens": max_tokens, 
					"stream": False, 
					"temperature": temperature,
					"stop_token_ids": [128009]})

		for f, q in zip(
			fragments,
			response.json()['choices']
			):
			output.append({
				'fragement':f,
				'searchable_text':f,
				'searchable_text_type': 'fragement',
			})
			if 'Q:' in q['text']:
				for m in q['text'].split('Q:'):
					if len(m) < 3:
						continue
					output.append({
						'fragement':f,
						'searchable_text':m.replace('\n', ''),
						'searchable_text_type': 'fragment_question_by_llama3_70B',
					})
			else:
				for m in q['text'].split('س:'):
					if len(m) < 3:
						continue
					output.append({
						'fragement':f,
						'searchable_text':m.replace('\n', ''),
						'searchable_text_type': 'fragment_question_by_llama3_70B',
					})
	else:
		response = requests.post(
			'http://192.168.0.69:3070/v1/completions',   
			json = {'prompt': prompts,
					"model": "../../../Meta-Llama-3-70B-Instruct-hf/", 
					"max_tokens": max_tokens, 
					"stream": False, 
					"temperature": temperature,
					"stop_token_ids": [128009]})
		for f, q in tqdm(zip(
			fragments,
			response.json()['choices']
			), desc='add to output'):
			output.append({
				'fragement':f,
				'searchable_text':f,
				'searchable_text_type': 'fragement',
			})
			if 'Q:' in q['text']:
				for m in q['text'].split('Q:'):
					if len(m) < 3:
						continue
					output.append({
						'fragement':f,
						'searchable_text':m.replace('\n', ''),
						'searchable_text_type': 'fragment_question_by_llama3_70B',
					})
			else:
				for m in q['text'].split('س:'):
					if len(m) < 3:
						continue
					output.append({
						'fragement':f,
						'searchable_text':m.replace('\n', ''),
						'searchable_text_type': 'fragment_question_by_llama3_70B',
					})


	return output

###########

def document_embedding(token_name, documents, batch_size = 100):

	fragement_num = conf.get("fragement", "fragement_num")

	batch_num = len(documents)/batch_size
	j = 0

	texts = []
	for i in tqdm(documents, desc='documents embedding'):
		texts.append(i['searchable_text'])
	embedding_vectors = bge_m3_embedding_function(texts).json()
	for r in tqdm(documents, desc='set id'):
		r['searchable_text_embedding'] = embedding_vectors[j]
		r['id'] = token_name + "|__|" + str(hash(str([r['searchable_text'], r['fragement']])))
		j += 1
	return documents

def get_score(fragements, question):
	question_embedding = embedding_function.encode(question).tolist()
	similarity = 0
	if isinstance(fragements, str):
		fragements = [fragements]
	for i in fragements:
		tmp = embedding_function.encode(i).tolist()
		tmp_score = np.dot(tmp,question_embedding)/(norm(tmp)*norm(question_embedding))
		try:
			tmp_score = tmp_score[0]
		except:
			pass
		similarity = max(tmp_score, similarity)
	return similarity

###########

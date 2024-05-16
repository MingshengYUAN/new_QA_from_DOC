import re 
import os 
import pandas as pd
import numpy as np
import requests
import stanza
from utils import embedding_function, bge_m3_embedding_function
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

conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

torch.backends.cudnn.enabled = False

stanza.download('en')
nlp = stanza.Pipeline('en',processors='tokenize',device = "cuda:0")

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
	file_type = 'None',
	file_path = 'None',
	):
	max_tokens = int(conf.get("llm", "max_tokens"))
	temperature = float(conf.get("llm", "temperature"))
	if file_type == 'None':
		doc = nlp(document_content)
		logger.info("Stanza part finished!")
		sentences = [s.text for s in doc.sentences]
	elif file_type == 'pdf':
		sentence = []
		data = obsClient.getObject(bucketName, file_path, loadStreamInMemory=True).body.buffer
		data_io = StringIO(data.decode('utf-8'))
		csv_reader = csv.reader(data_io)
		for i in csv_reader:
			if i[1] != 'None':
				sentence.append(i[1]+'\n'+i[2])
		logger.info("PDF analyse finished!")

	# prompts to questions
	output = []

	## prompts to get questions
	prompts = []

	# setence to fragements
	fragments = []

	### FOR FM
	output.append({
				'fragement':document_content,
				'searchable_text':'',
				'searchable_text_type': 'All_texts',
			})

	if 'Introduction' in sentences and 'Summary of Tasks' in sentences:
		new_doc = nlp(document_content.split('Summary of Tasks')[0])
		sentences = [s.text for s in new_doc.sentences]
		summary_text = document_content.split('Summary of Tasks')[1].split('Tasks')[0]
		fragments.append(f"Summary of Tasks :{summary_text}")
		tasks = document_content.split('Tasks')[2].split('Legislation, Regulations, and Guidance')[0]
		# for i in ['Please lists all tasks', 'list all tasks', 'show me all tasks']:
		# 	output.append({
		# 		'fragement':tasks,
		# 		'searchable_text':i,
		# 		'searchable_text_type': 'All_tasks',
		# 	})
		for i in tasks.split('\n\n'):
			if len(i) < 3:
				continue
			fragments.append(i)
		if 'Legislation, Regulations, and Guidance' in document_content:
			last_part = document_content.split('Legislation, Regulations, and Guidance')[1]
			fragments.append(last_part)

	## For the line
	filelist = ["THE LINE Adverse Weather Working Plan", "THE LINE - HSW Delivery Plan", "THE LINE - Reward and Recognition Guideline", "THE LINE - Fatigue Management Guideline",
                "THE LINE - Worker Welfare Plan", "THE LINE Adverse Weather Working Plan", "THE LINE OH&H plan draft", "NEOM-NPR-STD-001_01.00 Projects Health and Safety Assurance Standard_Jan24"]
	# filelist = []
	for i in os.listdir('./m-split/the_line/new_knowledge_share'):
		filelist.append(i.replace('.txt', '').strip(' '))
	
	filename = filename.replace('.txt', '').strip('  ').strip(' ')

	if filename.replace('m-split_', '') in filelist and 'THE LINE' in filename or 'NEOM-NPR' in filename:
		logger.info(f"THE LINE OLD FILE USE EXIST FILE: {filename}!")

		filename = filename.replace('m-split_', '')
		# excel_path = f"./m-split/{conf['application']['name']}/m-split_{filename}.xlsx"
		excel_path = os.path.join(f"./m-split/{conf['application']['name']}", f"m-split_{filename}.xlsx")
		try: 
			wb = xlrd.open_workbook(excel_path)
			m_split_sheet = wb.sheets()[0]
			all_rows = m_split_sheet.nrows
			for i in tqdm(range(all_rows), desc=f"Use m-split"):
				if not i:
					continue
				fragments.append(m_split_sheet.cell(i, 1).value.replace('\n\n', '\n'))
		except:
			logger.info(f"M-split {filename} ERROR!")
			pass
	elif filename in filelist and conf['application']['name'] in ['the_line', 'test-aramus-qa']:
		logger.info(f"Knowledge share USE EXIST FILE: {filename}!")
		filepath = f"./m-split/the_line/new_knowledge_share/{filename}.txt"
		with open(filepath, 'r', encoding='utf-8') as file:
			new_document_content = file.read()
		if '[/SPLIT]' not in new_document_content:
			fragments.append(new_document_content)
		else:
			for j in new_document_content.split('[/SPLIT]'):
				fragments.append(j)
	elif "line wiki" in filename:
		fragments.append(document_content)
	elif "Most_used" in filename:
		excel_path = os.path.join(f"./m-split/{conf['application']['name']}", f"{filename}.xlsx")
		try: 
			wb = xlrd.open_workbook(excel_path)
			m_split_sheet = wb.sheets()[0]
			all_rows = m_split_sheet.nrows
			for i in tqdm(range(all_rows), desc=f"Use m-split"):
				if not i:
					continue
				fragments.append(m_split_sheet.cell(i, 2).value.replace('\n\n', '\n'))
		except:
			logger.info(f"M-split {filename} ERROR!")
			pass
	elif "Hajj" in filename or "Family_and" in filename:
		excel_path = os.path.join(f"./m-split/{conf['application']['name']}", f"{filename}.xlsx")
		try: 
			wb = xlrd.open_workbook(excel_path)
			m_split_sheet = wb.sheets()[0]
			all_rows = m_split_sheet.nrows
			exist_file = []
			for i in tqdm(range(all_rows), desc=f"Use m-split"):
				if not i:
					continue
				title = m_split_sheet.cell(i, 1).value.replace('servicedetails_', '').split('_')[1].strip('.txt')
				if title in exist_file:
					continue
				else:
					fragments.append(m_split_sheet.cell(i, 2).value.replace('\n\n', '\n'))
					exist_file.append(title)
		except:
			logger.info(f"M-split {filename} ERROR!")
			pass
	elif "Test" in filename and conf['application']['name'] == 'gov':
		file_path = os.path.join(f"./m-split/{conf['application']['name']}", f"{filename}.txt")
		print(f"{file_path}")
		tmp_text = ''
		try:
			with open(file_path, 'r', encoding='utf-8') as file:
				for i in file.readlines():
					tmp_text += i
			fragments.append(tmp_text)
		except:
			pass
	else:
		start_sentence_idx = 0
		print(f"Sentence len : {len(sentences)}")
		for i in tqdm(range(len(sentences)//fragment_step_size)):
		# while(start_sentence_idx+fragment_window_size <= len(sentences)):
			start_idx = i * fragment_step_size
			end_idx = i * fragment_step_size + fragment_window_size
			fragment = sentences[start_idx:end_idx]
			fragment = ' '.join(fragment)
			fragments.append(fragment)
		fragment = sentences[len(sentences)//fragment_window_size * fragment_step_size:-1]
		fragment = ' '.join(fragment)
		fragments.append(fragment)
	print(f"fragements_len: {len(fragments)}")
	'''
	fragment_window_size = 3
	fragment_step_size = 1
	start_sentence_idx = 0
	while(start_sentence_idx+fragment_window_size <= len(sentences)):
		fragment = sentences[start_sentence_idx:start_sentence_idx+fragment_window_size]
		fragment = ' '.join(fragment)
		start_sentence_idx += fragment_step_size
		fragments.append(fragment)
	'''

	# fragements to prompts
	for f in fragments:
		prompt = f"""
    <s> [INST] <<SYS>> Read the following document and generate questions from its content. The question should be self-contained and complete. The generated questions are in the format of:

    Q:
    Q:
    Q:
    ...

    Do not generate answers. Do not include the phrase "according to the document..." or "according to the author" in the question. Each question should be shorter than 50 words.

    <</SYS>>

    document: {f}
    [/INST] Sure. Here are the generated questions:
        """
		prompts.append(prompt.strip())

	print(f"Len: {len(prompts)}")
	## llama-2

	# responses = []
	# for i in tqdm(prompts, desc='1st llm'):
	if len(prompts)> 41000:
		stack_num = len(prompts) // 40009
		for i in range(stack_num):
			start = i * 40009
			end = (i + 1) * 40009 if (i+1)!=stack_num else -1
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
			for m in q['text'].split('Q:'):
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
					
		for f, q in zip(
			fragments,
			response.json()['choices']
			):
			output.append({
				'fragement':f,
				'searchable_text':f,
				'searchable_text_type': 'fragement',
			})
			for m in q['text'].split('Q:'):
				if len(m) < 3:
					continue
				output.append({
					'fragement':f,
					'searchable_text':m.replace('\n', ''),
					'searchable_text_type': 'fragment_question_by_llama3_70B',
				})
		# for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
		# 	output.append({
		# 		'fragement':f,
		# 		'searchable_text':m.group('question'),
		# 		'searchable_text_type': 'fragment_question_by_mistral_7B',
		# 	})

	##

	# responses_1 = []
	# for i in tqdm(prompts, desc='1st llm _1'):
	# response_1 = requests.post(
	# 	# 'http://192.168.0.91:3090/generate',
	# 	'http://192.168.0.205:3091/generate',
	# 	json = {
	# 	"prompt":prompts,
	# 	"stream":False,
	# 	"max_tokens":max_tokens,
	# 	"temperature": temperature
	# 	}
	# )
	# 	# responses_1.append(response_1.json()['response'][0])

	# for f, q in zip(
	# 	fragments,
	# 	response_1.json()['response']
	# 	):
	# 	output.append({
	# 		'fragement':f,
	# 		'searchable_text':f,
	# 		'searchable_text_type': 'fragement',
	# 	})
	# 	for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
	# 		output.append({
	# 			'fragement':f,
	# 			'searchable_text':m.group('question'),
	# 			'searchable_text_type': 'fragment_question_by_llama_2',
	# 		})


	## mistral
	# responses = []
	# for i in tqdm(prompts, desc='2nd llm'):
	if len(prompts)> 41000:
		stack_num = len(prompts) // 40009
		for i in range(stack_num):
			start = i * 40009
			end = (i + 1) * 40009 if (i+1)!=stack_num else -1
			response = requests.post(
			'http://192.168.0.75:3090/v1/completions',   
			json = {'prompt': prompts[start:end], 
					"model": "../../../../Mixtral-8x22B-Instruct-v0.1-AWQ", 
					"max_tokens": max_tokens, 
					"stream": False, 
					"temperature": temperature})

		for f, q in zip(
			fragments,
			response.json()['choices']
			):
			output.append({
				'fragement':f,
				'searchable_text':f,
				'searchable_text_type': 'fragement',
			})
			for m in q['text'].split('\n'):
				if len(m) < 3:
					continue
				output.append({
					'fragement':f,
					'searchable_text':m.replace('\n', ''),
					'searchable_text_type': 'fragment_question_by_mistral_8x22B',
				})
	else:
		response = requests.post(
			'http://192.168.0.75:3090/v1/completions',   
			json = {'prompt': prompts, 
					"model": "../../../../Mixtral-8x22B-Instruct-v0.1-AWQ", 
					"max_tokens": max_tokens, 
					"stream": False, 
					"temperature": temperature})

		for f, q in zip(
			fragments,
			response.json()['choices']
			):
			output.append({
				'fragement':f,
				'searchable_text':f,
				'searchable_text_type': 'fragement',
			})
			for m in q['text'].split('\n'):
				if len(m) < 3:
					continue
				output.append({
					'fragement':f,
					'searchable_text':m.replace('\n', ''),
					'searchable_text_type': 'fragment_question_by_mistral_8x22B',
				})
		# for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
		# 	output.append({
		# 		'fragement':f,
		# 		'searchable_text':m.group('question'),
		# 		'searchable_text_type': 'fragment_question_by_mistral',
		# 	})


	# responses_1 = []
	# for i in tqdm(prompts, desc='2nd llm _1'):
	# 	response_1 = requests.post(
	# 		'http://192.168.0.138:3072/generate',
	# 		json = {
	# 		"stream":False,
	# 		"prompt":i,
	# 		"max_tokens":max_tokens,
	# 		"temperature": temperature
	# 		}
	# 	)
	# 	responses_1.append(response_1.json()['response'][0])

	# for f, q in zip(
	# 	fragments,
	# 	responses_1
	# 	):
	# 	output.append({
	# 		'fragement':f,
	# 		'searchable_text':f,
	# 		'searchable_text_type': 'fragement',
	# 	})
	# 	for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
	# 		output.append({
	# 			'fragement':f,
	# 			'searchable_text':m.group('question'),
	# 			'searchable_text_type': 'fragment_question_by_mistral',
	# 		})
	print(f"fragements_len_1: {len(fragments)}")

	### sentences
	sentence_num  = len(sentences)
	filelist = ["THE LINE Adverse Weather Working Plan", "THE LINE - HSW Delivery Plan", "THE LINE - Reward and Recognition Guideline", "THE LINE - Fatigue Management Guideline",
                "THE LINE - Worker Welfare Plan", "THE LINE Adverse Weather Working Plan", "THE LINE OH&H plan draft", "NEOM-NPR-STD-001_01.00 Projects Health and Safety Assurance Standard_Jan24"]
	filename = filename.strip('.txt').strip('  ').strip(' ').replace('m-split_', '')
	
	if filename not in filelist or 'line wiki' in filename or 'Test' in filename:
		## sentence level question
		for i in range(sentence_num):
			sentence = sentences[i]
			sentence_en = re.sub(r'[^A-z]+', r'', sentence)
			if len(sentence_en) > 16:
				start_idx = np.max([0, i-sentence_left_context_size])
				end_idx = np.min([sentence_num, i+sentence_right_context_size])
				fragment = sentences[start_idx:end_idx]
				fragment = ' '.join(fragment)
				output.append({
					'fragement':fragment,
					'searchable_text':sentence,
					'searchable_text_type': 'sentence',
				})

	output = [dict(t) for t in {tuple(d.items()) for d in output}]
	if conf['application']['name'] == 'the_line':
		for i in ["What can you do?", "What's your role?", "Who are you?", "What can you do for me?", "Hi", "Hello"]:
			tmp_fragement = {'fragement':"""Welcome to THE LINE Intelligence Assistant, your trusted companion in navigating the world of construction safety! I'm here to equip you with valuable insights and information to ensure a secure work environment. From personal protective equipment to safety protocols, best practices, and identifying common hazards on construction sites, I've got you covered. 
										
										While I can offer general guidance, please note that I can't provide specific advice for individual situations. In case of a serious safety concern, it's crucial to reach out to your line manager or supervisor promptly.

										Let's work together to foster a culture of safety excellence. If you have any questions or need assistance, feel free to ask, and let's build a safer tomorrow!""",
						'searchable_text': i,
						'searchable_text_type': 'basic_qa'}
			output.append(tmp_fragement)
	else:
		for i in ["What can you do?", "What's your role?"]:
			tmp_fragement = {'fragement':"""I'am an AI assistant. I can summarize the document you selected and answer the questions you asked.""",
						'searchable_text': i,
						'searchable_text_type': 'basic_qa'}
			output.append(tmp_fragement)
	print(f"fragements_len_2: {len(fragments)}")
	
	return output

###########

def document_embedding(token_name, documents, batch_size = 100):

	fragement_num = conf.get("fragement", "fragement_num")

	batch_num = len(documents)/batch_size
	j = 0

	texts = []
	for i in tqdm(documents):
		texts.append(i['searchable_text'])
	embedding_vectors = bge_m3_embedding_function(texts).json()
	# embedding_vectors = embedding_function.encode(texts).tolist()
	for r in documents:
		r['searchable_text_embedding'] = embedding_vectors[j]
		r['id'] = token_name + "|__|" + str(hash(str([r['searchable_text'], r['fragement']])))
		j += 1

	return documents

def get_score(fragements, question):
	# print(f"Fra: {fragements}, \n Q: {question}")
	question_embedding = embedding_function.encode(question).tolist()
	similarity = 0
	for i in fragements:
		# print(f"list fragement: {i}")
		tmp = embedding_function.encode(i).tolist()
		tmp_score = np.dot(tmp,question_embedding)/(norm(tmp)*norm(question_embedding))
		# print(f"tmp_sxore: {tmp_score}")
		try:
			tmp_score = tmp_score[0]
		except:
			pass
		# print(tmp_score)
		similarity = max(tmp_score, similarity)
	return similarity

###########

# document_embedding

	# print(f"document length !!!{len(documents)}")
	# for i in tqdm(range(int(batch_num)+1), desc='embedding part'):
	# 	try:
	# 		texts = [d['searchable_text'] for d in documents[i*batch_size: (i+1)*batch_size]]

	# 		embedding_vectors = embedding_function.encode(texts).tolist()

			
	# 		for r in documents[i*batch_size: (i+1)*batch_size]:
	# 			r['searchable_text_embedding'] = embedding_vectors[j]
	# 			r['id'] = token_name + "|__|" + str(j)
	# 			j += 1
	# 	except:
	# 		pass

###########

# def document_search(query_text,document_embeddings,):

# 	query_text_embedding = embedding_function.encode([query_text])[0].tolist()
# 	document_embeddings_current = []
# 	for r in document_embeddings:
# 		try:
# 			r_current = r
# 			r_current['score'] = np.dot('
# 				np.array(query_text_embedding), 
# 				np.array(r_current['searchable_text_embedding']))
# 			document_embeddings_current.append(r_current)
# 		except:
# 			pass

# 	document_embeddings_sorted = sorted(
# 		document_embeddings_current, 
# 		key=lambda d: d['score'], 
# 		reverse=True)

# 	return document_embeddings_sorted[0]
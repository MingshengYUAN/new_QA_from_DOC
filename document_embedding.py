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

	### FOR FM
	# output.append({
	# 			'fragement':document_content,
	# 			'searchable_text':'',
	# 			'searchable_text_type': 'All_texts',
	# 		})

	# if 'Introduction' in sentences and 'Summary of Tasks' in sentences:
	# 	new_doc = nlp(document_content.split('Summary of Tasks')[0])
	# 	sentences = [s.text for s in new_doc.sentences]
	# 	summary_text = document_content.split('Summary of Tasks')[1].split('Tasks')[0]
	# 	fragments.append(f"Summary of Tasks :{summary_text}")
	# 	tasks = document_content.split('Tasks')[2].split('Legislation, Regulations, and Guidance')[0]
	# 	# for i in ['Please lists all tasks', 'list all tasks', 'show me all tasks']:
	# 	# 	output.append({
	# 	# 		'fragement':tasks,
	# 	# 		'searchable_text':i,
	# 	# 		'searchable_text_type': 'All_tasks',
	# 	# 	})
	# 	for i in tasks.split('\n\n'):
	# 		if len(i) < 3:
	# 			continue
	# 		fragments.append(i)
	# 	if 'Legislation, Regulations, and Guidance' in document_content:
	# 		last_part = document_content.split('Legislation, Regulations, and Guidance')[1]
	# 		fragments.append(last_part)


	## For the line
	filelist = ["THE LINE Adverse Weather Working Plan", "THE LINE - HSW Delivery Plan", "THE LINE - Reward and Recognition Guideline", "THE LINE - Fatigue Management Guideline",
                "THE LINE - Worker Welfare Plan", "THE LINE Adverse Weather Working Plan", "THE LINE OH&H plan draft", "NEOM-NPR-STD-001_01.00 Projects Health and Safety Assurance Standard_Jan24"]
	# filelist = []
	filelist_gov_csv = ["Business_and_Entrepreneurship", "Education_and_Training", "Family_and_Life_Events", "Hajj_and_Umrah", "Health_Services", "Islamic_Affairs", "Personal_Documents",
						"Housing_Municipal_Services_and_Utilities", "Information_Communication_and_Postal_Services", "Jobs_and_Work_Place", "Justice_and_Law", "Most_used_services",
						"Residents_and_Visitors_Affairs", "Safety_and_Environment", "Social_Protection", "Tourism_Culture_and_Entertainments", "Vehicle_and_Transportation" ,"Zakat_and_Tax_Services"]
	for i in os.listdir('./m-split/the_line/new_knowledge_share'):
		filelist.append(i.replace('.txt', '').strip(' '))
	

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
	elif conf['application']['name'] == 'gov' and filename in filelist_gov_csv:
		logger.info(f"GOV CHATY CSV FILE USE EXIST FILE: {filename}!")
		with open(f"./m-split/gov/gov_sa/{filename}.csv", 'r', encoding="utf-8") as f:
			csv_reader = csv.reader(f)
			for num, i in enumerate(csv_reader):
				if not num:
					continue
				if 's' in i[1]:
					tmp_name = i[3].replace(f"_{i[1]}", '').replace('.txt', '').replace('_', ' ')
				else:
					tmp_name = i[3].replace(f"{i[1]}_", '').replace('.txt', '').replace('_', ' ')
				fragments.append(tmp_name + '\n' + i[4])
				

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

	elif "Hajj" in filename or "Family_and" in filename or "Most_used" in filename:
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
					if '[/SPLIT]' in i:
						fragments.append(tmp_text)
						tmp_text = ''
				fragments.append(tmp_text)
		except:
			pass
	else:

		start_sentence_idx = 0
		print(f"Sentence len : {len(sentences)}")
		if '_ar' in filename or check_lang_id("\n".join(document_content.split('\n')[0:10])) == 'ar':
			fragment_window_size = 20
		for i in tqdm(range(len(sentences)//fragment_step_size), desc='sentence level add'):
		# while(start_sentence_idx+fragment_window_size <= len(sentences)):
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
	## llama-2

	# responses = []
	# for i in tqdm(prompts, desc='1st llm'):
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
		# if reponse.json()['code'] == 400 or reponse.json()['object'] == 'error':
		# 	print(response.json())
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

	# if len(prompts)> 41000:
	# 	stack_num = len(prompts) // 40009
	# 	for i in range(stack_num):
	# 		start = i * 40009
	# 		end = (i + 1) * 40009 if (i+1)!=stack_num else -1
	# 		response = requests.post(
	# 		'http://192.168.0.75:3090/v1/completions',   
	# 		json = {'prompt': prompts[start:end], 
	# 				"model": "../../../../Mixtral-8x22B-Instruct-v0.1-AWQ", 
	# 				"max_tokens": max_tokens, 
	# 				"stream": False, 
	# 				"temperature": temperature})

	# 	for f, q in zip(
	# 		fragments,
	# 		response.json()['choices']
	# 		):
	# 		output.append({
	# 			'fragement':f,
	# 			'searchable_text':f,
	# 			'searchable_text_type': 'fragement',
	# 		})
	# 		for m in q['text'].split('\n'):
	# 			if len(m) < 3:
	# 				continue
	# 			output.append({
	# 				'fragement':f,
	# 				'searchable_text':m.replace('\n', ''),
	# 				'searchable_text_type': 'fragment_question_by_mistral_8x22B',
	# 			})
	# else:
	# 	response = requests.post(
	# 		'http://192.168.0.75:3090/v1/completions',   
	# 		json = {'prompt': prompts, 
	# 				"model": "../../../../Mixtral-8x22B-Instruct-v0.1-AWQ", 
	# 				"max_tokens": max_tokens, 
	# 				"stream": False, 
	# 				"temperature": temperature})

# 	for f, q in zip(
	# 		fragments,
	# 		response.json()['choices']
	# 		):
	# 		output.append({
	# 			'fragement':f,
	# 			'searchable_text':f,
	# 			'searchable_text_type': 'fragement',
	# 		})
	# 		for m in q['text'].split('\n'):
	# 			if len(m) < 3:
	# 				continue
	# 			output.append({
	# 				'fragement':f,
	# 				'searchable_text':m.replace('\n', ''),
	# 				'searchable_text_type': 'fragment_question_by_mistral_8x22B',
	# 			})


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
                "THE LINE - Worker Welfare Plan", "THE LINE Adverse Weather Working Plan", "THE LINE OH&H plan draft", "NEOM-NPR-STD-001_01.00 Projects Health and Safety Assurance Standard_Jan24",
				"Business_and_Entrepreneurship", "Education_and_Training", "Family_and_Life_Events", "Hajj_and_Umrah", "Health_Services", "Islamic_Affairs", "Personal_Documents",
				"Housing_Municipal_Services_and_Utilities", "Information_Communication_and_Postal_Services", "Jobs_and_Work_Place", "Justice_and_Law", "Most_used_services",
				"Residents_and_Visitors_Affairs", "Safety_and_Environment", "Social_Protection", "Tourism_Culture_and_Entertainments", "Vehicle_and_Transportation" ,"Zakat_and_Tax_Services"]

	filename = filename.strip('.txt').strip('  ').strip(' ').replace('m-split_', '')

	if filename not in filelist or 'line wiki' in filename or 'Test' in filename:
		## sentence level question
		logger.info(f"sentence_num: {sentence_num}")

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
	print(f"len output: {len(output)}")
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
	for i in tqdm(documents, desc='documents embedding'):
		texts.append(i['searchable_text'])
	embedding_vectors = bge_m3_embedding_function(texts).json()
	# embedding_vectors = embedding_function.encode(texts).tolist()
	for r in tqdm(documents, desc='set id'):
		r['searchable_text_embedding'] = embedding_vectors[j]
		r['id'] = token_name + "|__|" + str(hash(str([r['searchable_text'], r['fragement']])))
		j += 1
	# print('embedding')
	return documents

def get_score(fragements, question):
	# print(f"Fra: {fragements}, \n Q: {question}")
	question_embedding = embedding_function.encode(question).tolist()
	similarity = 0
	if isinstance(fragements, str):
		fragements = [fragements]
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
from log_info import logger
import json
from utils import Prompter, embedding_function, add_qa_pairs, save_redis
from document_embedding import document_split, document_embedding, get_score
import numpy as np
import chromadb
import requests
import configparser
from share_args import ShareArgs
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')
prompter = Prompter('./prompt.json')
print(f"conf info: {conf}")
# client = chromadb.PersistentClient(path="./chromadb")
client = chromadb.PersistentClient(path=f"./chromadb/{conf['application']['name']}")

####################

def del_select_collection(token_name):
	try:
		collection = client.get_collection(token_name)
		num = collection.count()
		client.delete_collection(token_name)
		collection = client.get_collection("share")
		ids_list = [f"{token_name}|__|{i}" for i in range(num)]
		collection.delete(ids_list)
	except Exception as e:
		logger.info(f"DELETE COLLECTION ERROR: {e}")
		return "Delete ERROR!"
	return "Delete success"

####################

def empty_collection(collection_name):
	name_list = []
	if not len(collection_name):
		tmp = client.list_collections()
		for i in tmp:
			client.delete_collection(i.name)
			name_list.append(i.name)
		return name_list
	else:
		for i in collection_name:
			try:
				client.delete_collection(i)
				name_list.append(i)
			except:
				pass
		return name_list

############

def build_rag_chain_from_text(text, token_name, filename, level='None', together=0):
	
	try:
		collection = client.get_collection(token_name)
		num = collection.count()
		client.delete_collection(token_name)
		collection = client.get_collection("share")
		ids_list = [f"{token_name}|__|{i}" for i in range(num)]
		collection.delete(ids_list)
	except Exception as e:
		pass
	collection = client.create_collection(name=token_name, metadata={"hnsw:space": "cosine"})

	
	fragements = document_split(document_content=text, filename=filename)
	print(conf['application']['name'] in ['test-aramus-qa', 'the_line'])
	if conf['application']['name'] in ['the_line']:
		fragements = add_qa_pairs(fragements, filename)

	documents_vectores = document_embedding(token_name,fragements)

	# print(np.shape(documents_vectores))
	# print(documents_vectores[0].keys())
	# print(documents_vectores[0]["fragement"])
	# print(documents_vectores[0]["searchable_text"])
	# print(documents_vectores[0]["searchable_text_type"])
	# print(documents_vectores[0]["id"])

	# dict_keys(['fragement', 'searchable_text', 'searchable_text_type', 'searchable_text_embedding', 'id'])
	# I'm a developer and designer from the Netherlands. I'm a developer and designer from the Netherlands. I'm a developer and designer from the Netherlands. I'm a developer and designer from the Netherlands. I'm a developer and designer from the Netherlands.
	# How many times is it mentioned that the person is a developer and designer from the Netherlands?
	# fragment_question_by_mistral
	# id0
	# save_dict = {}
	document_list, id_list, embedding_list, metadata_list = [], [], [], []
	all_num = 0
	for i in documents_vectores:
		try:
			id_list.append(i['id'])
		except:
			print(i)
			all_num+=1
			continue
		document_list.append(i['fragement']+f"|___|{filename}")
		
		embedding_list.append(i['searchable_text_embedding'])
		if level != 'None':
			metadata_list.append({"source": i['searchable_text_type'], "searchable_text": i['searchable_text'], "filename": filename, 'level': level})
		else:
			metadata_list.append({"source": i['searchable_text_type'], "searchable_text": i['searchable_text'], "filename": filename})

	# 	if i['fragement'] in save_dict and i["searchable_text_type"] != 'sentence':
	# 		save_dict[i['fragement']].append(i['searchable_text'] + "|__|" + i["searchable_text_type"])
	# 	else:
	# 		save_dict[i['fragement']] = [i['searchable_text'] + "|__|" + i["searchable_text_type"]]
	# with open('./fragement_questions.json', 'w', encoding='utf-8') as file:
	# 	json.dump(save_dict, file, indent=4)
	if len(document_list) > 40000:
		block = len(document_list) // 40000
		for i in range(block):
			start = i * 40000
			end = (i+1) * 40000
			collection.add(documents=document_list[start:end], embeddings=embedding_list[start:end], metadatas=metadata_list[start:end], ids=id_list[start:end])
		collection.add(documents=document_list[block*40000:-1], embeddings=embedding_list[block*40000:-1], metadatas=metadata_list[block*40000:-1], ids=id_list[block*40000:-1])
	else:
		collection.add(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)

	try:
		collection = client.get_collection(name="share")
	except:
		collection = client.create_collection(name="share", metadata={"hnsw:space": "cosine"})
	# client.get_or_create_collection("share")
	collection.add(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)

	return "Success"

############

def document_search(question, token_name, fragement_num, level='None'):
	basic_qa = 0
	try:
		collection = client.get_collection(token_name)
	except Exception as e:
		logger.info(f"Load colletion ERROR: {e}")
		return "Load colletion error!"
	
	query_embedding = embedding_function.encode(question).tolist()
	searchable_text = []
	tmp_all_tasks = [[]]
	# Init return 2 fragements
	if level != 'None':
		fragement_candidates = collection.query(query_embeddings=[query_embedding], n_results=1, where={"level":level})['documents']
		tmp_searchable_text = collection.query(query_embeddings=[query_embedding], n_results=1, where={"level":level})['metadatas'][0]		
	else:
		# print(collection.query(query_embeddings=[query_embedding], n_results=1))
		# {'ids': [['ec814ed9e31f4896878e490cd9efd48b|__|31']], 'distances': [[0.4945688247680664]], 'metadatas': [[{'filename': '05+Traction Lifts.txt', 'searchable_text': ' What is the title of Task 2?', 'source': 'fragment_question_by_mixtral_8x7B'}]], 'embeddings': None, 'documents': [['Task 2 is titled Operational tasks to be carried out in addition to any maintenance or tests carried out by the maintenanceorganisation, which is categorized under  Amber  criticality group. The recommended frequency of performing this task is not Unspecified. Skillset group is  Specialist. Actions required:  A full ascent and descent to assess any changes in the quality of the ride or damage to the equipment.Typical items to be checked to ensure that they are in place, undamaged and functioning correctly are:a) landing doors and bottom door tracks;b) stopping accuracy;c) indicators that are not located in a reserved area;d) landing push controls;e) car push controls;f) door open controls;g) two-way means of communication in the car which provides permanent contact with a rescue service;h) normal car lighting;i) door reversal device;j) safety signs/pictograms. Notes: |___|05+Traction Lifts.txt']], 'uris': None, 'data': None}
		fragement_candidates = collection.query(query_embeddings=[query_embedding], n_results=1)['documents']

		# retrive directly the fragement 
		fragement_self_candidates = collection.query(query_embeddings=[query_embedding], n_results=1, where={"source": "fragement"})['documents']

		tmp_searchable_text = collection.query(query_embeddings=[query_embedding], n_results=1)['metadatas'][0]
		if conf['application']['name'] == 'the_line':
			tmp_all_tasks = collection.query(query_embeddings=[query_embedding], n_results=1, where={"source": "QA_pairs"})['documents']
		else:
			tmp_all_tasks = collection.query(query_embeddings=[query_embedding], n_results=1, where={"source": "All_texts"})['documents']


	other_candidate = ''
	if len(tmp_all_tasks[0]):
		other_candidate = tmp_all_tasks[0][0]

	for i in tmp_searchable_text:
		searchable_text.append(i['searchable_text'])
		if i['source'] == 'basic_qa':
			basic_qa = 1
		elif i['source'] == 'All_tasks' :
			basic_qa = 2
		elif i['source'] == 'QA_pairs':
			basic_qa = 3
	# print(f"Document search: {fragement_candidates}, {searchable_text}, {basic_qa}, {other_candidate}")
	return fragement_candidates, searchable_text, basic_qa, other_candidate, fragement_self_candidates

############

def answer_from_doc(token_name, question, msg_id, chat_id, condense_question, stream=False, level='None'):

	fragement_num = conf.get("fragement", "fragement_num")

	llm_dict = {}
	for i in conf['llm']:
		llm_dict[i] = conf['llm'][i]
	# llm_dict["llm"] = i

	#  Try to use condense question to find the fragment
	if level != 'None':
		fragement_candidates, searchable_text, basic_qa, other_candidate, fragement_self_candidates = document_search(condense_question, token_name, fragement_num, level)
	else:
		fragement_candidates, searchable_text, basic_qa, other_candidate, fragement_self_candidates = document_search(condense_question, token_name, fragement_num)
	logger.info(f"fragement_candidates: {fragement_candidates}")

	if len(searchable_text) == 0:
		similarity_score = 0.0
	else:
		similarity_score = get_score(searchable_text, question)

	if basic_qa == 3 :
		response = fragement_candidates[0][0].split('|___|')[1].strip('.txt')
		filename = fragement_candidates[0][0].split('|___|')[2].strip('.txt')
		fragement_candidates = fragement_candidates[0][0].split('|___|')[0].strip('.txt')
		return response, fragement_candidates, similarity_score, filename
	
	elif conf['application']['name'] == 'the_line':
		similarity_score_the_line = get_score([other_candidate], question)
		logger.info(f"the_line_sim_score: {similarity_score_the_line}")
		logger.info(f"the_line_qa: {other_candidate}")

		if similarity_score_the_line > 0.75:
			sleep(0.5)
			response = fragement_candidates[0][0].split('|___|')[1].strip('.txt')
			filename = fragement_candidates[0][0].split('|___|')[2].strip('.txt')
			fragement_candidates = fragement_candidates[0][0].split('|___|')[0].strip('.txt')
			return response, fragement_candidates, similarity_score, filename
	
	# connect all fragements
	context_fragements = ''
	for i in fragement_candidates[0]:
		context_fragements += i.split('|___|')[0]
	logger.info(f"context_fragements_len: {len(context_fragements)}")

	# use all info
	logger.info(f"Similarity_score: {similarity_score}")
	filename = ''
	try:
		filename = fragement_candidates[0][0].split('|___|')[1].strip('.txt')
	except:
		pass
	
	if similarity_score < 0.4:
		tmp_sim_score = get_score(fragement_self_candidates, question)
		if tmp_sim_score > 0.4:
			prompt = prompter.generate_prompt(question=question, context=fragement_self_candidates, prompt_serie=conf['prompt']['prompt_serie'])
			fragement_candidates = fragement_self_candidates
			basic_qa = 0
			logger.info(f"USE RETRIEVE FRAGMENT!")
		else:
			prompt = prompter.generate_prompt(question=question, context='', prompt_serie='chat')
	else:
		prompt = prompter.generate_prompt(question=condense_question, context=context_fragements, prompt_serie=conf['prompt']['prompt_serie'])


	# if similarity_score < 0.4 and other_candidate != '':
	# 	prompt = prompter.generate_prompt(question=question, context=other_candidate, prompt_serie=conf['prompt']['prompt_serie'])
	# 	fragement_candidates = other_candidate
	# 	basic_qa = 0
	# 	logger.info(f"USE ALL TEXTS!")
	# else:
	# 	prompt = prompter.generate_prompt(question=question, context=context_fragements, prompt_serie=conf['prompt']['prompt_serie'])

	if basic_qa == 1 and similarity_score > 0.8:
		if "Welcome to THE LINE Intelligence Assistant" in context_fragements:
			context_fragements = """Hi, I'm THE LINE Intelligence Assistant, your trusted companion in navigating the world of construction safety! I'm here to equip you with valuable insights and information to ensure a secure work environment. From personal protective equipment to safety protocols, best practices, and identifying common hazards on construction sites, I've got you covered.
 
While I can offer general guidance, please note that I can't provide specific advice for individual situations. In case of a serious safety concern, it's crucial to reach out to your line manager or supervisor promptly.
 
Let's work together to foster a culture of safety excellence. If you have any questions or need assistance, feel free to ask, and let's build a safer tomorrow!"""
		response = context_fragements.replace('  ', ' ')
		save_redis(chat_id, msg_id, response, 0)
		save_redis(chat_id, msg_id, '', 1)
	else:
		# print(f"--------------{prompt}\n-----------------")
		if stream:
			response = requests.post(
					# 'http://192.168.0.91:3072/generate',
					'http://192.168.0.223:3074/generate',
					json = {'prompt': prompt, 'max_tokens': 1024, 'temperature': 0.0, 'stream': stream, 'msg_id': msg_id, 'id':chat_id, 'application':conf['application']['name']}
				).status_code
			if response == 200:
				response = ''
		else:
			response = requests.post(
					# 'http://192.168.0.91:3072/generate',
					'http://192.168.0.223:3074/generate',
					json = {'prompt': prompt, 'max_tokens': 1024, 'temperature': 0.0, 'stream': stream}
				).json()['response'][0]
	# print(f"response: {response}")
	
	try:
		fragement_candidates = fragement_candidates[0][0] if type(fragement_candidates) is not str else fragement_candidates
	except:
		pass
	if basic_qa == 1:
		return response, '', None, ''
	else:
		return response, fragement_candidates, similarity_score, filename
	
############	
	


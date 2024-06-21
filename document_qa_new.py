from log_info import logger
import json
import random
from utils import Prompter, embedding_function, add_qa_pairs, save_redis, retrieve_top_fragment, bge_m3_embedding_function, check_lang_id, llm_relative_determine
from document_embedding import document_split, document_embedding, get_score
import numpy as np
import chromadb
import requests
import configparser
from tqdm import tqdm
from share_args import ShareArgs
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')
prompter = Prompter('./prompt.json')
print(f"conf info: {conf}")
client = chromadb.PersistentClient(path=f"./chromadb/{conf['application']['name']}")

####################

# delet specific file from the share collection
def del_select_collection(token_name):
	try:
		collection = client.get_collection(token_name)
		num = collection.count()
		client.delete_collection(token_name)
		collection = client.get_collection("share")
		ids_list = []
		collection.delete(ids_list, where={"token_name":token_name})
	except Exception as e:
		logger.info(f"DELETE COLLECTION ERROR: {e}")
		return "Delete success!"
	return "Delete success"

####################

# delete specific collection
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

# Main pipline of the file -> DB 
def build_rag_chain_from_text(text, token_name, filename, level='None', together=0, file_path='None', file_type='None'):
	
	try:
		collection = client.get_collection(token_name)
		num = collection.count()
		client.delete_collection(token_name)
		collection = client.get_collection("share")
		ids_list = []
		collection.delete(ids_list, where={"token_name":token_name})
		logger.info(f"Delete old colletion success!")
	except Exception as e:
		logger.info(f"Delete old colletion Fail!")
	collection = client.create_collection(name=token_name, metadata={"hnsw:space": "cosine"})

	# split document to fragement & get Questions for fragment
	fragements = document_split(document_content=text, filename=filename, file_type = file_type, file_path = file_path)

	# Do Embedding
	documents_vectores = document_embedding(token_name,fragements)

	# Reform data to metadata & document_list
	document_list, id_list, embedding_list, metadata_list = [], [], [], []
	all_num = 0
	for i in tqdm(documents_vectores, desc='process document vectors'):
		try:
			id_list.append(i['id'])
		except:
			print(i)
			all_num+=1
			continue
		document_list.append(i['fragement']+f"|___|{filename}")
		
		embedding_list.append(i['searchable_text_embedding'])
		if level != 'None':
			metadata_list.append({"source": i['searchable_text_type'], "searchable_text": i['searchable_text'], "filename": filename, "token_name": token_name ,'level': level})
		else:
			metadata_list.append({"source": i['searchable_text_type'], "searchable_text": i['searchable_text'], "filename": filename, "token_name": token_name})

	# Save to specific database
	logger.info(f"Upsert to specific colletion")
	if len(document_list) > 40000:
		block = len(document_list) // 40000
		for i in range(block):
			start = i * 40000
			end = (i+1) * 40000
			collection.upsert(documents=document_list[start:end], embeddings=embedding_list[start:end], metadatas=metadata_list[start:end], ids=id_list[start:end])
		collection.upsert(documents=document_list[block*40000:-1], embeddings=embedding_list[block*40000:-1], metadatas=metadata_list[block*40000:-1], ids=id_list[block*40000:-1])
	else:
		collection.upsert(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)

	# Save to share database
	logger.info(f"Upsert to share colletion")
	try:
		collection = client.get_collection(name="share")
	except:
		collection = client.create_collection(name="share", metadata={"hnsw:space": "cosine"})
	collection.upsert(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)

	return "Success"

############

# retrieve the fragement from the DB
def document_search(question, token_name, fragement_num, level='None', top_k=1):

	logger.info(f"Document search question: {question}")
	try:
		collection = client.get_collection(token_name)
		logger.info(f"Load colletion success")
	except Exception as e:
		logger.info(f"Load colletion ERROR: {e}")
		return "Load colletion error!"
	
	query_embedding = bge_m3_embedding_function(question).json()
	searchable_text = []
	tmp_all_texts = [[]]

	# Init return 2 fragements
	if level != 'None' and conf['application']['name'] == 'the_line':
		# retrieve candidates & metadata
		fragement_candidates = collection.query(query_embeddings=[query_embedding], n_results=3, where={"level":level})['documents']
		tmp_searchable_text = collection.query(query_embeddings=[query_embedding], n_results=3, where={"level":level})['metadatas'][0]	
		logger.info(f"level retrieve top fragment")
		
		# Rerank
		fragement_candidates, top_index, top_score = retrieve_top_fragment(fragement_candidates, question)
		tmp_searchable_text = [tmp_searchable_text[i] for i in top_index[0:top_k]]

	else:
		fragement_candidates_all = collection.query(query_embeddings=[query_embedding], n_results=3)
		logger.info(f"1st retrieve top fragment")
		fragement_candidates = fragement_candidates_all['documents'][0]
		# Agent determine if the fragment include the answer
		fragement_candidates_agent = llm_relative_determine(fragement_candidates, question)
		logger.info(f"Len: {len(fragement_candidates_agent)}")

		if len(fragement_candidates_agent) > 0:
			# Link metadata
			for i in fragement_candidates_agent:
				position = fragement_candidates.index(i)

				searchable_text.append(fragement_candidates_all['metadatas'][0][position]['searchable_text'])
			fragement_candidates = fragement_candidates_agent
			logger.info(f"Before retrieve_top_fragment: {fragement_candidates}")

			# Rerank
			fragement_candidates, _, _ = retrieve_top_fragment(fragement_candidates, question)
			logger.info(f"retrieve_top_fragment: {fragement_candidates}")

		else:
			top_index = []
			top_score = 0.0

		# retrive directly the fragement 
		fragement_self_candidates = collection.query(query_embeddings=[query_embedding], n_results=5, where={"source": "fragement"})['documents']
		logger.info(f"2nd retrieve top fragment")
		# Agent
		fragement_self_candidates = llm_relative_determine(fragement_self_candidates, question)

		#Rerank
		if len(fragement_self_candidates) > 0:
			fragement_self_candidates, _, self_top_score = retrieve_top_fragment(fragement_self_candidates, question, 2 if len(fragement_self_candidates)>=2 else 1)
		else:
			self_top_score = 0.0
		
		if len(fragement_candidates_agent) == 0:
			logger.info(f"Len 2nd: {len(fragement_self_candidates)}")

			fragement_candidates = fragement_self_candidates

		# retrieve Full text as fragment
		tmp_all_texts = collection.query(query_embeddings=[query_embedding], n_results=1, where={"source": "All_texts"})['documents']

	# rename Full text
	other_candidate = ''
	if len(tmp_all_texts[0]):
		other_candidate = tmp_all_texts[0][0]

	# print(f"Document search: {fragement_candidates}, {searchable_text}, {basic_qa}, {other_candidate}")
	return fragement_candidates, searchable_text, other_candidate, fragement_self_candidates, self_top_score

############

# Answer from doc main pipline from retrieve fragment to form a messages to the LLM
def answer_from_doc(token_name, question, msg_id, chat_id, condense_question, messages, gather_question, stream=False, level='None', use_condense=False, pre_prompt=''):

	fragement_num = conf.get("fragement", "fragement_num")

	llm_dict = {}
	for i in conf['llm']:
		llm_dict[i] = conf['llm'][i]
	
	# if not use_condense and conf['application']['name'] != 'test-aramus-qa':
	# 	condense_question = question
	
	#  Try to use condense question to find the fragment
	logger.info(f"document search start")
	
	if level != 'None' and conf['application']['name'] == 'the_line':
		# Use two way : condense question & normal question
		fragement_candidates_1, searchable_text_1, other_candidate_1, fragement_self_candidates_1, self_top_score_1 = document_search(question=question, token_name=token_name, fragement_num=fragement_num, level=level) #question, token_name, fragement_num, original_question
		fragement_candidates_2, searchable_text_2, other_candidate_2, fragement_self_candidates_2, self_top_score_2 = document_search(question=condense_question, token_name=token_name, fragement_num=fragement_num, level=level) #question, token_name, fragement_num, original_question
	else:
		fragement_candidates_1, searchable_text_1, other_candidate_1, fragement_self_candidates_1, self_top_score_1 = document_search(question=question, token_name=token_name, fragement_num=fragement_num)
		fragement_candidates_2, searchable_text_2, other_candidate_2, fragement_self_candidates_2, self_top_score_2 = document_search(question=condense_question, token_name=token_name, fragement_num=fragement_num)

	fragement_candidates = fragement_candidates_1 + fragement_candidates_2
	searchable_text = searchable_text_1 + searchable_text_2
	other_candidate = other_candidate_1 + other_candidate_2
	fragement_self_candidates = fragement_self_candidates_1 + fragement_self_candidates_2
	self_top_score = [self_top_score_1, self_top_score_2]

	if isinstance(fragement_candidates[0], list):
		fragement_candidates = fragement_candidates[0]
	logger.info(f"fragement_candidates: {fragement_candidates}")
	
	if len(searchable_text) == 0:
		similarity_score = 0.0
		similarity_score_fragment = get_score(fragement_candidates[0].split('|___|')[0].strip('.txt'), question)
	else:
		logger.info(f"searchable_text len: {len(searchable_text)}")
		# cal Q2Q score & fragment2Q score
		similarity_score = get_score(searchable_text, question)
		similarity_score_fragment = get_score(fragement_candidates[0].split('|___|')[0].strip('.txt'), question)

	
	if conf['application']['name'] == 'the_line':
		similarity_score_the_line = get_score([other_candidate], question)
		logger.info(f"the_line_sim_score: {similarity_score_the_line}")
		logger.info(f"the_line_qa: {other_candidate}")

		if similarity_score_the_line > 0.75:
			sleep(0.5)
			response = fragement_candidates[0].split('|___|')[1].strip('.txt')
			filename = fragement_candidates[0].split('|___|')[2].strip('.txt')
			fragement_candidates = fragement_candidates[0].split('|___|')[0].strip('.txt')
			return response, fragement_candidates, similarity_score, filename
	
	# connect all fragements
	context_fragements = ''
	if isinstance(fragement_candidates[0], list):
		fragement_candidates = fragement_candidates[0]
	for i in fragement_candidates:
		context_fragements += i.split('|___|')[0] + '\n'
	logger.info(f"context_fragements_len: {len(context_fragements)}")

	logger.info(f"Similarity_score: {similarity_score}")
	logger.info(f"Similarity_score_fragment: {similarity_score_fragment}")

	# Retrieve filename
	filename = ''
	try:
		if len(fragement_candidates) == 1:
			filename = fragement_candidates[0].split('|___|')[1].strip('.txt')
		else:
			filename = ''
			for i in fragement_candidates:
				filename += i.split('|___|')[1].strip('.txt') + '  &  '
	except:
		pass

	# check language to determine use which prompt (For gov)
	question_lang = check_lang_id(question)
	if conf['application']['name'] == 'gov':
		if question_lang == 'en':
			conf['prompt']['prompt_serie'] = "rag-prompt-sys-gov-en"
		else:
			conf['prompt']['prompt_serie'] = "rag-prompt-sys-gov-ar"
	# generate the final prompt for the message
	if fragement_candidates:
		final_prompt = prompter.generate_prompt_with_answer(question=condense_question, context=context_fragements, answer='', prompt_serie='chat_standard')
		
		if fragement_candidates[0] == fragement_self_candidates[0] and similarity_score < 0.4:
			similarity_score = similarity_score_fragment if similarity_score_fragment > 0.4 else random.uniform(0.4, 0.5)
	
	else:
		fragement_candidates = ''
		filename = ''
		final_prompt = prompter.generate_prompt(question=question, context='', prompt_serie='chat')

	# Generate the messages by using history QA & system prompt & final prompt
	sys_prompt = prompter.generate_prompt(question=question, context=context_fragements, prompt_serie=conf['prompt']['prompt_serie'])
	if pre_prompt != '':
		sys_prompt = pre_prompt + sys_prompt
	llm_messages = [{"role":"system", "content":sys_prompt}]
	if len(messages) == 0:
		prompt = prompter.generate_prompt(question=question, context=context_fragements, prompt_serie='rag-prompt-standard-2nd')
		llm_messages.append({"role": "user", "content":prompt})
	else:
		for num, i in enumerate(messages):
			if i['role'] == 'user':
				user_former_input = i['content']
			else:
				assistant_former_answer = i['content']
			if num % 2:
				if num == 1:
					prompt = prompter.generate_prompt(question=question, context='', prompt_serie='chat_standard')
					llm_messages.append({"role": "user", "content":prompt})
					llm_messages.append({"role": "assistant", "content":assistant_former_answer})
				elif num == len(messages) - 1 and num and num != 1:
					continue
				else:
					prompt = prompter.generate_prompt(question=question, context='', prompt_serie='rag-prompt-standard-2nd')
					llm_messages.append({"role": "user", "content":prompt})
					llm_messages.append({"role": "assistant", "content":assistant_former_answer})

		llm_messages.append({"role": "user", "content":final_prompt})

	try:
		fragement_candidates = fragement_candidates[0] if type(fragement_candidates) is not str else fragement_candidates
	except:
		pass
	if not fragement_candidates:
			fragement_candidates = ''


	return 'Workflow', fragement_candidates, similarity_score, filename, llm_messages
	
from log_info import logger
from utils import Prompter, embedding_function
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

####################3

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

def build_rag_chain_from_text(text, token_name):
	
	try:
		client.get_collection(token_name)
		client.delete_collection(token_name)
	except Exception as e:
		pass
	collection = client.create_collection(name=token_name, metadata={"hnsw:space": "cosine"})


	fragements = document_split(text)
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

	document_list, id_list, embedding_list, metadata_list = [], [], [], []
	all_num = 0
	for i in documents_vectores:
		try:
			id_list.append(i['id'])
		except:
			print(i)
			all_num+=1
			continue
		document_list.append(i['fragement'])
		
		embedding_list.append(i['searchable_text_embedding'])
		metadata_list.append({"source": i['searchable_text_type'], "searchable_text": i['searchable_text']})
		
	collection.add(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)	
	try:
		collection = client.get_collection(name="share")
	except:
		collection = client.create_collection(name="share", metadata={"hnsw:space": "cosine"})
	# client.get_or_create_collection("share")
	collection.add(documents=document_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)




	return "Success"

############

def document_search(question, token_name, fragement_num):
	try:
		collection = client.get_collection(token_name)
	except Exception as e:
		logger.info(f"Load colletion ERROR: {e}")
		return "Load colletion error!"
	
	query_embedding = embedding_function.encode(question).tolist()
	
	# Init return 2 fragements
	fragement_candidates = collection.query(query_embeddings=[query_embedding], n_results=1)['documents']

	return fragement_candidates

############

def answer_from_doc(token_name, question):

	fragement_num = conf.get("fragement", "fragement_num")

	llm_dict = {}
	for i in conf['llm']:
		llm_dict[i] = conf['llm'][i]
	# llm_dict["llm"] = i

	fragement_candidates = document_search(question, token_name, fragement_num)
	logger.info(f"fragement_candidates: {fragement_candidates}")

	if len(fragement_candidates) == 0:
		similarity_score = 0.0
	else:
		similarity_score = get_score(fragement_candidates, question)
	
	prompt = prompter.generate_prompt(question=question, context=fragement_candidates, prompt_serie=conf['prompt']['prompt_serie'])

	response = requests.post(
			'http://192.168.0.91:3090/generate',
			json = {'prompt': prompt, 'max_tokens': 512, 'temperature': 0.0, 'stream': False}
		).json()['response'][0]
	print(f"response: {response}")
	return response, fragement_candidates, similarity_score[0], ''
	
############	
	


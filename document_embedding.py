import re 
import os 
import pandas as pd
import numpy as np
import requests
import stanza
from utils import embedding_function
import torch
from tqdm import tqdm
from log_info import logger
from numpy.linalg import norm
import configparser
from share_args import ShareArgs
conf = configparser.ConfigParser()
conf.read(ShareArgs.args['config_path'], encoding='utf-8')

torch.backends.cudnn.enabled = False

stanza.download('en')
nlp = stanza.Pipeline('en',processors='tokenize',device = "cuda:0")


###########

def document_split(
	document_content,
	fragment_window_size = 5,
	fragment_step_size = 4,
	sentence_left_context_size = 2,
	sentence_right_context_size = 5,
	):
	max_tokens = int(conf.get("llm", "max_tokens"))
	temperature = float(conf.get("llm", "temperature"))

	doc = nlp(document_content)
	logger.info("Stanza part finished!")
	sentences = [s.text for s in doc.sentences]

	## prompts to get questions
	prompts = []

	# setence to fragements
	fragments = []

	start_sentence_idx = 0
	while(start_sentence_idx+fragment_window_size <= len(sentences)):
		fragment = sentences[start_sentence_idx:start_sentence_idx+fragment_window_size]
		fragment = ' '.join(fragment)
		start_sentence_idx += fragment_step_size
		fragments.append(fragment)

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

	# prompts to questions

	output = []

	## llama-2

	# responses = []
	# for i in tqdm(prompts, desc='1st llm'):
	response = requests.post(
		# 'http://192.168.0.91:3090/generate',
		'http://192.168.0.205:3091/generate',
		json = {
		"prompt":prompts,
		"stream":False,
		"max_tokens":max_tokens,
		"temperature": temperature
		}
	)
		# responses.append(response.json()['response'][0])
	for f, q in zip(
		fragments,
		response.json()['response']
		):
		output.append({
			'fragement':f,
			'searchable_text':f,
			'searchable_text_type': 'fragement',
		})
		for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
			output.append({
				'fragement':f,
				'searchable_text':m.group('question'),
				'searchable_text_type': 'fragment_question_by_llama_2',
			})

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
		
	response = requests.post(
		# 'http://192.168.0.138:3072/generate',
		'http://192.168.0.205:3092/generate',

		json = {
		"stream":False,
		"prompt":prompts,
		"max_tokens":max_tokens,
		"temperature": temperature
		}
	)
		

		# responses.append(response.json()['response'][0])

	for f, q in zip(
		fragments,
		response.json()['response']
		):
		output.append({
			'fragement':f,
			'searchable_text':f,
			'searchable_text_type': 'fragement',
		})
		for m in re.finditer(r'Q:\s*(?P<question>[^\n].*?)(\n|$)', q):
			output.append({
				'fragement':f,
				'searchable_text':m.group('question'),
				'searchable_text_type': 'fragment_question_by_mistral',
			})

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

	### sentences
	sentence_num  = len(sentences)

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


	return output

###########

def document_embedding(token_name, documents, batch_size = 100):

	fragement_num = conf.get("fragement", "fragement_num")

	batch_num = len(documents)/batch_size
	j = 0

	texts = []
	for i in tqdm(documents):
		texts.append(i['searchable_text'])
	embedding_vectors = embedding_function.encode(texts).tolist()
	for r in documents:
		r['searchable_text_embedding'] = embedding_vectors[j]
		r['id'] = token_name + "|__|" + str(j)
		j += 1

	return documents

def get_score(fragements, question):
	question_embedding = embedding_function.encode(question).tolist()
	similarity = 0
	for i in fragements:
		tmp = embedding_function.encode(i).tolist()
		similarity += np.dot(tmp,question_embedding)/(norm(tmp)*norm(question_embedding))
	return similarity/len(fragements)

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
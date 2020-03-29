#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:15:57 2020

@author: dpanugroho, 

"""

from allennlp.predictors.predictor import Predictor
import pickle
import pandas as pd
from multiprocessing.pool import ThreadPool
import torch
import itertools
import uuid

def answer_from_full_text(full_text_question_idx_pair_batch):  
    full_text_indice_to_be_processed = []
    question_to_be_processed = []
    batch_to_process = []
    doi = []
    questions_series = []
    full_text_series = []
    for batch in full_text_question_idx_pair_batch:
        batch_to_process.append({
            "question":questions[batch[1]],
            "passage":full_texts.iloc[batch[0]]['text']
            })
        doi.append(full_texts.iloc[batch[0]]['doi'])
        questions_series = questions[batch[1]]
        full_text_series = full_texts.iloc[batch[0]]['text']
        full_text_indice_to_be_processed.append(batch[0])
        question_to_be_processed.append(batch[1])
    processed_full_text_indice_string = " ".join(str(x) for x in full_text_indice_to_be_processed)

    # Reading the last state
    try:
        with open('rc.state', 'r') as state_file:
            processed_index = state_file.read()
    except:
        print("State file not found, creating a new one..\n")
        processed_index = []
    
    if (processed_full_text_indice_string in processed_index):
        return "Question already answered on this full text, skipping.."
    else:
        # Answering the question on the full text                       
        answers = pd.DataFrame(predictor.predict_batch_json(batch_to_process))
        answers = answers[['best_span','best_span_str']]
        answers['doi'] = doi
        answers['question'] = questions_series
        answers['full_text'] = full_text_series
        
        # Write the answer to a file
        answers.to_pickle(output_dir + str(uuid.uuid4()) + ".pkl")
        
        # Clear CUDA's cache
        del answers
        torch.cuda.empty_cache()
    
    
    with open('rc.state', 'a') as state_file:
        state_file.write(processed_full_text_indice_string+' ')
    return 'Finished answering on full text '+processed_full_text_indice_string+"\n"

# Credit: https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

if __name__ == "__main__":
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
    
    predictor._model = predictor._model.cuda()    
    full_texts_path = sys.argv[1]
    questions_path = sys.argv[2]
    output_dir = sys.argv[3]
    batch_size= sys.argv[4]
    
    # Read the full text and questions to determin number of iteration
    with open(full_texts_path, 'rb') as f:
        full_texts = pickle.load(f)
    with open(questions_path, 'r') as f:
        questions = f.read().splitlines() 
        
    # Sort full_text and questions to ensure consistent index
    full_texts.sort_values(by='doi', inplace=True)
    questions.sort()
    
    # Perform cross product to pair up questions and full texts
    full_text_and_question_pairs = list(itertools.product(range(len(full_texts)),
                                                          range(len(questions))))

    batches = list(chunks(full_text_and_question_pairs, batch_size))
    results = ThreadPool(1).imap_unordered(answer_from_full_text, batches)
    
    for r in results:
        print(r)
    

        

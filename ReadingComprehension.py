#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:15:57 2020

@author: dpanugroho, 

"""

from allennlp.predictors.predictor import Predictor
import sys
import pickle
import pandas as pd
from multiprocessing.pool import ThreadPool
import torch
import itertools
import uuid

def answer_from_full_text(full_text_question_idx_pair):  
    
    # Extract full_text and question
    full_text_idx, question_idx = full_text_question_idx_pair
    
    # Reading the last state
    try:
        with open('rc.state', 'r') as state_file:
            processed_index = state_file.read()
    except:
        print("State file not found, creating a new one..\n")
        processed_index = []
    
    if (str(full_text_question_idx_pair)) in processed_index:
        return "Question already answered on this full text, skipping.."
    else:
        # Answering the question on the full text
        current_answer = predictor.predict(questions[question_idx],full_texts.iloc[full_text_idx]['text'])
        
        # Save the answer as a dataframe
        answers = pd.DataFrame.from_dict([{"full_text":full_texts.iloc[full_text_idx]['text'],
                        "question":questions[question_idx],
                        "highlight":current_answer['best_span_str'],
                        "highlight_start":current_answer['best_span'][0],
                        "highlight_end":current_answer['best_span'][1],
                        "doi":full_texts.iloc[full_text_idx]['doi']}])        
        
        # Write the answer to a file
        answers.to_pickle(output_dir + str(uuid.uuid4()) + ".pkl")
        
        # Clear CUDA's cache
        del current_answer
        torch.cuda.empty_cache()
            
        with open('rc.state', 'a') as state_file:
            state_file.write(str(full_text_question_idx_pair)+' ')
        return 'Finished answering "'+questions[question_idx]+ '" on full text '+str(full_text_idx)+"\n"

if __name__ == "__main__":
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
    predictor._model = predictor._model.cuda()    
    full_texts_path = sys.argv[1]
    questions_path = sys.argv[2]
    output_dir = sys.argv[3]

    # Read the full text and questions to determin number of iteration
    with open(full_texts_path, 'rb') as f:
        full_texts = pickle.load(f)
    with open(questions_path, 'r') as f:
        questions = f.read().splitlines() 
        
    # Sort full_text and questions to ensure consistent index
    full_texts.sort_values(by='doi', inplace=True)
    questions.sort()
    
    # Perform cross produce to pair up questions and full texts
    full_text_and_question_pairs = list(itertools.product(range(len(full_texts)),
                                                          range(len(questions))))

    results = ThreadPool(3).imap_unordered(answer_from_full_text, full_text_and_question_pairs)
    
    for r in results:
        print(r)
        

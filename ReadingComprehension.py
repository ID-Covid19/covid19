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
                         
if __name__ == "__main__":
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
    predictor._model = predictor._model.cuda()
    full_texts_path = sys.argv[1]
    questions_path = sys.argv[2]
    output_path = sys.argv[3]


    with open(full_texts_path, 'rb') as f:
        full_texts = pickle.load(f)
    
    with open(questions_path, 'r') as f:
        quesitons = f.read().splitlines() 
    
    answers = []
    for question in quesitons:
        print("Question: "+question)
        for full_text in full_texts: 
            print("Processing on next full text")
            current_answer = predictor.predict(question,full_text)
            answers.append({"full_text":full_text,
                            "question":question,
                            "highlight":current_answer['best_span_str'],
                            "highlight_start":current_answer['best_span'][0],
                            "highlight_end":current_answer['best_span'][1],
                            "source":"-"})
            
    
    pd.DataFrame.from_dict(answers).to_pickle(output_path)
    

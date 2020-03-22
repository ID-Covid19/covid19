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
    
    # Reading the last state
    try:
        with open('rc.state', 'r') as state_file:
            latest_success_index = int(state_file.read())
    except:
        latest_success_index = 0
    
    # Main loop    
    for full_text in full_texts[latest_success_index:]: 
        print("Processing on next full text")
        for question in quesitons:
            print("Question: "+question)
            current_answer = predictor.predict(question,full_text)

            # Read current answer file
            try:
                answers = pd.read_pickle(output_path)
            except:
                print("File not found, creating a new one")
                answers = pd.DataFrame(columns=["full_text",
                                                "question",
                                                "highlight",
                                                "highlight_start",
                                                "highlight_end",
                                                "source"])
            # Append the new result
            print("Pre append: "+str(len(answers)))
            current_result = pd.DataFrame.from_dict([{"full_text":full_text,
                            "question":question,
                            "highlight":current_answer['best_span_str'],
                            "highlight_start":current_answer['best_span'][0],
                            "highlight_end":current_answer['best_span'][1],
                            "source":"-"}])
            answers = answers.append(current_result)
            
            print("Post append: "+str(len(answers)))
            # Write back answer file
            answers.to_pickle(output_path)
            latest_success_index+=1
            with open('rc.state', 'w') as state_file:
                state_file.write(str(latest_success_index))    

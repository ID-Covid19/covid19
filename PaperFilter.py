#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 05:54:26 2020

@author: dpanugroho

Filter paper ralted to given keyword
"""

import pandas as pd
import sys
import spacy
import dask.dataframe as dd
import glob
import json


class PaperFilter(object):
    def __init__(self, spacy_model, all_json):
        self.all_json = all_json
        self.nlp= spacy.load(spacy_model)
    
    def get_paper_path(self, sha):
        raise("Not implemented")
        return 
    
    """
    Returns text of paper (concatenation from title, abstract and full text) of a paper
    from a row in metadata
    """
    def get_text(self,row):
        try:
            path_to_full_paper = [r for r in self.all_json if row['sha'] in r][0]
            with open(path_to_full_paper) as json_data:
                full_text = [row['text'] for row in json.load(json_data)['body_text']]
                full_text = ' '.join(full_text)
                
        except:
            # If full paper not found just return empty string
            full_text = ''
        return {"text":row['title'] + '. ' + row['abstract']+'. '+full_text,
                "doi":row['doi']}
    """
    Returns similarity measure between a keyword to a row in metadata
        row: a row in metadata dataframe
        keywords: a list nlp object of a keyword
    """
    def get_document_similarity(self, row, nlp_keywords):
        # Instantiate variable to store highest similarity score
        highest_similarity = 0
        
        current_row = self.nlp((row['title'] + '. ' + row['abstract']))
        # Find maximum similarity with the queries
        for nlp_keyword in nlp_keywords:
            similarity = current_row.similarity(nlp_keyword)
            if similarity > highest_similarity:
                highest_similarity = similarity
        return highest_similarity
    
    """
    Returns dataframe containing only metadata of paper related to given keywords
    
    """
    def filter_metadata_by_keyword(self, metadata, keywords, similarity_threshold):
        # load spacy's nlp object
        
        
        nlp_keywords = []
        for keyword in keywords:
            nlp_keywords.append(self.nlp(keyword))

        dist_metadata = dd.from_pandas(metadata, npartitions=12)
        metadata['similarity_score'] = dist_metadata.apply(self.get_document_similarity, 
                                                           nlp_keywords=nlp_keywords, 
                                                           axis=1,
                                                           meta=('float64')).compute(scheduler='threads')  
        
        return metadata[metadata['similarity_score']>similarity_threshold]        

if __name__ == "__main__":
    spacy.prefer_gpu()
    dataset_dir = sys.argv[1]
    keyword_path = sys.argv[2]
    output_filename = sys.argv[3] 
    similarity_threshold = float(sys.argv[4]) 

    # Instantiate PaperFilter object
    all_json = glob.glob(f'{dataset_dir}/**/*.json', recursive=True)
    pp = PaperFilter('en_core_sci_md',all_json)
    
    # Read keyword input from a text file
    path_to_keyword_file = keyword_path
    with open(path_to_keyword_file, 'r') as f:
        keywords = f.readlines()
    
    # Read metadata input
    metadata = pd.read_csv(dataset_dir+'/metadata.csv')[:5]
    
    # Drop entries without title, abstract, or doi
    metadata.dropna(subset=['abstract','title', 'doi'], inplace=True)
    
    # Filter to only paper related to the keywords
    filtered_metadata = pp.filter_metadata_by_keyword(metadata, keywords, similarity_threshold)
    
    # Get text of each paper
    paper_texts = []
    for index, row in filtered_metadata.iterrows():
        paper_texts.append(pp.get_text(row))
        
    # Save list of text to a pickle file
    pd.DataFrame(paper_texts).to_pickle(output_filename)




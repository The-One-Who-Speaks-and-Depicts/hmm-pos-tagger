#!/usr/bin/python
# -*- coding: utf-8 -*

from random import shuffle
import itertools
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import numpy as np

def test_split(word, pos, join, grams, double):
    n_grams = []
    if ((grams == 0) or (len(word) <= grams)):
        if (join == 1):
            n_grams.append([word + pos])
        else:
            n_grams.append([word])
    else:
        counter = 0
        while ((len(word) - counter) >= grams):
            resulting_word = ''
            repetitions = 0
            for i in range (counter, counter + grams):
                if (double == 1):
                    if ((counter > 0) and (i == counter) and ((word[i] == 'о') and (word[i + 1] == 'у'))):
                        resulting_word += word[i - 1]
                        resulting_word += word[i]
                    elif ((counter + i < len(word)) and (i == counter + grams - 1) and ((word[i] == 'о') and (word[i + 1] == 'у'))):
                        resulting_word += word[i]
                        resulting_word += word[i + 1]
                    elif ((counter + i < len(word)) and ((i > counter) and (i < counter + grams - 1)) and ((word[i] == 'о') and (word[i + 1] == 'у')) and (counter + grams + repetitions < len(word))):
                        resulting_word += word[i]                        
                        repetitions = repetitions + 1
                    elif ((counter > 0) and (i == counter) and (word[i] == word [i - 1])):
                        resulting_word += word[i - 1]
                        resulting_word += word[i]
                    elif ((counter + i < len(word)) and (i == counter + grams - 1) and (word[i] == word[i + 1])):
                        resulting_word += word[i]
                        resulting_word += word[i + 1]
                    elif ((counter + i < len(word)) and ((i > counter) and (i < counter + grams - 1)) and (word[i] == word[i + 1]) and (counter + grams + repetitions < len(word))):
                        resulting_word += word[i]                        
                        repetitions = repetitions + 1
                    else:
                        resulting_word += word[i]
                else:
                    resulting_word += word[i]
            if (repetitions > 0):
                for i in range(counter + grams, counter + grams + repetitions):
                    resulting_word += word[i]
            if (join == 0):
                n_grams.append([resulting_word])
            else:
                n_grams.append([resulting_word + pos])
            counter = counter + 1
    return n_grams

def n_gram_vectorizer(data, grams, register_change):
    total_n_grams = []
    for index, sequence in enumerate(data):
        if (register_change == 1):
            analyzed_token = sequence[0][0].lower()
        else:
            analyzed_token = sequence[0][0]
        for gram_complex in test_split(analyzed_token, "", 0, grams, 0):
            total_n_grams.append(' '.join(gram for gram in gram_complex))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(total_n_grams)
    return X

def tf_idf_n_gram(k, ngram_count, max_ngram_count, corpus_length, words_with_ngram_count):
    return (k + (1 - k) * (ngram_count/max_ngram_count)) * (math.log(corpus_length/(1 + words_with_ngram_count)))

def get_data(filepath):
    raw_data = open(filepath, encoding='utf8').readlines()
    all_sequences = []    
    for instance in raw_data:
        current_sequence = []
        if (instance[0] != "#" and instance.strip()):
            cols = instance.split('\t')
            current_sequence.append((cols[1], cols[3]))
            all_sequences.append(current_sequence)
    return all_sequences

def get_test_data(filepath):
    raw_data = open(filepath, encoding='utf8').readlines()
    all_sequences = []    
    for instance in raw_data:
        current_sequence = []
        if (instance[0] != "#" and instance.strip()):
            cols = instance.split()
            current_sequence.append((cols[1], cols[2]))
            all_sequences.append(current_sequence)
    return all_sequences

def get_data_for_prediction(filepath):
    all_sequences = [] 
    with open(filepath, encoding='utf8') as f:
        d = json.load(f)    
    for t in d["texts"]:
        for c in t["clauses"]:
            for r in c["realizations"]:
                current_sequence = []
                if r["lexemeTwo"].strip():
                    current_sequence.append((r["lexemeTwo"], "", r["textID"] + "_" + r["clauseID"] + "_" + r["realizationID"]))
                    all_sequences.append(current_sequence)                
    return all_sequences

def split_data(data, percent):
    shuffle(data)
    train_size = int(len(data) * int(percent) / 100)
    return data[:train_size], data[train_size:]

def normalize(matrix):
    row_sums = matrix.sum(axis=1)
    np.seterr(divide='ignore', invalid='ignore')
    return matrix / row_sums[:, np.newaxis]            

def enumerate_list(data):
    return {instance: index for index, instance in enumerate(data)}
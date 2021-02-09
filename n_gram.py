#!/usr/bin/python
# -*- coding: utf-8 -*

import re
import pandas as pd
import os
from data_preparation import *
from collections import Counter
import numpy as np
import pickle
from metrics import accuracy, build_confusion_matrices

def n_gram_train(filepath, grammage, folder, register_change, start_end_symbols, weighed, tf_idf_coefficient, length, double):
    tf_idf_coefficient = float(tf_idf_coefficient)
    dataset = pd.DataFrame(columns=['WORD', 'TAG'])
    raw_data = open(filepath, encoding='utf8').readlines()
    counter = 0   
    for instance in raw_data:
      if (instance[0] != "#" and instance.strip()):
        cols = instance.split('\t')
        if (int(register_change) == 0):
            dataset.loc[counter] = [cols[1], cols[3]]
        else:
            dataset.loc[counter] = [cols[1].lower(), cols[3]]
        counter = counter + 1
    names = dataset['TAG'].unique().tolist()
    final_dictionary = {}
    start_end_symbols = int(start_end_symbols)
    if (length == 1):
        by_length_dictionary = {}
    if (weighed == 1):
        corpus_length = dataset.shape[0]
        ngram_in_word_dictionary = {}
        words_with_ngram_dictionary = {}
    for name in names:
        clone = dataset[dataset['TAG'] == name]
        n_grams = []
        if (length == 1):
            words_lengths = []
        for word in clone['WORD']:          
          if (start_end_symbols == 1): 
            word = "#" + word + "#"
          if (length == 1):
            words_lengths.append(len(word))
          if (weighed == 1):
            ngrams_of_word_dictionary = {}
          for gram in test_split(word, "", 0, int(grammage), double):
            n_grams.extend(gram)            
            if (weighed == 1):
                if gram[0] in words_with_ngram_dictionary.keys():
                    words_with_ngram_dictionary[gram[0]].append(word)
                else:
                    words_with_ngram_dictionary[gram[0]] = []
                    words_with_ngram_dictionary[gram[0]].append(word)
                if gram[0] in ngrams_of_word_dictionary.keys():
                    ngrams_of_word_dictionary[gram[0]] += 1
                else:
                    ngrams_of_word_dictionary[gram[0]] = 1
          if (weighed == 1):
              for gram in ngrams_of_word_dictionary.keys():
                if gram in ngram_in_word_dictionary.keys():
                    if (ngrams_of_word_dictionary[gram] > ngram_in_word_dictionary[gram]):
                        ngram_in_word_dictionary[gram] = ngrams_of_word_dictionary[gram]
                else:
                    ngram_in_word_dictionary[gram] = ngrams_of_word_dictionary[gram]
          if (length == 1):
            by_length_dictionary[name] = round(np.mean(words_lengths))
        cnt = Counter(n_grams)
        grams = []
        if (weighed == 0):
            for gram in cnt.most_common(2):
              grams.append(gram[0])            
        else:
            weighed_grams = {}
            for gram in cnt.most_common():
                weighed_grams[gram[0]] = tf_idf_n_gram(tf_idf_coefficient, gram[1], ngram_in_word_dictionary[gram[0]], corpus_length, len(words_with_ngram_dictionary[gram[0]]))
            weighed_grams = dict(reversed(sorted(weighed_grams.items(), key=lambda item: item[1])))
            for key in list(weighed_grams.keys())[0:2]:
                grams.append(key)
        final_dictionary[name] = grams        
    with open(folder + "\\" + grammage + 'grams.pkl', 'wb+') as f:
        pickle.dump(final_dictionary, f, pickle.HIGHEST_PROTOCOL)
    if (length == 1):
        with open(folder + "\\length_" + grammage + 'grams.pkl', 'wb+') as f:
            pickle.dump(by_length_dictionary, f, pickle.HIGHEST_PROTOCOL)
        
        
def n_gram_test(data, folder, grammage, register_change, start_end_symbols, length):
    test_dataset = pd.DataFrame(columns=['WORD', 'TAG'])
    raw_data = open(data, encoding='utf8').readlines()
    counter = 0   
    start_end_symbols = int(start_end_symbols)
    for instance in raw_data:
      if (instance[0] != "#" and instance.strip()):
        cols = instance.split('\t')
        if (int(register_change) == 0):
            test_dataset.loc[counter] = [cols[1], cols[3]]
        else:
            if (start_end_symbols == 0):
                test_dataset.loc[counter] = [cols[1].lower(), cols[3]]
            else:
                test_dataset.loc[counter] = ["#" + cols[1].lower() + "#", cols[3]]
        counter = counter + 1
    with open(folder + "\\" + grammage + "grams.pkl", 'rb') as f:
        final_dictionary = pickle.load(f)
    if (length == 1):
        with open(folder + "\\length_" + grammage + 'grams.pkl', 'rb') as f:
            by_length_dictionary = pickle.load(f)
    correct = 0
    total = 0
    correct_by_part = []
    total_by_part = []
    true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
    for index, row in test_dataset.iterrows():
      key_found = False
      for key in final_dictionary.keys():
        if re.search(final_dictionary[key][0], row['WORD']):
          if key == row['TAG']:
            correct = correct + 1
            correct_by_part.append(key)
          key_found = True
          true_pred_dataset.loc[index] = [row['TAG'], key]
          break
        elif re.search(final_dictionary[key][1], row['WORD']):
          if key == row['TAG']:
            correct = correct + 1
            correct_by_part.append(key)
          key_found = True
          true_pred_dataset.loc[index] = [row['TAG'], key]
          break
        elif length == 1:
            if len(row['WORD']) == by_length_dictionary['CCONJ']:
                if row['TAG'] == 'CCONJ':
                    correct = correct + 1
                    correct_by_part.append(key)
                key_found = True
                true_pred_dataset.loc[index] = [row['TAG'], key]
                break
            elif len(row['WORD']) == by_length_dictionary['ADP']:
                if row['TAG'] == 'ADP':
                    correct = correct + 1
                    correct_by_part.append(key)
                key_found = True
                true_pred_dataset.loc[index] = [row['TAG'], key]
                break
            elif len(row['WORD']) == by_length_dictionary['VERB']:
                if row['TAG'] == 'VERB':
                    correct = correct + 1
                    correct_by_part.append(key)
                key_found = True
                true_pred_dataset.loc[index] = [row['TAG'], key]
                break
      if not key_found:
        if row['TAG'] == 'VERB':
           correct = correct + 1
           correct_by_part.append(key)
        true_pred_dataset.loc[index] = [row['TAG'], 'VERB']
      total = total + 1
      total_by_part.append(key)
    accuracy(correct_by_part, total_by_part, correct, total)        
    build_confusion_matrices(true_pred_dataset)
 
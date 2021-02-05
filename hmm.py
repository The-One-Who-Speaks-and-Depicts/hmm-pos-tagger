#!/usr/bin/python
# -*- coding: utf-8 -*
import os
from random import shuffle
from random import seed
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import itertools
from subcategorization import is_punct, is_frag, is_digit
import argparse
from collections import Counter
import pickle
import json
import treetaggerwrapper
from sklearn.metrics import confusion_matrix
import pandas as pd
import re
from sklearn.ensemble import ExtraTreesRegressor
import math

def tf_idf_n_gram(k, ngram_count, max_ngram_count, corpus_length, words_with_ngram_count):
    return (k + (1 - k) * (ngram_count/max_ngram_count)) * (math.log(corpus_length/(1 + words_with_ngram_count)))


def tree_tag(data):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='bg')
    correct = 0
    total = 0
    correct_by_part = []
    total_by_part = []
    true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
    for index, sequence in enumerate(data):  
        tags = tagger.tag_text(sequence[0][0])
        tags2 = treetaggerwrapper.make_tags(tags)
        tag_acquired = tags2[0].pos
        if (tag_acquired.startswith('A') or tag_acquired.startswith('Mo') or tag_acquired.startswith('Md') or tag_acquired.startswith('My') or tag_acquired.startswith('H')):
            tag_acquired = 'ADJ'
        elif (tag_acquired.startswith('D')):
            tag_acquired = 'ADV'
        elif(tag_acquired.startswith('I')):
            tag_acquired = 'INTJ'
        elif (tag_acquired.startswith('Nc')):
            tag_acquired = 'NOUN'
        elif (tag_acquired.startswith('Np')):
            tag_acquired = 'PROPN'
        elif (tag_acquired.startswith('Vn') or tag_acquired.startswith('Vp')):
            tag_acquired = 'VERB'
        elif (tag_acquired.startswith('R')):
            tag_acquired = 'ADP'
        elif (tag_acquired.startswith('Vx') or tag_acquired.startswith('Vy') or tag_acquired.startswith('Vi')):
            tag_acquired = 'AUX'
        elif (tag_acquired.startswith('Cc') or tag_acquired.startswith('Cr') or tag_acquired.startswith('Cp')):
            tag_acquired = 'CCONJ'
        elif (tag_acquired.startswith('Ps')):
            tag_acquired = 'DET'
        elif (tag_acquired.startswith('Mc')):
            tag_acquired = 'NUM'
        elif (tag_acquired.startswith('T')):
            tag_acquired = 'PART'
        elif (tag_acquired.startswith('Pp') or tag_acquired.startswith('Pd') or tag_acquired.startswith('Pr') or tag_acquired.startswith('Pc') or tag_acquired.startswith('Pi') or tag_acquired.startswith('Pf') or tag_acquired.startswith('Pn')):
            tag_acquired = 'PRON'
        elif (tag_acquired.startswith('Cs')):
            tag_acquired = 'SCONJ'
        else:
            tag_acquired = 'X'
        if (tag_acquired == data[index][0][1]):
            correct = correct + 1
            correct_by_part.append(data[index][0][1])
        total = total + 1
        total_by_part.append(data[index][0][1])
        true_pred_dataset.loc[index] = [data[index][0][1], tag_acquired]
    correct_by_part_fin = Counter(correct_by_part)
    total_by_part_fin = Counter(total_by_part)
    for correct_part in correct_by_part_fin.keys():
        for total_part in total_by_part_fin.keys():
            if correct_part == total_part:
                print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')
    print('Total accuracy score: ' + str(correct/total*100) + '%')
    for pos in true_pred_dataset['pred'].unique().tolist():
        this_pos_split = true_pred_dataset[true_pred_dataset['pred'] == pos]
        cm = confusion_matrix(this_pos_split['true'], this_pos_split['pred'])
        print(f'Raw confusion matrix for {pos}.\n{cm}')
        for index, row in this_pos_split.iterrows():
            if (row['pred'] == row['true']):
                row['pred'] = 1
            else:
                row['pred'] = 0
            row['true'] = 1            
        try:
            tn, fp, fn, tp = confusion_matrix(this_pos_split['true'].astype('int'), this_pos_split['pred'].astype('int')).ravel()
            print(f'Binarized confusion matrix for {pos}. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
        except:
            print(f'Unable to binarize confusion matrix for {pos}')
    cm = confusion_matrix(true_pred_dataset['true'], true_pred_dataset['pred'])
    print(f'Raw total confusion matrix.\n{cm}')        
    for index, row in true_pred_dataset.iterrows():
        if (row['pred'] == row['true']):
            row['pred'] = 1
        else:
            row['pred'] = 0
        row['true'] = 1
    tn, fp, fn, tp = confusion_matrix(true_pred_dataset['true'].astype('int'), true_pred_dataset['pred'].astype('int')).ravel()
    print(f'Binarized total confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
    

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
    correct_by_part_fin = Counter(correct_by_part)
    total_by_part_fin = Counter(total_by_part)
    for correct_part in correct_by_part_fin.keys():
        for total_part in total_by_part_fin.keys():
            if correct_part == total_part:
                print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')  
    print(f'Accuracy score: {correct/total*100}%')
    for pos in true_pred_dataset['pred'].unique().tolist():
        this_pos_split = true_pred_dataset[true_pred_dataset['pred'] == pos]
        cm = confusion_matrix(this_pos_split['true'], this_pos_split['pred'])
        print(f'Raw confusion matrix for {pos}.\n{cm}')
        for index, row in this_pos_split.iterrows():
            if (row['pred'] == row['true']):
                row['pred'] = 1
            else:
                row['pred'] = 0
            row['true'] = 1            
        try:
            tn, fp, fn, tp = confusion_matrix(this_pos_split['true'].astype('int'), this_pos_split['pred'].astype('int')).ravel()
            print(f'Binarized confusion matrix for {pos}. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
        except:
            print(f'Unable to binarize confusion matrix for {pos}')
    cm = confusion_matrix(true_pred_dataset['true'], true_pred_dataset['pred'])
    print(f'Raw total confusion matrix.\n{cm}')        
    for index, row in true_pred_dataset.iterrows():
        if (row['pred'] == row['true']):
            row['pred'] = 1
        else:
            row['pred'] = 0
        row['true'] = 1
    tn, fp, fn, tp = confusion_matrix(true_pred_dataset['true'].astype('int'), true_pred_dataset['pred'].astype('int')).ravel()
    print(f'Binarized total confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
    

class HMM:

    def __init__(self, train_data, test_data,unknown_to_singleton,printSequences):
        self.train_data = train_data
        self.test_data = test_data
        self.tags = []
        self.words = []
        self.tag_dict = {}
        self.word_dict = {}
        self.num_of_tags = 0
        self.unknown_tags = []
        self.transition_probs = None
        self.emission_probs = None
        self.initial_probs = None
        self.unknown_to_singleton = unknown_to_singleton
        self.printSequences = printSequences

    def train(self):
        self.tags = sorted(list(set([tag for sequence in self.train_data for word, tag in sequence])))
        self.tag_dict = enumerate_list(self.tags)
        tokens = list([word.lower() for sequence in self.train_data for word, tag in sequence])  
        self.words = list(set(tokens))
        self.word_dict = enumerate_list(self.words)
        self.num_of_tags = len(self.tags)
        transition_probs= np.zeros((self.num_of_tags, self.num_of_tags)) 
        emission_probs = np.zeros((self.num_of_tags, len(self.words)))
        initial_probs = np.zeros(self.num_of_tags)
        for sequence in self.train_data:
            prev_tag = 'None'
            for word, tag in sequence:
                word_lower = word.lower()
                tag_id = self.tag_dict[tag]
                word_id = self.word_dict[word_lower]
                if prev_tag != 'None':
                    transition_probs[self.tag_dict[prev_tag], tag_id] += 1  
                    prev_tag = tag
                else:
                    prev_tag = tag
                    initial_probs[tag_id] += 1    
                if word_id not in emission_probs[tag_id]:
                    emission_probs[tag_id, word_id] = 1
                else:
                    emission_probs[tag_id, word_id] += 1
        if self.unknown_to_singleton==1:
            tokencounts = Counter(tokens)
            singleton_word_indices =  list(map(lambda a: tokencounts[a]==1,self.words))
            for tag in emission_probs:
                self.unknown_tags.append(np.dot(tag,singleton_word_indices))
            singletonCount = sum(singleton_word_indices)
            self.unknown_tags = [i/singletonCount for i in self.unknown_tags ]
        self.transition_probs = normalize(transition_probs)
        self.emission_probs = normalize(emission_probs)
        self.initial_probs = initial_probs/initial_probs.sum()           

    def test(self):
        word_truth = [0, 0]
        for index, sequence in enumerate(self.test_data):
            actual_tags = list(map(lambda x: self.tag_dict[x[1]], sequence))
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))            
            if self.printSequences==1:
                print(' '.join(map(lambda x: x[0], sequence)))
                print(str(index) + ' '.join([word for word, tag in sequence]))
                print(' '.join(map(lambda x: x[1], sequence)))
                print(' '.join([str(self.tags[tag]) for tag in predicted_tags]))
                print(' '.join(map(lambda x: str(self.word_dict.get(x[0].lower(),-1)), sequence)))            
            for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
                if actual_tag == predicted_tag:
                    word_truth[0] += 1
                else: 
                    word_truth[1] += 1
        print("word truth : "+str(word_truth[0]/(word_truth[0]+word_truth[1]))+"%")

    def viterbi(self, sequence):
        seq_len = len(sequence)
        V = np.zeros((self.num_of_tags, seq_len))
        B = np.zeros((self.num_of_tags, seq_len))    
        V[:, 0] = self.initial_probs * self.get_emission_prob(sequence[0], -1)
        B[:, 0] = np.argmax(self.initial_probs * self.get_emission_prob(sequence[0], -1))
        for t in range(1, seq_len):
            for s in range(self.num_of_tags):
                e_prob = self.get_emission_prob(sequence[t], s)
                result = V[:, t-1] * self.transition_probs[:, s] * e_prob
                V[s, t] = max(result)
                B[s, t] = np.argmax(result)
        x = np.empty(seq_len, 'B')
        x[-1] = np.argmax(V[:, seq_len - 1])
        for i in reversed(range(1, seq_len)):
            x[i - 1] = B[x[i], i]
        return x.tolist()

    def get_emission_prob(self, word, state=-1):
        index = self.word_dict.get(word.lower(), -1)   
        if index != -1:
            if state == -1:
                return self.emission_probs[:, index]
            else:
                prob = self.emission_probs[state, index] 
                return self.emission_probs[state, index]

        tag_likelihoods = {'PUNCT': False, 'FRAG': False}
        probable_tags = [k for k, v in tag_likelihoods.items() if v == True]
        if len(probable_tags) == 0:
            if self.unknown_to_singleton == 1:
                return self.unknown_tags
            else:
               probable_tags.append('NOUN')
        if state == -1:
            all_emissions = np.zeros(len(self.tags))
            for tag in probable_tags:
                all_emissions[self.tag_dict[tag]] = self.emission_probs[self.tag_dict[tag]].mean()
            return all_emissions
        else:     
            return np.matrix([self.emission_probs[self.tag_dict[tag]] for tag in probable_tags]).mean()
    
    def predict(self, data, folder, grammage):
        with open(folder + "\\" + grammage + "grams.pkl", 'rb') as f:
            final_dictionary = pickle.load(f)
        predicted = []
        for index, sequence in enumerate(data):
            if is_frag(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'FRAG'
                predicted.append(sequence[0][2] + '\t' + tag_acquired)
            elif is_punct(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'PUNCT'
                predicted.append(sequence[0][2] + '\t' + tag_acquired)
            elif is_digit(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'DIGIT'
                predicted.append(sequence[0][2] + '\t' + tag_acquired)
            else:
                predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
                if (re.search(final_dictionary['ADJ'][0], sequence[0][0]) or re.search(final_dictionary['ADJ'][1], sequence[0][0])):
                    tag_acquired = 'ADJ'
                if (re.search(final_dictionary['VERB'][0], sequence[0][0]) or re.search(final_dictionary['VERB'][1], sequence[0][0])):
                    tag_acquired = 'VERB'            
                if (re.search(final_dictionary['X'][0], sequence[0][0]) or re.search(final_dictionary['X'][1], sequence[0][0])):
                    tag_acquired = 'X'
                predicted.append(sequence[0][2] + '\t' + tag_acquired)
        return predicted
    
    def competitive_predict(self, data, folder, grammage, register_change):
        with open(folder + "\\" + grammage + "grams.pkl", 'rb') as f:
            final_dictionary = pickle.load(f)
        comparison_dataset = pd.DataFrame(columns=['TOKEN', 'HMM', 'ENHANCED'])
        for index, sequence in enumerate(data):
            if is_frag(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'FRAG'
                comparison_dataset.loc[index] = [sequence[0][0], tag_acquired, tag_acquired]
            elif is_punct(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'PUNCT'
                comparison_dataset.loc[index] = [sequence[0][0], tag_acquired, tag_acquired]
            elif is_digit(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'DIGIT'
                comparison_dataset.loc[index] = [sequence[0][0], tag_acquired, tag_acquired]
            else:
                predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_hmm = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
                if (register_change == 1):
                    analyzed_token = sequence[0][0].lower()
                else:
                    analyzed_token = sequence[0][0]
                tag_acquired = tag_hmm
                if (re.search(final_dictionary['ADJ'][0], analyzed_token) or re.search(final_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                if (re.search(final_dictionary['VERB'][0], analyzed_token) or re.search(final_dictionary['VERB'][1], analyzed_token)):
                    tag_acquired = 'VERB'
                if (re.search(final_dictionary['X'][0], analyzed_token) or re.search(final_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
                comparison_dataset.loc[index] = [sequence[0][0], tag_hmm, tag_acquired]
        return comparison_dataset
        
    def accuracy_score(self, data):
        correct = 0
        total = 0
        correct_by_part = []
        total_by_part = []
        true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
        for index, sequence in enumerate(data):  
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            tag_acquired = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            if (tag_acquired == data[index][0][1]):
                correct = correct + 1
                correct_by_part.append(data[index][0][1])
            total = total + 1
            total_by_part.append(data[index][0][1])
            true_pred_dataset.loc[index] = [data[index][0][1], tag_acquired]
        correct_by_part_fin = Counter(correct_by_part)
        total_by_part_fin = Counter(total_by_part)
        for correct_part in correct_by_part_fin.keys():
            for total_part in total_by_part_fin.keys():
                if correct_part == total_part:
                    print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')
        print('Total accuracy score: ' + str(correct/total*100) + '%')
        for pos in true_pred_dataset['pred'].unique().tolist():
            this_pos_split = true_pred_dataset[true_pred_dataset['pred'] == pos]
            cm = confusion_matrix(this_pos_split['true'], this_pos_split['pred'])
            print(f'Raw confusion matrix for {pos}.\n{cm}')
            for index, row in this_pos_split.iterrows():
                if (row['pred'] == row['true']):
                    row['pred'] = 1
                else:
                    row['pred'] = 0
                row['true'] = 1            
            try:
                tn, fp, fn, tp = confusion_matrix(this_pos_split['true'].astype('int'), this_pos_split['pred'].astype('int')).ravel()
                print(f'Binarized confusion matrix for {pos}. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
            except:
                print(f'Unable to binarize confusion matrix for {pos}')
        cm = confusion_matrix(true_pred_dataset['true'], true_pred_dataset['pred'])
        print(f'Raw total confusion matrix.\n{cm}')        
        for index, row in true_pred_dataset.iterrows():
            if (row['pred'] == row['true']):
                    row['pred'] = 1
            else:
                row['pred'] = 0
            row['true'] = 1
        tn, fp, fn, tp = confusion_matrix(true_pred_dataset['true'].astype('int'), true_pred_dataset['pred'].astype('int')).ravel()
        print(f'Binarized total confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
    
    def hybrid_accuracy_score(self, data, folder, grammage, register_change, start_end_symbols):
        register_change = int(register_change)
        start_end_symbols = int(start_end_symbols)
        if grammage == 'double_3_and_4':
            with open(folder + "\\3grams.pkl", 'rb') as f:
                three_gram_dictionary = pickle.load(f)
            with open(folder + "\\4grams.pkl", 'rb') as f:
                four_gram_dictionary = pickle.load(f)
        else:
            with open(folder + "\\" + grammage + "grams.pkl", 'rb') as f:
                final_dictionary = pickle.load(f)
        correct = 0
        total = 0
        correct_by_part = []
        total_by_part = []
        true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
        overall_positions = []
        verb_positions = []
        adj_positions = []
        x_positions = []
        for index, sequence in enumerate(data):  
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            tag_acquired = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            found_with_ngram = False
            if (register_change == 0):
                analyzed_token = sequence[0][0]
            else:
                if (start_end_symbols == 0):
                    analyzed_token = sequence[0][0].lower()
                else:
                    analyzed_token = "#" + sequence[0][0].lower() + "#"
            if ((grammage == '3') and (register_change == 1) and (start_end_symbols == 1)) :
                if (re.search(final_dictionary['AUX'][0], analyzed_token) or re.search(final_dictionary['AUX'][1], analyzed_token)):
                    tag_acquired = 'AUX'
                if (re.search(final_dictionary['X'][0], analyzed_token) or re.search(final_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
                if (re.search(final_dictionary['SCONJ'][0], analyzed_token) or re.search(final_dictionary['SCONJ'][1], analyzed_token)):
                    tag_acquired = 'SCONJ'
                if (re.search(final_dictionary['PROPN'][0], analyzed_token) or re.search(final_dictionary['PROPN'][1], analyzed_token)):
                    tag_acquired = 'PROPN'
                if (re.search(final_dictionary['ADJ'][0], analyzed_token) or re.search(final_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                #if (re.search(final_dictionary['ADV'][0], analyzed_token) or re.search(final_dictionary['ADV'][1], analyzed_token)):
                    #tag_acquired = 'ADV' 
                if (re.search(final_dictionary['PRON'][0], analyzed_token) or re.search(final_dictionary['PRON'][1], analyzed_token)):
                    tag_acquired = 'PRON'
            elif ((grammage == '3') and ((register_change == 1) and (start_end_symbols == 0))):
                if (re.search(final_dictionary['VERB'][0], analyzed_token) or re.search(final_dictionary['VERB'][1], analyzed_token)):
                    tag_acquired = 'VERB'
                    found_with_ngram = True
                if (re.search(final_dictionary['ADJ'][0], analyzed_token) or re.search(final_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                    found_with_ngram = True
                #if (re.search(final_dictionary['ADV'][0], analyzed_token) or re.search(final_dictionary['ADV'][1], analyzed_token)):
                    #tag_acquired = 'ADV' 
                if (re.search(final_dictionary['X'][0], analyzed_token) or re.search(final_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
                    found_with_ngram = True
            elif ((grammage == '3') and ((register_change == 0) and (start_end_symbols == 0))):
                if (re.search(final_dictionary['VERB'][0], analyzed_token) or re.search(final_dictionary['VERB'][1], analyzed_token)):
                    tag_acquired = 'VERB'
                if (re.search(final_dictionary['ADJ'][0], analyzed_token) or re.search(final_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                #if (re.search(final_dictionary['ADV'][0], analyzed_token) or re.search(final_dictionary['ADV'][1], analyzed_token)):
                    #tag_acquired = 'ADV' 
                if (re.search(final_dictionary['X'][0], analyzed_token) or re.search(final_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
            elif grammage == '4':
                if (re.search(final_dictionary['VERB'][0], analyzed_token) or re.search(final_dictionary['VERB'][1], analyzed_token)):
                    tag_acquired = 'VERB'
                if (re.search(final_dictionary['ADJ'][0], analyzed_token) or re.search(final_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                #if (re.search(final_dictionary['ADV'][0], analyzed_token) or re.search(final_dictionary['ADV'][1], analyzed_token)):
                    #tag_acquired = 'ADV'
                #if (re.search(final_dictionary['PRON'][0], analyzed_token) or re.search(final_dictionary['PRON'][1], analyzed_token)):
                    #tag_acquired = 'PRON'
                if (re.search(final_dictionary['X'][0], analyzed_token) or re.search(final_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
            elif grammage == 'double_3_and_4':
                if (re.search(four_gram_dictionary['VERB'][0], analyzed_token) or re.search(four_gram_dictionary['VERB'][1], analyzed_token)):
                    tag_acquired = 'VERB'
                if (re.search(three_gram_dictionary['ADJ'][0], analyzed_token) or re.search(three_gram_dictionary['ADJ'][1], analyzed_token)):
                    tag_acquired = 'ADJ'
                #if (re.search(three_gram_dictionary['ADV'][0], analyzed_token) or re.search(three_gram_dictionary['ADV'][1], analyzed_token)):
                    #tag_acquired = 'ADV'
                #if (re.search(four_gram_dictionary['PRON'][0], analyzed_token) or re.search(four_gram_dictionary['PRON'][1], analyzed_token)):
                    #tag_acquired = 'PRON'                    
                if (re.search(three_gram_dictionary['X'][0], analyzed_token) or re.search(three_gram_dictionary['X'][1], analyzed_token)):
                    tag_acquired = 'X'
            if (tag_acquired == data[index][0][1]):
                correct = correct + 1
                correct_by_part.append(data[index][0][1])
                if found_with_ngram:
                    if tag_acquired == 'VERB':
                        if (analyzed_token.find(final_dictionary['VERB'][0]) != -1):
                            verb_positions.append(analyzed_token.find(final_dictionary['VERB'][0]))
                            overall_positions.append(analyzed_token.find(final_dictionary['VERB'][0]))
                        else:
                            verb_positions.append(analyzed_token.find(final_dictionary['VERB'][1]))
                            overall_positions.append(analyzed_token.find(final_dictionary['VERB'][1]))
                    elif tag_acquired == 'ADJ':
                        if (analyzed_token.find(final_dictionary['ADJ'][0]) != -1):
                            adj_positions.append(analyzed_token.find(final_dictionary['ADJ'][0]))
                            overall_positions.append(analyzed_token.find(final_dictionary['ADJ'][0]))
                        else:
                            adj_positions.append(analyzed_token.find(final_dictionary['ADJ'][1]))
                            overall_positions.append(analyzed_token.find(final_dictionary['ADJ'][1]))
                    elif tag_acquired == 'X':
                        if (analyzed_token.find(final_dictionary['X'][0]) != -1):
                            x_positions.append(analyzed_token.find(final_dictionary['X'][0]))
                            overall_positions.append(analyzed_token.find(final_dictionary['X'][0]))
                        else:
                            x_positions.append(analyzed_token.find(final_dictionary['X'][1]))
                            overall_positions.append(analyzed_token.find(final_dictionary['X'][1]))
            true_pred_dataset.loc[index] = [data[index][0][1], tag_acquired]
            total = total + 1
            total_by_part.append(data[index][0][1])
        correct_by_part_fin = Counter(correct_by_part)
        total_by_part_fin = Counter(total_by_part)
        for correct_part in correct_by_part_fin.keys():
            for total_part in total_by_part_fin.keys():
                if correct_part == total_part:
                    print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')
        print('Total accuracy score: ' + str(correct/total*100) + '%')
        for pos in true_pred_dataset['pred'].unique().tolist():
            this_pos_split = true_pred_dataset[true_pred_dataset['pred'] == pos]
            cm = confusion_matrix(this_pos_split['true'], this_pos_split['pred'])
            print(f'Raw confusion matrix for {pos}.\n{cm}')
            for index, row in this_pos_split.iterrows():
                if (row['pred'] == row['true']):
                    row['pred'] = 1
                else:
                    row['pred'] = 0
                row['true'] = 1            
            try:
                tn, fp, fn, tp = confusion_matrix(this_pos_split['true'].astype('int'), this_pos_split['pred'].astype('int')).ravel()
                print(f'Binarized confusion matrix for {pos}. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
            except:
                print(f'Unable to binarize confusion matrix for {pos}')
        cm = confusion_matrix(true_pred_dataset['true'], true_pred_dataset['pred'])
        print(f'Raw total confusion matrix.\n{cm}')        
        for index, row in true_pred_dataset.iterrows():
            if (row['pred'] == row['true']):
                    row['pred'] = 1
            else:
                row['pred'] = 0
            row['true'] = 1
        tn, fp, fn, tp = confusion_matrix(true_pred_dataset['true'].astype('int'), true_pred_dataset['pred'].astype('int')).ravel()
        print(f'Binarized total confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
        verb_positions_mean = np.mean(verb_positions)
        adj_positions_mean = np.mean(adj_positions)
        x_positions_mean = np.mean(x_positions)
        overall_positions_mean = np.mean(overall_positions)
        print(f'Adjective definitive ngram mean position: {adj_positions_mean}\nVerb definitive ngram mean position: {verb_positions_mean}\nX definitive ngram mean position: {x_positions_mean}\nDefinitive ngram mean position:{overall_positions_mean}')
    
    def hybrid_accuracy_score_with_classification(self, data_test, data_train, folder, grammage, register_change):
        with open(folder + "\\" + grammage + "grams.pkl", 'rb') as f:
            final_dictionary = pickle.load(f)
        tags_gram = []
        tags_hmm = []
        tags_golden = []
        for index, sequence in enumerate(data_train):  
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            tag_hmm = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            tag_changed = False
            if (int(register_change) == 0):
                analyzed_token = sequence[0][0]
            else:
                analyzed_token = sequence[0][0].lower()
            for name in final_dictionary.keys():
                if (re.search(final_dictionary[name][0], analyzed_token) or re.search(final_dictionary[name][1], analyzed_token)):
                    tag_gram = name
                    tag_changed = True
            if not tag_changed:
                tag_gram = tag_hmm
            tags_gram.append(tag_gram)
            tags_hmm.append(tag_hmm)
            tags_golden.append(data_train[index][0][1])
        counter = 0
        quantified_tags = {}
        for tag in list(set(tags_golden)):
            quantified_tags[tag] = counter
            counter = counter + 1
        inverted_tags = {v: k for k, v in quantified_tags.items()}
        dataset = pd.DataFrame(columns=['HMM', 'GRAM', 'RES'])
        for i in range(len(tags_golden)):
            dataset.loc[i] = [quantified_tags[tags_hmm[i]], quantified_tags[tags_gram[i]], quantified_tags[tags_golden[i]]]
        y = dataset.iloc[:,2]
        y = y.astype('int')
        X = dataset.iloc[:,:2]
        X = np.array(X.values.tolist())        
        ETR = ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=0)
        ETR.fit(X, y)
        correct = 0
        total = 0
        correct_by_part = []
        total_by_part = []        
        true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
        for index, sequence in enumerate(data_test):  
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            tag_hmm = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            tag_changed = False
            if (int(register_change) == 0):
                analyzed_token = sequence[0][0]
            else:
                analyzed_token = sequence[0][0].lower()
            for name in final_dictionary.keys():
                if (re.search(final_dictionary[name][0], analyzed_token) or re.search(final_dictionary[name][1], analyzed_token)):
                    tag_gram = name
                    tag_changed = True            
            if tag_changed:
                tag_final = inverted_tags[round(ETR.predict(np.array([[quantified_tags[tag_hmm], quantified_tags[tag_gram]]]))[0])]
                if (tag_final == data_test[index][0][1]):
                    correct = correct + 1
                    correct_by_part.append(data_test[index][0][1])
                true_pred_dataset.loc[index] = [data_test[index][0][1], tag_final]
            else:
                if (tag_hmm == data_test[index][0][1]):
                    correct = correct + 1
                    correct_by_part.append(data_test[index][0][1])
                true_pred_dataset.loc[index] = [data_test[index][0][1], tag_hmm]
            total = total + 1
            total_by_part.append(data_test[index][0][1])
        correct_by_part_fin = Counter(correct_by_part)
        total_by_part_fin = Counter(total_by_part)
        for correct_part in correct_by_part_fin.keys():
            for total_part in total_by_part_fin.keys():
                if correct_part == total_part:
                    print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')
        print('Total accuracy score: ' + str(correct/total*100) + '%')
        for pos in true_pred_dataset['pred'].unique().tolist():
            this_pos_split = true_pred_dataset[true_pred_dataset['pred'] == pos]
            cm = confusion_matrix(this_pos_split['true'], this_pos_split['pred'])
            print(f'Raw confusion matrix for {pos}.\n{cm}')
            for index, row in this_pos_split.iterrows():
                if (row['pred'] == row['true']):
                    row['pred'] = 1
                else:
                    row['pred'] = 0
                row['true'] = 1            
            try:
                tn, fp, fn, tp = confusion_matrix(this_pos_split['true'].astype('int'), this_pos_split['pred'].astype('int')).ravel()
                print(f'Binarized confusion matrix for {pos}. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')
            except:
                print(f'Unable to binarize confusion matrix for {pos}')
        cm = confusion_matrix(true_pred_dataset['true'], true_pred_dataset['pred'])
        print(f'Raw total confusion matrix.\n{cm}')        
        for index, row in true_pred_dataset.iterrows():
            if (row['pred'] == row['true']):
                    row['pred'] = 1
            else:
                row['pred'] = 0
            row['true'] = 1
        tn, fp, fn, tp = confusion_matrix(true_pred_dataset['true'].astype('int'), true_pred_dataset['pred'].astype('int')).ravel()
        print(f'Binarized total confusion matrix. True negatives: {tn}, false positives: {fp}, false negatives: {fn}, true positives: {tp}')

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
            current_sequence.append((cols[1], cols[3]))
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


def main(args):
    if (args.modus == 'training'):
        if (args.method == 'hmm'):
            seed(5)
            all_sequences = get_data(args.data)
            train_data, test_data = split_data(all_sequences, args.split)
            hmm = HMM(train_data, test_data, int(args.unknown_to_singleton),int(args.printSequences))
            hmm.train()
            hmm.test()
            with open(args.folder + '\\hmm.pkl', 'wb') as output:
                pickle.dump(hmm, output, pickle.HIGHEST_PROTOCOL)
                print("Way to file: " + args.folder + "\\hmm.pkl")
        elif (args.method == 'grams'):
            n_gram_train(args.data, args.grammage, args.folder, args.register_change, args.start_end_symbols, int(args.weighed), args.tf_idf_coefficient, int(args.length), int(args.double))
            print("Way to file: " + args.folder + "\\" + args.grammage + "grams.pkl")
        else:
            print('Wrong method!')
    elif (args.modus == 'accuracy'):
        if (args.method == 'hmm'):
            with open(args.folder + '\\hmm.pkl', 'rb') as inp:
                predictor = pickle.load(inp)
                predictor.accuracy_score(get_test_data(args.data))
        elif (args.method == 'grams'):
            n_gram_test(args.data, args.folder, args.grammage, args.register_change, args.start_end_symbols, int(args.length))
        elif (args.method == 'hmmg'):
            with open(args.folder + '\\hmm.pkl', 'rb') as inp:
                predictor = pickle.load(inp)
                predictor.hybrid_accuracy_score(get_test_data(args.data), args.folder, args.grammage, args.register_change, args.start_end_symbols)
        elif (args.method == 'hmmc'):
            with open(args.folder + '\\hmm.pkl', 'rb') as inp:
                predictor = pickle.load(inp)
                predictor.hybrid_accuracy_score_with_classification(get_test_data(args.data), get_test_data(args.train_data), args.folder, args.grammage, args.register_change)
        elif (args.method == 'tt'):
            tree_tag(get_test_data(args.data))
        else:
            print('Wrong method!')
    elif (args.modus == 'prediction'):
        with open(args.folder + '\\hmm.pkl', 'rb') as inp:
            predictor = pickle.load(inp)
            predictions = predictor.predict(get_data_for_prediction(args.data), args.folder, args.grammage)
            with open(args.data, encoding='utf8') as f:
                d = json.load(f)    
            for t in d["texts"]:
                for c in t["clauses"]:
                    for r in c["realizations"]:
                        for p in predictions:
                            id, pos = p.split('\t')
                            textID, clauseID, realizationID = id.split('_')
                            if ((r["textID"] == textID) and (r["clauseID"] == clauseID) and (r["realizationID"] == realizationID)):                                
                                r["realizationFields"].append({"PoS":[pos]})        
            with open(args.data, 'w', encoding='utf8') as f:
                json.dump(d, f, ensure_ascii=False)                    
    elif (args.modus == 'competitive_prediction'):
        with open(args.folder + '\\hmm.pkl', 'rb') as inp:
            predictor = pickle.load(inp)
            predictions = predictor.competitive_predict(get_data_for_prediction(args.data), args.folder, args.grammage, args.register_change)
            predictions.to_csv(args.folder + "\\predictions.csv", index=False)
    else:
        print('Incorrect modus!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--train_data', default='')
    parser.add_argument('--split', default='90')
    parser.add_argument('--unknown_to_singleton', default='0')
    parser.add_argument('--printSequences',default='0')
    parser.add_argument('--folder', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--modus', default='training')
    parser.add_argument('--method', default='hmm')
    parser.add_argument('--grammage', default='3')
    parser.add_argument('--register_change', default='1')
    parser.add_argument('--start_end_symbols', default='0')
    parser.add_argument('--tf_idf_coefficient', '-k', default='0.5')
    parser.add_argument('--weighed', '-w', default='0')
    parser.add_argument('--length', '-l', default='0')
    parser.add_argument('--double', '-d', default='0')

    args = parser.parse_args()
    main(args)
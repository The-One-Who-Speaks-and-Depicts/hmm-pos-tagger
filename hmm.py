#!/usr/bin/python
# -*- coding: utf-8 -*

import os
import numpy as np
import itertools
from subcategorization import is_punct, is_frag, is_digit
from collections import Counter
import pickle
import json
import pandas as pd
import re
from sklearn.ensemble import ExtraTreesRegressor
from data_preparation import *

from metrics import accuracy, build_confusion_matrices

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
    
    def predict(self, data, folder, grammage, register_change):
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
                if (register_change == 1):
                    analyzed_token = sequence[0][0].lower()
                else:
                    analyzed_token = sequence[0][0]
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
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            word_tagged = ' '.join(map(lambda x: x[0], sequence))
            tag_hmm = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            if (register_change == 1):
                analyzed_token = sequence[0][0].lower()
            else:
                analyzed_token = sequence[0][0]
            tag_acquired = tag_hmm
            if is_frag(analyzed_token):
                tag_acquired = 'FRAG'
            if is_punct(analyzed_token):
                tag_acquired = 'PUNCT'
            if is_digit(analyzed_token):
                tag_acquired = 'DIGIT'
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
        accuracy(correct_by_part, total_by_part, correct, total)        
        build_confusion_matrices(true_pred_dataset)

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
        accuracy(correct_by_part, total_by_part, correct, total)        
        build_confusion_matrices(true_pred_dataset)
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
        accuracy(correct_by_part, total_by_part, correct, total)        
        build_confusion_matrices(true_pred_dataset)
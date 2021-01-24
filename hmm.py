#!/usr/bin/python
# -*- coding: utf-8 -*
import os
from random import shuffle
from random import seed
import numpy as np
#import seaborn as sns; sns.set()
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import itertools
from subcategorization import is_punct, is_frag, is_digit
import argparse
from collections import Counter
import pickle
import json

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
        sentence_truth = [0, 0]
        word_truth = [0, 0]
        conf_mat = np.zeros((self.num_of_tags, self.num_of_tags))
        for index, sequence in enumerate(self.test_data):
            actual_tags = list(map(lambda x: self.tag_dict[x[1]], sequence))
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))            
            if self.printSequences==1:
                print(' '.join(map(lambda x: x[0], sequence)))
                print(str(index) + ' '.join([word for word, tag in sequence]))
                print(' '.join(map(lambda x: x[1], sequence)))
                print(' '.join([str(self.tags[tag]) for tag in predicted_tags]))
                print(' '.join(map(lambda x: str(self.word_dict.get(x[0].lower(),-1)), sequence)))
            if actual_tags == predicted_tags:
                sentence_truth[0] += 1
            else: 
                sentence_truth[1] += 1
            for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
                if actual_tag == predicted_tag:
                    word_truth[0] += 1
                else: 
                    word_truth[1] += 1
                conf_mat[actual_tag, predicted_tag] += 1
        print("sentence truth : "+str(sentence_truth[0]/(sentence_truth[0]+sentence_truth[1]))+"%")
        print("word truth : "+str(word_truth[0]/(word_truth[0]+word_truth[1]))+"%")
        return word_truth, sentence_truth, conf_mat

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
    
    def predict(self, data):
        predicted = []
        for index, sequence in enumerate(data):
            if is_frag(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'FRAG'
                predicted.append(word_tagged + ' ' + tag_acquired)
            elif is_punct(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'PUNCT'
                predicted.append(word_tagged + ' ' + tag_acquired)
            elif is_digit(sequence[0][0]):
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = 'DIGIT'
                predicted.append(word_tagged + ' ' + tag_acquired)
            else:
                predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
                word_tagged = ' '.join(map(lambda x: x[0], sequence))
                tag_acquired = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
                predicted.append(word_tagged + ' ' + tag_acquired)
        return predicted
    
    def accuracy_score(self, data):
        correct = 0
        total = 0
        for index, sequence in enumerate(data):  
            predicted_tags = self.viterbi(list(map(lambda x: x[0], sequence)))
            tag_acquired = ' '.join([str(self.tags[tag]) for tag in predicted_tags])
            if (tag_acquired == data[index][0][1]):
                correct = correct + 1
            total = total + 1
        print('Total accuracy score: ' + str(correct/total*100) + '%')


def get_data(filepath):
    raw_data = open(filepath, encoding='utf-8').readlines()
    all_sequences = []    
    for instance in raw_data:
        current_sequence = []
        if (instance[0] != "#" and instance.strip()):
            cols = instance.split('\t')
            current_sequence.append((cols[1], cols[3]))
            all_sequences.append(current_sequence)
    return all_sequences

def get_test_data(filepath):
    raw_data = open(filepath, encoding='utf-8').readlines()
    all_sequences = []    
    for instance in raw_data:
        current_sequence = []
        if (instance[0] != "#" and instance.strip()):
            cols = instance.split()
            current_sequence.append((cols[1], cols[3]))
            all_sequences.append(current_sequence)
    return all_sequences

def get_data_for_prediction(filepath):
    import json
    all_sequences = [] 
    with open(filepath, encoding='utf-8') as f:
        d = json.load(f)    
    for t in d["texts"]:
        for c in t["clauses"]:
            for r in c["realizations"]:
                current_sequence = []
                if r["lexemeTwo"].strip():
                    current_sequence.append((r["lexemeTwo"], ""))
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
        seed(5)
        all_sequences = get_data(args.data)
        train_data, test_data = split_data(all_sequences, args.split)
        hmm = HMM(train_data, test_data, int(args.unknown_to_singleton),int(args.printSequences))
        hmm.train()
        hmm.test()
        with open(args.folder + '\\hmm.pkl', 'wb') as output:
            pickle.dump(hmm, output, pickle.HIGHEST_PROTOCOL)
            print("Way to file: " + args.folder + "\\hmm.pkl")
    elif (args.modus == 'accuracy'):
        with open(args.folder + '\\hmm.pkl', 'rb') as inp:
            predictor = pickle.load(inp)
            predictor.accuracy_score(get_test_data(args.data))
    elif (args.modus == 'prediction'):
        with open(args.folder + '\\hmm.pkl', 'rb') as inp:
            predictor = pickle.load(inp)
            predictions = predictor.predict(get_data_for_prediction(args.data))
            with open(args.folder + '\\result.txt', 'w', encoding='utf-8') as out:
                for p in predictions:
                    out.write(p + '\n')
    else:
        print('Incorrect modus!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--split', default='90')
    parser.add_argument('--unknown_to_singleton', default='0')
    parser.add_argument('--printSequences',default='0')
    parser.add_argument('--folder', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--modus', default='training')

    args = parser.parse_args()
    main(args)
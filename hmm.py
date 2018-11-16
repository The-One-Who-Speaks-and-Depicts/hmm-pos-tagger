#!/usr/bin/python
from random import shuffle
from random import seed
import numpy as np
import seaborn as sns; sns.set()
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import itertools
class HMM:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tags = []
        self.words = []
        self.tag_dict = {}
        self.word_dict = {}
        self.num_of_tags = 0
        self.transition_probs = None
        self.emission_probs = None
        self.initial_probs = None

    def train(self):
        self.tags = list(set([tag for sequence in train_data for word, tag in sequence]))
        self.tag_dict = enumerate_list(self.tags)    
        self.words = list(set([word.lower() for sequence in train_data for word, tag in sequence]))
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
        self.transition_probs = normalize(transition_probs)
        self.emission_probs = normalize(emission_probs)
        self.initial_probs = initial_probs/initial_probs.sum()           

    def test(self):
        sentence_truth = [0, 0]
        word_truth = [0, 0]
        conf_mat = np.zeros((self.num_of_tags, self.num_of_tags))
        for sequence in self.test_data:
            actual_tags = list(map(lambda x: self.tag_dict[x[1]], sequence))
            predicted_tags = self.viterbi(list(map(lambda x: self.word_dict.get(x[0].lower(), -1), sequence)))
            print(' '.join(map(lambda x: x[0], sequence)))
            print(' '.join([str(tag) for tag in predicted_tags]))
            #sentence_truth[actual_tags == predicted_tags] += 1
            if actual_tags == map(lambda x: self.tags[x], predicted_tags):
                sentence_truth[0] += 1
            else: 
                sentence_truth[1] += 1
            for actual_tag, predicted_tag in zip(actual_tags, predicted_tags):
                #word_truth[actual_tag == predicted_tag] += 1
                if actual_tag == predicted_tag:
                    print('correct')
                    word_truth[0] += 1
                else: 
                    print('wrong')
                    word_truth[1] += 1
                conf_mat[actual_tag, predicted_tag] += 1  
        print(self.tag_dict)
        print(sentence_truth)
        print(word_truth)
        plot_confusion_matrix(conf_mat, normalize=False)
        #ax = sns.heatmap(conf_mat, cmap='viridis')
        #ax.set_title('Confusion Matrix')
        #ax.figure.savefig("output.png")
        return word_truth, sentence_truth, conf_mat

    def viterbi(self, sequence):
        seq_len = len(sequence)
        V = np.zeros((self.num_of_tags, seq_len))
        B = np.zeros((self.num_of_tags, seq_len))    
        V[:, 0] = self.initial_probs * self.emission_probs[:, sequence[0]]
        for t in range(1, seq_len):
            for s in range(self.num_of_tags):
                result = V[:, t-1] * self.transition_probs[:, s] * self.emission_probs[s, sequence[t]]
                V[s, t] = max(result)
                B[s, t] = np.argmax(result)
       
        V[:, seq_len-1] = max(V[:, seq_len-1])  
        B[:, seq_len-1] = np.argmax(V[:, seq_len-1])    

        x = np.empty(seq_len, 'B')
        x[-1] = np.argmax(V[:, seq_len - 1])
        for i in reversed(range(1, seq_len)):
            x[i - 1] = B[x[i], i]
        return x


def get_data(filepath):
    raw_data = open(filepath, encoding='utf8').readlines()
    all_sequences = []
    current_sequence = []
    for instance in raw_data:
        if instance != '\n':
            cols = instance.split()
            if cols[1] != '_':
                if cols[3] == 'satın':
                    current_sequence.append((cols[1], 'Noun'))
                else:    
                    current_sequence.append((cols[1], cols[3]))
        else:
            all_sequences.append(current_sequence)
            current_sequence = []   
    return all_sequences

def split_data(data, percent):
    shuffle(data)
    train_size = int(len(data) * percent / 100)
    return data[:train_size], data[train_size:]

def normalize(matrix):
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]            

def enumerate_list(data):
    return {instance: index for index, instance in enumerate(data)}

def plot_confusion_matrix(cm, cmap=None, normalize = True, target_names = None, title = "Confusion Matrix"):
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(24, 18))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('output.png')

seed(5)
all_sequences = get_data('Project (Application 1) (MetuSabanci Treebank).conll')
train_data, test_data = split_data(all_sequences, 90)
hmm = HMM(train_data, test_data)
hmm.train()
hmm.test()
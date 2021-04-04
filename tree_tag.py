#!/usr/bin/python
# -*- coding: utf-8 -*
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import treetaggerwrapper
import pandas as pd
import itertools
from metrics import accuracy, build_confusion_matrices

def tree_tag(data, folder, lang):
    try:
        tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)
        correct = 0
        total = 0
        correct_by_part = []
        total_by_part = []
        true_pred_dataset = pd.DataFrame(columns=['true', 'pred'])
        for index, sequence in enumerate(data):  
            tags = tagger.tag_text(sequence[0][0])
            tags2 = treetaggerwrapper.make_tags(tags)
            tag_acquired = tags2[0].pos
            if (lang == 'bg'):
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
            elif (lang == 'ru'):
                if (tag_acquired.startswith('A') or (tag_acquired.startswith('Mc'))):
                    tag_acquired = 'ADJ'
                elif (tag_acquired == 'C'):
                    tag_acquired = 'CCONJ'
                elif (tag_acquired == 'I'):
                    tag_acquired = 'INTJ'
                elif (tag_acquired.startswith('Mo') or tag_acquired.startswith('Mp')):
                    tag_acquired = 'NUM'
                elif (tag_acquired.startswith('Nc')):
                    tag_acquired = 'NOUN'
                elif (tag_acquired.startswith('Np')):
                    tag_acquired = 'PROPN'
                elif (tag_acquired.startswith('P')):
                    tag_acquired = 'PRON'
                elif (tag_acquired == 'Q'):
                    tag_acquired = 'PART'
                elif (tag_acquired.startswith('R')):
                    tag_acquired = 'ADV'
                elif (tag_acquired.startswith('S')):
                    tag_acquired = 'ADP'
                elif (tag_acquired.startswith('V')):
                    tag_acquired = 'VERB'
                else:
                    tag_acquired = 'X'
            elif (lang == 'sk'):
                if (tag_acquired.startswith('S')):
                    tag_acquired = 'NOUN'
                elif (tag_acquired.startswith('A')):
                    tag_acquired = 'ADJ'
                elif (tag_acquired.startswith('P') or tag_acquired.startswith('R')):
                    tag_acquired = 'PRON'
                elif (tag_acquired.startswith('N')):
                    tag_acquired = 'NUM'
                elif (tag_acquired.startswith('V')):
                    if (tag_acquired.startswith('VB')):
                        tag_acquired = 'AUX'
                    else:
                        tag_acquired = 'VERB'
                elif (tag_acquired.startswith('G') or tag_acquired.startswith('Y')):
                    tag_acquired = 'VERB'
                elif (tag_acquired.startswith('D')):
                    tag_acquired = 'ADV'
                elif (tag_acquired.startswith('E')):
                    tag_acquired = 'ADP'
                elif (tag_acquired.startswith('O')):
                    tag_acquired = 'CCONJ'
                elif (tag_acquired.startswith('T')):
                    tag_acquired = 'PART'
                elif (tag_acquired.startswith('J')):
                    tag_acquired = 'INTJ'
                elif (tag_acquired.startswith('W') or tag_acquired.startswith(':r')):
                    tag_acquired = 'PROPN'
                elif (tag_acquired.startswith('Z') or tag_acquired.startswith('Q') or tag_acquired.startswith('%') or tag_acquired.startswith('0') or tag_acquired.startswith(':q') or tag_acquired.startswith('#') or tag_acquired.startswith('?')):
                    tag_acquired = 'X'
                else:
                    tag_acquired = 'X'            
            if (tag_acquired == data[index][0][1]):
                correct = correct + 1
                correct_by_part.append(data[index][0][1])
            total = total + 1
            total_by_part.append(data[index][0][1])
            true_pred_dataset.loc[index] = [data[index][0][1], tag_acquired]
        final_dataset = pd.DataFrame(columns=['tok', 'true', 'pred'])
        for index, sequence in enumerate(data):
            final_dataset[index] = [data[index][0][0], data[index][0][1], true_pred_dataset.loc[index]['pred']]
        final_dataset.to_csv(folder + "\\" + 'res.csv', index=False)
        accuracy(correct_by_part, total_by_part, correct, total)        
        build_confusion_matrices(true_pred_dataset)
    except treetaggerwrapper.TreeTaggerError as e:
        print(e)
    
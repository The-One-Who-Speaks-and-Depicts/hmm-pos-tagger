from collections import Counter
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

def accuracy(correct_by_part, total_by_part, correct, total):
    correct_by_part_fin = Counter(correct_by_part)
    total_by_part_fin = Counter(total_by_part)
    for correct_part in correct_by_part_fin.keys():
        for total_part in total_by_part_fin.keys():
            if correct_part == total_part:
                print(f'Accuracy for {correct_part}: {correct_by_part_fin[correct_part]/total_by_part_fin[total_part]*100}%')
    print('Total accuracy score: ' + str(correct/total*100) + '%')

def build_confusion_matrices(true_pred_dataset):
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

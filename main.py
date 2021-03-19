#!/usr/bin/python
# -*- coding: utf-8 -*
import os
from random import seed
import argparse
import pickle
import json
from hmm import HMM
from data_preparation import *
from n_gram import n_gram_train, n_gram_test
from tree_tag import tree_tag

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
                predictor.accuracy_score(args.folder, get_test_data(args.data))
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
            tree_tag(get_test_data(args.data), args.folder, args.lang)
        else:
            print('Wrong method!')
    elif (args.modus == 'prediction'):
        with open(args.folder + '\\hmm.pkl', 'rb') as inp:
            predictor = pickle.load(inp)
            predictions = predictor.predict(get_data_for_prediction(args.data), args.folder, args.grammage, int(args.register_change))
            with open(args.data, encoding='utf8') as f:
                d = json.load(f)    
            for t in d["texts"]:
                for c in t["clauses"]:
                    for r in c["realizations"]:
                        field_exists = False
                        for f in r["realizationFields"]:
                            if "PoS" in f.keys():
                                field_exists = True
                                break
                        if not field_exists:
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
    elif (args.modus == 'vectorization'):
        vectors = n_gram_vectorizer(get_test_data(args.data), int(args.grammage), int(args.register_change))
        with open(args.folder + "\\vectors_" + args.grammage + "grams.pkl", "wb") as out:
            pickle.dump(vectors, out, pickle.HIGHEST_PROTOCOL)
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
    parser.add_argument('--lang', default='bg')

    args = parser.parse_args()
    main(args)
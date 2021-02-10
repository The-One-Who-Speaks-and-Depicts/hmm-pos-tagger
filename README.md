# hmm-pos-tagger

A tool for Old Church Slavonic PoS tagging by different methods. The baseline is Python TreeTagger implementation that uses the Bulgarian language parameters. The basal methods are HMM (enhanced with Viterbi algorithm), and n-gram analysis, that makes PoS prediction on the basis of occurence of the most frequent n-grams of the particular PoS in a token. The hybridisation techniques include using rules (for types of tokens that are not present in the training dataset, yet existing in the language itself, such as fragmentary tokens, punctuation marks, and digits) and n-gram analysis to correct results of HMM, and choice between their predictions, made by ExtraTreesRegressor. Additionally, a model may build vectors of n-grams in particular file.


## Installation

* If you intend to use the TreeTagger part of the program, you shall proceed with the [TreeTagger installation instructions](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/). In the original paper, the bulgarian parameters are used, so you need to download them as well.

* The required packages for the program can be downloaded by the below command:

      pip3 install −r requirements.txt
    
## Usage

### Training (modus by default)

The result of program training modus, is .pkl file, containing trained model, called "hmm" (if HMM model is trained), or "{n}-gram.pkl" (if n-gram model is trained, n is chosen via **grammage** argument, prpvided by user). The file is located in **folder**.

* HMM (method by default):

		python  main.py −−data <path> −−split <percentage> −−unknown_to_singleton <0 or 1> −−printSequences <0 or 1> −−folder <path>

	 * **data:** corresponds to the path of the file containing data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
	 * **split:** refers to the split percentage of the data for dividing data into train and test data. By default, it is equal to 90 which means 90% of the data is used in training and 10% of the data is used in testing.
	 * **unknown to singleton:** specifies prediction type of unknown words. If it’s 1, then the unknown words are assumed to act like singleton words. By default it is 0 which means probable tags of unknown words are predicted by morphological patterns and the mean probabilities of these tags are used.
	* **print sequences:** Prints obtained and expected sequences to console, if 1 is chosen. Default value is 0 which means the sequences are not printed.
	* **folder** correspods to the path of the directory where the final model is going to be storaged. By default, the directory, where main.py is located.

* n-gram:

		python main.py −−method grams −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−start_end_symbols <0 or 1> −−weighed <0 or 1> −−tf-idf-coefficient <0 < k < 1> −−length <0 or 1> −−double <0 or 1> −−folder <path>


    * **data:** corresponds to the path of the file containing data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
    * **grammage:** refers to n in n-grams. By default, it is 3, which means that tokens are going to be split into 3-grams.
    * **register_change:** determines, whether tokens are going to be decapitalised (1) or not (0). By default, the parameter is 1.
    * **start_end_symbols:** refers to addition (1) or not (0) of the symbol "#" to the beginning and the end of a particular token. By default, equals to 0, which means that special symbols are not going to be added to the beginning and the end of tokens.
    * **weighed**: determines, whether the result of n-gram frequency analysis is going to be weighed by tf-idf for n-grams (1), or not. By default, the result is not weighed.
    * **tf-idf-coefficient**: specifies the coefficient, used in tf-idf. By default, equals to 0.5.
    * **length**: refers to whether the length of token is used as an additional training parameter for n-gram model, which, this way, may also determine the PoS of token, taking into consideration the length of the latter. By default, equals to 0 (length is not used as an additional training parameter). If length is 1, additional object with the information acquired, "length_n_grams.pkl" (where n is **grammage**), is created in **folder**. 
    * **double**: specifies, whether repeating graphemes (and digraph *оу*) are treated as a single grapheme. By default, equals to 0, which means that each repeating grapheme, and parts of digraphs, is counted. The alternative option is 1: repeating grahemes and digraphs are considered as a single unit in n-gram.
    * **folder** correspods to the path of the directory where the final model is going to be storaged. By default, the directory, where main.py is located.

### Accuracy

Modus for models testing. The results are shown in a terminal.

* HMM:

		python main.py −−modus accuracy −−data <path> −−folder <path>

	* **data:** corresponds to the path of the file containing test data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
	* **folder** correspods to the path of the directory where the previously trained model is storaged. By default, the directory, where main.py is located.

* n-gram:

		python main.py −−modus accuracy −−method grams −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−start_end_symbols <0 or 1> −−length <0 or 1> −−folder <path>

	* **data:** corresponds to the path of the file containing test data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
    * **grammage:** should match the parameter in original model.
    * **register_change:** should match the parameter in original model.
    * **start_end_symbols:** should match the parameter in original model.
    * **length**: should match the parameter in original model.
    * **folder** correspods to the path of the directory where the previously trained model is storaged. By default, the directory, where main.py is located.

* HMM/n-gram hybrid:

		python main.py −−modus accuracy −−method hmmg −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−start_end_symbols <0 or 1> −−folder <path>

	* **data:** corresponds to the path of the file containing test data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
    * **grammage:** should match the parameter in original n-gram model.
    * **register_change:** should match the parameter in original n-gram model.
    * **start_end_symbols:** should match the parameter in original n-gram model.
    * **folder** correspods to the path of the directory where the previously trained models are storaged. By default, the directory, where main.py is located.

* HMM/n-gram/regressor hybrid:

		python main.py −−modus accuracy −−method hmmc −−data <path> −−train_data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−folder <path>

	* **data:** corresponds to the path of the file containing test data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
	* **train_data:** corresponds to the path of the file containing training data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar.
    * **grammage:** should match the parameter in original n-gram model. Required argument.
    * **register_change:** should match the parameter in original n-gram model.
    * **folder** correspods to the path of the directory where the previously trained models are storaged. By default, the directory, where main.py is located.

* TreeTagger:

		python main.py −−modus accuracy −−method tt −−data <path>

	* **data:** corresponds to the path of the file containing test data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.


## Competitive Prediction

Takes three pre-trained models (specifically, HMM, n-gram and rule-enhanced HMM/n-gram hybrid), and compares their predictions for the out-of-domain data. The predictions are saved in predictions.csv file in **folder**.

	python main.py −−modus competitive_prediction −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−folder <path>

* **data:** corresponds to the path of the file containing test data. It should be previously saved as a specific object [see pp. 10-11](https://iling-ran.ru/web/sites/default/files/conferences/2020/2020_lingforum_abstracts.pdf) in JSON. Required argument.
* **grammage:** should match the parameter in original n-gram model.
* **register_change:** should match the parameter in original n-gram model.
* **folder** correspods to the path of the directory where the previously trained models are storaged. By default, the directory, where main.py is located.

## Vectorization

Returns the n-gram vectors in "vectors_n_grams.pkl" file, saved in directory **folder**.

	python main.py −−modus vectorization −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−folder <path>

* **data:** corresponds to the path of the file containing data. It should be in [Universal Dependencies CoNLL-U](https://universaldependencies.org/format.html) format, or similar. Required argument.
* **grammage:** refers to n in n-grams. By default, it is 3, which means that tokens are going to be split into 3-grams.
* **register_change:** determines, whether tokens are going to be decapitalised (1) or not (0). By default, the parameter is 1.
* **folder** correspods to the path of the directory where the previously trained models are storaged. By default, the directory, where main.py is located.

## Prediction

Takes HMM/n-gram hybrid with specifically trained parameters, and the data of the OCS corpus, writing predictions for the tokens of the latter into the file, from where it extracts them.

	python main.py −−modus prediction −−data <path> −−grammage <positive_integer> −−register_change <0 or 1> −−folder <path>

* **data:** corresponds to the path of the file containing test data. It should be previously saved as a specific object [see pp. 10-11](https://iling-ran.ru/web/sites/default/files/conferences/2020/2020_lingforum_abstracts.pdf) in JSON. Required argument.
* **grammage:** should match the parameter in original n-gram model.
* **register_change:** should match the parameter in original n-gram model.
* **folder** correspods to the path of the directory where the previously trained models are storaged. By default, the directory, where main.py is located.
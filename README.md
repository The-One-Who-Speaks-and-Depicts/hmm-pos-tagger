# hmm-pos-tagger

A tool for Old Church Slavonic PoS tagging by different methods. The baseline is Python TreeTagger implementation that uses the Bulgarian language parameters. The basal methods are HMM (enhanced with Viterbi algorithm), and n-gram analysis, that makes PoS prediction on the basis of occurence of the most frequent n-grams of the particular PoS in a token. The hybridisation techniques include using rules (for types of tokens that are not present in the training dataset, yet existing in the language itself, such as fragmentary tokens, punctuation marks, and digits) and n-gram analysis to correct results of HMM, and choice between their predictions, made by ExtraTreesRegressor. Additionally, a model may build vectors of n-grams in particular file.


## Installation

* If you intend to use the TreeTagger part of the program, you shall proceed with the [TreeTagger installation instructions](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/). In the original paper, the bulgarian parameters are used, so you need to download them as well.

* The required packages for the program can be downloaded by the below command:

      pip3 install −r requirements.txt
    
## Usage

### Training

* HMM:

      python  main.py −−data <path> −−split <percentage> −−unknown_to_singleton <0 or 1> −−printSequences <0 or 1> --folder <path>

	 * **data:** corresponds to the path of the file containing data.
	 * **split:** refers to the split percentage of the data for dividing data into train and test data. By default, it is equal to 90 which means 90% of the data is used in training and 10% of the data is used in testing.
	 * **unknown to singleton:** specifies prediction type of unknown words. If it’s 1, then the unknown words are assumed to act like singleton words. By default it is 0 which means probable tags of unknown words are predicted by morphological patterns and the mean probabilities of these tags are used.
	* **print sequences:** Prints obtained and expected sequences to console, if 1 is chosen. Default value is 0 which means the sequences are not printed.
	* **folder** correspods to the path of the directory where the final model is going to be storaged
* N-gram:
      

	*NOTE: Data should be the same format with [METU-Sabancı Turkish Dependency Treebank](https://web.itu.edu.tr/gulsenc/treebanks.html)*

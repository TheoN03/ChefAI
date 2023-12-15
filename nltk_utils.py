import nltk
import numpy as np
# nltk.download("punkt") - run the code once with this!!!
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Is used to the JSON file intents.json. 
    The purpose is to tokenize(break down the JSON file into individual tokens).
    So, it breaks down the file contents into basic units.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Is used to process the words to their root/base form. 
    The purpose is to map different inflections
    or derivations of a word to a common root, so variations of the same 
    word are treated as equivalent.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    it uses the Bag of Words(BoW) technique used in 
    Natural Language Processing(NLP). The basic idea behind the Bag of Words model is 
    to treat a document as an unordered set of words or tokens, disregarding 
    grammar and word order but keeping track of word frequency.
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag

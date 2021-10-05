import nltk 
import numpy as np
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

print('Hi import successful cutie')


def tokenize(sentence):
    return nltk.word_tokenize(sentence) 
    


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words ):
    bag = np.zeros(len(all_words),dtype = np.float32)
    tokenized_sentence = [stem(i) for i in tokenized_sentence]
    for idx,i in enumerate(all_words):
        if i in tokenized_sentence:
            bag[idx] = 1.0
    return  bag
        
     

  





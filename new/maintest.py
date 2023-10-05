from gensim import corpora
import nltk
from gensim.parsing import remove_stopwords
from googletrans import Translator
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import en_core_web_sm
import glob
import re
import os
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import gensim.models.word2vec as w2v

class preprocess3:
    def __init__(self, inputtext):
        self.output = {}
        inText=self.inText(inputtext)
        print("translate:", inText)
        self.output["translate"] = inText

        filter_text = self.filter_text(inText, punctFlag=True, lowerFlag=True, htmlFlag=True, spaceFlag=True)
        print("filter:", filter_text)
        self.output["filter"] = filter_text

        stop_words_sentence = self.stopwords(filter_text)
        print("stopwords:", stop_words_sentence)
        self.output["stopwords"] = stop_words_sentence

        lemop = self.lemop(stop_words_sentence)
        print("lemma:", lemop)
        self.output["Lemmataization"] = lemop

        output = lemop
        self.txtfile(output)
        self.output["output"] = output

    def inText(self,inputtext):
         translator = Translator()
         inText = translator.translate(inputtext, dest="en")
         print(inText.text)
         return inText.text

    def filter_text(self, inText, lowerFlag=True, upperFlag=False, numberFlag=False, htmlFlag=False, urlFlag=False,
                    punctFlag=False, spaceFlag=False, hashtagFlag=False, emojiFlag=False):

        if lowerFlag:
            inText = inText.lower()

        if upperFlag:
            inText = inText.upper()

        if numberFlag:
            import re
            inText = re.sub(r"\d+", '', inText)

        if htmlFlag:
            import re
            inText = re.sub(r'<[^>]*>', '', inText)

        if urlFlag:
            import re
            inText = re.sub(r'(https?|ftp|www)\S+', '', inText)

        if punctFlag:
            import re
            import string
            exclist = string.punctuation  # removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
            # remove punctuations and digits from oldtext
            table_ = inText.maketrans('', '', exclist)
            inText = inText.translate(table_)

        if spaceFlag:
            import re
            inText = re.sub(' +', " ", inText).strip()

        if hashtagFlag:
            pass

        if emojiFlag:
            pass

        return inText

    def stopwords(self, filter_text):

        filtered_sentence = remove_stopwords(filter_text)

        stop_words_sentence = nltk.word_tokenize(filtered_sentence)
        return stop_words_sentence

    def lemop(self, word_list):
        lemmatizer = WordNetLemmatizer()
        lemop = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

        print(lemop)

        return lemop


        # > the bat see the cat with good stripe hang upside down by -PRON- foot





    def output(self):
        for x, v in self.output.items():
            print(x, v)
        return self.output

    def txtfile(self, output):
        result = r'C:\Users\sharayu\PycharmProjects\multilingual\docdb'
        locs = glob.glob(r"C:\Users\sharayu\PycharmProjects\multilingual\docdb\*.txt")
        print(locs)
        names = [w.split("/")[-1] for w in locs]
        arr = []
        for n in names:
            arr.append(max(re.findall(r'\d+', n)))
        print(max([int(a) for a in arr]))
        final_num = 1 + max([int(a) for a in arr])
        print(final_num)

        with open(result + r"\docdb" + str(final_num) + ".txt", "w+", encoding="utf-8") as ff:
            ff.write(str(output))
            ff.close()



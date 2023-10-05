# Core Packages

import tkinter as tk
from tkinter import *
from tkinter import ttk
import os.path

import gensim
import tkinter as tk

import numpy as np
from gensim.models import KeyedVectors
import nltk
from googletrans import Translator
import os
from stop_words import get_stop_words
from nltk import WordNetLemmatizer
from maintest import preprocess3 as g
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Structure and Layout
window = Tk()
window.title("Output")
window.geometry("700x400")
window.config(background='black')
model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\sharayu\pycharmm\textsimilarity\result\GoogleNews-vectors-negative300.bin',binary=True)

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn', )

# TAB LAYOUT
tab_control = ttk.Notebook(window, style='lefttab.TNotebook')

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)


# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Suspecious Document":^20s}')
tab_control.add(tab2, text=f'{"Data":^20s}')


label1 = Label(tab1, text='Text Similarity', padx=5, pady=5)
label1.grid(column=0, row=0)
label2 = Label(tab2, text='Text Similarity', padx=5, pady=5)
label2.grid(column=0, row=0)


tab_control.pack(expand=1, fill='both')

final = {}


# Functions
def translate_text(text, target_language='en'):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text


def preprocess_hindi(text):
    # lowercase the text
    stop_words = get_stop_words('hi')

    text = text.lower()

    # tokenize the text
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatize the tokens

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # return the preprocessed text as a string
    return ' '.join(tokens)


def calculate_similarity(suspicious_file_path, english_file_path):
    # read the suspicious Hindi file
    with open(suspicious_file_path, 'r', encoding='utf-8') as f:
        hindi_text = f.read()

    # preprocess the Hindi text
    preprocessed_hindi = preprocess_hindi(hindi_text)

    # translate Hindi text to English

    translated_text = translate_text(' '.join(preprocessed_hindi))
    # tokenize the English text
    english_tokens = nltk.word_tokenize(translated_text)

    # remove out-of-vocabulary tokens
    english_tokens = [token for token in english_tokens if token in model.key_to_index]

    # calculate the vector representation of the English text
    vector_1 = sum([model[token] for token in english_tokens])

    # read the English file
    with open(english_file_path, 'r', encoding='utf-8') as f:
        english_text = f.read()

    # tokenize the English text
    english_tokens_2 = nltk.word_tokenize(english_text)

    # remove out-of-vocabulary tokens
    english_tokens_2 = [token for token in english_tokens_2 if token in model.key_to_index]

    # calculate the vector representation of the English text
    vector_2 = sum([model[token] for token in english_tokens_2])

    # calculate the cosine similarity between the vectors
    similarity = vector_1.dot(vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))


    return similarity



def on_button_click():
    suspicious_file_path = entry_1.get()
    dir_path = entry_2.get()

    # Find all files in directory
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # Compute similarity with each file
    similarities = []
    for file_path in file_paths:
        similarity = calculate_similarity(suspicious_file_path,file_path)
        similarities.append(similarity)

    # Display results
    for i, similarity in enumerate(similarities):
        label = Label(window, text=f'{file_paths[i]}: {similarity:.2f}')
        label.grid(row=i, column=0, padx=10, pady=10)
def process():
    global final
    raw_text = str(entry.get('1.0', tk.END))
    final = g(raw_text).output

def translate():
    global final
    result = '\ntranslate :{}\n'.format(final["translate"])
    tab1_display.insert(tk.END, result)

def filter():
    global final
    result = '\nfilter :{}\n'.format(final["filter"])
    tab1_display.insert(tk.END, result)


def stopword():
    global final
    result = '\nstopword:{}\n'.format(final["stopwords"])
    tab1_display.insert(tk.END, result)


def lemma():
    global final
    result1 = '\nLemmataization:{}\n'.format(final["Lemmataization"])
    tab1_display.insert(tk.END, result1)



def clear_text():
    global final
    entry.delete('1.0', END)


def clear_display_result():
    tab1_display.delete('1.0', END)



# MAIN NLP TAB
l1 = Label(tab1, text="Enter Text To Summarize")
l1.grid(row=1, column=0)

entry = Text(tab1, height=10, width=130)
entry.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

# BUTTONS
button1 = Button(tab1, text="process", command=process, width=12, bg='#03A9F4', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(tab1, text="filter", command=filter, width=12, bg='#03A9F4', fg='#fff')
button2.grid(row=4, column=1, padx=10, pady=10)

button3 = Button(tab1, text="Clear Result", command=clear_display_result, width=12, bg='#03A9F4', fg='#fff')
button3.grid(row=5, column=0, padx=10, pady=10)

button4 = Button(tab1, text="stop words", width=12, command=stopword, bg='#03A9F4', fg='#fff')
button4.grid(row=5, column=1, padx=10, pady=10)

button5 = Button(tab1, text="lemmatization", width=12, command=lemma, bg='#03A9F4', fg='#fff')
button5.grid(row=6, column=0, padx=10, pady=10)

button6 = Button(tab1, text="Translate", width=12, command=translate, bg='#03A9F4', fg='#fff')
button6.grid(row=6, column=1, padx=10, pady=10)

label_1 = Label(tab1, text='Suspicious File:')
label_1.grid(row=7, column=1, padx=10, pady=10)

entry_1 = Entry(tab1)
entry_1.grid(row=7, column=0, padx=10, pady=10)

label_2 = Label(tab1, text='Directory Path:')
label_2.grid(row=8, column=1, padx=10, pady=10)

entry_2 = Entry(tab1)
entry_2.grid(row=8, column=0, padx=10, pady=10)

button =Button(tab1, text='Compute Similarities', command=on_button_click, bg='#03A9F4', fg='#fff')
button.grid(row=9, column=1, padx=10, pady=10)



# Display Screen For Result
tab1_display = Text(tab1, width=130)
tab1_display.grid(row=15, column=0, columnspan=3, padx=5, pady=5)



window.mainloop()
# To become more successful i
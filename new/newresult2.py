import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
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
# Import the functions from your original code here
model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\sharayu\PycharmProjects\multilingual\new\GoogleNews-vectors-negative300.bin',binary=True)

# Create the Tkinter window
window = tk.Tk()
window.title("Text Similarity of Multilingual(Hindi-English) text ")
window.geometry("800x600")

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
        # translate Hindi text to English
    translated_text = translate_text(' '.join(hindi_text))

    # preprocess the Hindi text
    preprocessed_hindi = preprocess_hindi(translated_text)


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
        label = Label(tab3, text=f'{file_paths[i]}: {similarity:.2f}')
        label.grid(row=i + 2, column=1, padx=10, pady=10)

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

def clear_similarity_result():
    tab3_display.delete('1.0', END)

# Create labels and entry widgets
label_1 = ttk.Label(window, text="Enter path of suspicious file:")
label_1.grid(row=0, column=0, padx=10, pady=10)

entry_1 = ttk.Entry(window, width=50)
entry_1.grid(row=0, column=1, padx=10, pady=10)

label_2 = ttk.Label(window, text="Enter directory path of English files:")
label_2.grid(row=1, column=0, padx=10, pady=10)

entry_2 = ttk.Entry(window, width=50)
entry_2.grid(row=1, column=1, padx=10, pady=10)

# Create buttons
browse_button = ttk.Button(window, text="Browse", command=lambda: entry_1.insert(0, filedialog.askopenfilename()))
browse_button.grid(row=0, column=2, padx=10, pady=10)

browse_dir_button = ttk.Button(window, text="Browse", command=lambda: entry_2.insert(0, filedialog.askdirectory()))
browse_dir_button.grid(row=1, column=2, padx=10, pady=10)

check_button = ttk.Button(window, text="Text Similarity", command=on_button_click)
check_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

clear_result_button = ttk.Button(window, text="Clear Results", command=clear_display_result)
clear_result_button.grid(row=2, column=1, padx=10, pady=10)

clear_similarity_result = ttk.Button(window, text="Clear Results", command=clear_similarity_result)
clear_similarity_result.grid(row=2, column=2,padx=10, pady=10)
# Create a tab control with two tabs
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text="Preprocessing")
tab_control.add(tab2, text="Results")
tab_control.add(tab3, text="Similarity Results")
tab_control.grid(row=3, column=0, columnspan=3, padx=10, pady=10)


# Functions



# Create widgets for the preprocessing tab
label_tab1 = ttk.Label(tab1, text="Preprocessing Steps")
label_tab1.grid(row=0, column=0, padx=10, pady=10)

button1 =ttk.Button(tab1, text="process", command=process)
button1.grid(row=4, column=0, padx=10, pady=10)

translate_button = ttk.Button(tab1, text="Translate", command=translate)
translate_button.grid(row=4, column=1, padx=10, pady=10)

filter_button = ttk.Button(tab1, text="Filter", command=filter)
filter_button.grid(row=5, column=0, padx=10, pady=10)

stopword_button = ttk.Button(tab1, text="Stopwords", command=stopword)
stopword_button.grid(row=5, column=1, padx=10, pady=10)

lemma_button = ttk.Button(tab1, text="Lemmataization", command=lemma)
lemma_button.grid(row=6, column=0, padx=10, pady=10)

clear_button = ttk.Button(tab1, text="Clear Text", command=clear_text)
clear_button.grid(row=6, column=1, padx=10, pady=10)

entry = Text(tab1, height=10, width=130)
entry.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

# Create a text widget to display the results
tab1_display = tk.Text(tab2)
tab1_display.grid(row=0, column=0, padx=10, pady=10)

tab3_display = tk.Label(tab3)
tab3_display.grid(row=0, column=0, padx=10, pady=10)

# Start the Tkinter event loop
window.mainloop()

import pandas as pd
import numpy as np
import gensim
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os

import numpy
import threading

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import regex as re
import string

from ipywidgets import interact, widgets
from IPython.display import display
from sklearn.metrics import accuracy_score, f1_score

import speech_recognition as sr
from os import path

from matplotlib.ticker import MaxNLocator


rootdir = './CrisisNLP_labeled_data_crowdflower_v2'
ext = '.tsv'
data = pd.DataFrame()
for root, subdirs, files in os.walk(rootdir):
    for file in files:
        if file != "2014_Chile_Earthquake_cl_labeled_data.tsv": #remove spanish one
            f_ext = os.path.splitext(file)[-1].lower()
            if f_ext == ext:
                path = os.path.join(root,file)
                data_event = pd.read_csv(path, sep='\t')
                data = pd.concat([data, data_event], ignore_index=True)
# data


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def processData(data, labels, model):
    punctuation = list(string.punctuation)
    stop = stopwords.words('english')
    sap = punctuation+stop+['RT','【','】','']
    twt = TweetTokenizer(strip_handles=True, reduce_len=True)
    
    # Format text
    data['tweet_text_token'] = data['tweet_text'].apply(remove_URL)
    data['tweet_text_token'] = data['tweet_text_token'].apply(twt.tokenize)
    data['tweet_text_token'] = data['tweet_text_token'].apply(lambda x: [w.replace('#','') for w in x])
    data['tweet_text_token'] = data['tweet_text_token'].apply(lambda x: [word for word in x if word not in sap])
    data['tweet_text_token'] = data['tweet_text_token'].apply(lambda x: [wnl.lemmatize(word) for word in x])
    data['tweet_text_token'] = data['tweet_text_token'].apply( lambda x: [word.lower() for word in x])
#     data['wvec'] = data['tweet_text_token'].apply(lambda x: np.array(np.mean([model[w] for w in x if w in model.vocab], axis=0)))

    # Create embedding for each word and calculate mean for sentence
    data['wvec'] = data['tweet_text_token'].apply(lambda x: np.array(np.mean([model[w] for w in x if w in model], axis=0)))
    labels = labels.drop(data[data['wvec'].isna() == True].index)
    data = data.drop(data[data['wvec'].isna() == True].index)
    return np.vstack(np.array(data['wvec'])), labels


model = gensim.models.KeyedVectors.load_word2vec_format('crisisNLP_word2vec_model_v1.2/crisisNLP_word2vec_model/crisisNLP_word_vector.bin', binary=True)


fulltrain, test = train_test_split(data, test_size=0.3, random_state=42)
train, val = train_test_split(fulltrain, test_size=0.2, random_state=42)
Xtrain, ytrain = processData(train, train['label'], model)
Xval, yval = processData(val, val['label'], model)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(np.array(Xtrain), np.array(ytrain))
logreg.fit(np.array(Xtrain), np.array(ytrain))
y_pred = logreg.predict(np.array(Xval))
print('Testing accuracy %s' % accuracy_score(yval, y_pred))
print('Testing F1 score: {}'.format(f1_score(yval, y_pred, average='weighted')))



def classifyText(text, live_data):
    # replace by something useful
    print(text)
    textinput = text
    punctuation = list(string.punctuation)
    stop = stopwords.words('english')
    sap = punctuation+stop+['RT','【','】','']
    twt = TweetTokenizer(strip_handles=True, reduce_len=True)
    
    # Format text
    text = remove_URL(textinput)
    text = twt.tokenize(text)
    text = [w.replace('#','') for w in text]
    text = [word for word in text if word not in sap]
    text = [wnl.lemmatize(word) for word in text]
    text = [word.lower() for word in text]
    
    # Create embedding for each word and calculate mean for sentence
    vec = np.array(np.mean([model[w] for w in text if w in model], axis=0))
    vec = vec.reshape(1, -1)
    
    # Predict the class of the sentence using the average embedding 
    y_pred = logreg.predict(np.array(vec))
    
    live_data.append(y_pred[0])
    # update_line(old_data, y_pred)

    print(y_pred)
    


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response accordingly
    try:
        response = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response = None
        print("API unavailable")
    except sr.UnknownValueError:
        # speech was unintelligible
        response = None
        print("Unable to recognize speech")

    return response

# def start_speech(path):
def start_speech(live_data):
    closingWords = ['thank you', 'thanks', 'later', 'bye bye', 'bye', 'good night']
    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    listening = True
    old_data = []
    
    while listening:
    
        while True:
            words = recognize_speech_from_mic(recognizer, microphone)
            if words:
                break
        
        classifyText(words, live_data)
        for word in closingWords:
            if word in words:
                listening = False
                os.system("TASKKILL /IM winword.exe")



# speechChecker = threading.Thread(target=start_speech,args=(path,))
print('starting speech')
live_data = []
speechChecker = threading.Thread(target=start_speech,args=(live_data,))
speechChecker.start()   

fg = plt.figure()
ax = fg.gca()
hl = plt.hist(live_data, bins=15, align='left')
# plt.xticks(rotation=90);

l_curr = len(live_data)
l_prev = 0
while True:

    if l_curr != l_prev:
        print('plotting')
        plt.cla()
        # hl = plt.hist(live_data, bins=15, align='left')
        hl = plt.hist(live_data, bins=15, align='mid')
        # plt.tight_layout()
        # hl = plt.hist(live_data, bins=np.arange(15)-0.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # hl = plt.hist(live_data, bins=np.arange(15)-0.5, align='left')
        # hl = plt.hist(live_data, bins=15, align='left',orientation='horizontal')
        plt.xticks(rotation=-12);
        plt.xticks(fontsize=6)
        # plt.show()
        # plt.draw()
        plt.pause(1)
        l_prev = l_curr
    else:
        l_curr = len(live_data)

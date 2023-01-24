#Data management
import pandas as pd
import numpy as np
np.random.seed(0)
#from pandas_profiling import ProfileReport

#TextBlob Features
from textblob import TextBlob

#Plotting
import matplotlib.pyplot as plt

#SciKit-Learn
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#to save the trained model
import pickle

#nltk
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')

#Tensorflow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras

#seaborn
import seaborn as sns

#Test
from collections import Counter

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["GET","POST"])
def predict():

    #inp_text = "I hate Facebook"
    #inp_entity = "Facebook"

    inp_entity = request.form['inp_entity']
    inp_text = request.form['inp_text']
    
    
    new_inp_list = ["12345", inp_entity, "Positive", inp_text]

    #Training Data
    #path = "../datasets/twitter_training.csv"
    #train_df = pd.read_csv(path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"])

    #Test Data (Not to be used until the full model has been trained)
    test_path = "../datasets/twitter_validation.csv"
    test_df = pd.read_csv(test_path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"])


    test_df.iloc[0] = new_inp_list

    #drop nulls
    #train_df = train_df.dropna()
    test_df = test_df.dropna()

    #drop duplicates
    #train_df.drop_duplicates(inplace=True)

    #df = train_df

    #onehot = pd.get_dummies(df["Entity"], prefix="Entity")

    #Join these new columns back into the DataFrame
    #df = df.join(onehot)

    ##Test
    onehot = pd.get_dummies(test_df["Entity"], prefix="Entity")

    test_df = test_df.join(onehot)

    #single input
    #onehot = pd.get_dummies(inp_df["Entity"], prefix="Entity")
    #inp_df = inp_df.join(onehot)

    #Adding dimensions with textblob
    def tb_enrich(ls):
        
        tb_polarity = []
        tb_subject = []

        for tweet in ls:
            tb_polarity.append(TextBlob(tweet).sentiment[0])
            tb_subject.append(TextBlob(tweet).sentiment[1])
        

        return tb_polarity, tb_subject

    #Enrich using TextBlob's built in sentiment analysis

    ##Train
    #df["Polarity"], df["Subjectivity"] = tb_enrich(list(df["Tweet_Content"]))

    
    ##Test
    test_df["Polarity"], test_df["Subjectivity"] = tb_enrich(list(test_df["Tweet_Content"]))

    #single input
    #inp_df["Polarity"], inp_df["Subjectivity"] = tb_enrich(list(inp_df["Tweet_Content"]))

    #Define the indexing for each possible label in a dictionary
    class_to_index = {"Neutral":0, "Irrelevant":1, "Negative":2, "Positive": 3}

    #Creates a reverse dictionary
    index_to_class = dict((v,k) for k, v in class_to_index.items())

    #Creates lambda functions, applying the appropriate dictionary
    names_to_ids = lambda n: np.array([class_to_index.get(x) for x in n])
    ids_to_names = lambda n: np.array([index_to_class.get(x) for x in n])

    #Convert the "Sentiment" column into indexes

    ##Train
    #df["Sentiment"] = names_to_ids(df["Sentiment"])
    #y = df["Sentiment"]

        ##Test
    test_df["Sentiment"] = names_to_ids(test_df["Sentiment"])
    y_test = test_df["Sentiment"]

    #single input
    #inp_df["Sentiment"] = names_to_ids(inp_df["Sentiment"])
    #y_inp = inp_df["Sentiment"]


    #Removing stopwords and lemmatising
    lemmatiser = WordNetLemmatizer()
    stop_english = Counter(stopwords.words())

    def remove_stopwords(ls):
        #Lemmatises, then removes stop words
        ls = [lemmatiser.lemmatize(word) for word in ls if word not in (stop_english) and (word.isalpha())]
        
        #Joins the words back into a single string
        ls = " ".join(ls)
        return ls

    ##Train
    #Splits each string into a list of words
    #df["Tweet_Content_Split"] = df["Tweet_Content"].apply(word_tokenize)

    #Applies the above function to each entry in the DataFrame
    #lemmatiser = WordNetLemmatizer()
    #stop_english = Counter(stopwords.words()) #Here we use a Counter dictionary on the cached
                                            # list of stop words for a huge speed-up
    #df["Tweet_Content_Split"] = df["Tweet_Content_Split"].apply(remove_stopwords)


    ##Test
    test_df["Tweet_Content_Split"] = test_df["Tweet_Content"].apply(word_tokenize)

    #print(test_df["Tweet_Content_Split"])

    test_df["Tweet_Content_Split"] = test_df["Tweet_Content_Split"].apply(remove_stopwords)

    #print(test_df["Tweet_Content_Split"])

        #single input
    #inp_df["Tweet_Content_Split"] = inp_df["Tweet_Content"].apply(word_tokenize)

    #inp_df["Tweet_Content_Split"] = inp_df["Tweet_Content_Split"].apply(remove_stopwords)

    #Tokenisation

    #Define the Tokeniser
    #tokeniser = Tokenizer(num_words=1000, lower=True)

    #Create the corpus by finding the most common 
    #tokeniser.fit_on_texts(df["Tweet_Content_Split"])

    ##Train
    #Tokenise our column of edited Tweet content
    #tweet_tokens = tokeniser.texts_to_matrix(list(df["Tweet_Content_Split"]))

    # loading the tokenizer
    with open('../saved_tokenizer.pkl', 'rb') as handle:
        tokeniser = pickle.load(handle)
    ##Test
    #Tokenise our column of edited Tweet content
    tweet_tokens_test = tokeniser.texts_to_matrix(list(test_df["Tweet_Content_Split"]))

    #tweet_tokens_inp = tokeniser.texts_to_matrix(list(test_df["Tweet_Content_Split"]))

    #Combining the dataframe with the tokens using pd.concat

    #Reset axes to avoid overlapping
    #df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    ##Train
    #full_df = pd.concat([df, pd.DataFrame(tweet_tokens)], sort=False, axis=1)

    ##Test
    full_test_df = pd.concat([test_df, pd.DataFrame(tweet_tokens_test)], sort=False, axis=1) 

    #single input
    #inp_df.reset_index(drop=True, inplace=True)

    #full_inp_df = pd.concat([inp_df, pd.DataFrame(tweet_tokens_inp)], sort=False, axis=1)

    #Final prep

    ##Train
    #Drop all non-useful columns
    #full_df = full_df.drop(["Sentiment", "Tweet_ID", "Tweet_Content", "Tweet_Content_Split", "Entity"], axis=1)


    ##Test
    full_test_df = full_test_df.drop(["Sentiment", "Tweet_ID", "Tweet_Content", "Tweet_Content_Split", "Entity"], axis=1)

    #adding input
    #new_s = full_test_df.loc[0]
    #new_s = new_s.to_frame().T

    #Single Input
    #full_inp_df = full_inp_df.drop(["Sentiment", "Tweet_ID", "Tweet_Content", "Tweet_Content_Split", "Entity"], axis=1)

    # It can be used to reconstruct the model identically.
    reconstructed_model = keras.models.load_model("../saved_model")

    y_pred = np.argmax(reconstructed_model.predict(full_test_df), axis=1)

    #Assign labels to predictions and test data
    y_pred_labels = ids_to_names(y_pred)
    y_test_labels = ids_to_names(y_test)

    # y_unique = list(set(y_test_labels))
    # cm = confusion_matrix(y_test_labels, y_pred_labels, labels = y_unique, nobrmalize='true')

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_unique)
    # disp.plot()

    #print(classification_report(y_pred_labels, y_test_labels))

    #print(y_pred_labels)

    result = str(y_pred_labels[0])

    return render_template("result.html", result = result)

if __name__=="__main__":
    app.run(debug=True)


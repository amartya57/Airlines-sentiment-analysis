# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:40:52 2021

@author: Amartya057
"""
'''Importing libraries required for API'''
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

'''Importing libraries required to load and run the model'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

'''Importing libraries for preprocessing of data'''
import re
import emoji


'''Defining parameters for data preprocessing'''
max_len=32
trunc_type="post"
pad_type="post"


'''Creating the app instance'''
app=FastAPI()

'''
Creating the base url
It contains a simple function that returns a welcome message
'''
@app.get('/')
def welcome():
    return{"Welcome":"Welcome to sentiment classification"}


'''
Creating a route where the text is directly passed in the url.
A JSON response will be returned which would classify the text as
positive or negative
'''
@app.get('/classify/{data}')
def classify_sentiment_get(data):

    '''loading the tokenizer which was saved as a pickle file in the same level
    as the main.py
    The tokenizer is later applied on the input text which converts it into
    numbers suitable for feeding in to the model'''
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    '''Preprocessing the input text'''
    def preprocess(s):
        '''Removing the tags and hashtags'''
        s = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#+)", " ", s).split())
        
        '''Removing the urls from the text'''
        s = ' '.join(re.sub("(\w+:\/\/\S+)", " ", s).split())
        
        '''Converting the emojis into text'''
        s = emoji.demojize(s)
        s = s.replace(":"," ")
        s = s.replace("_"," ")
        s = ' '.join(s.split())
        s = s.lower()
        
        return s
    
    '''Converting the input data into list'''
    data=[preprocess(data)]
    
    '''Applying the tokenizer on the list data to convert it from
    texts to sequences'''
    data=tokenizer.texts_to_sequences(data)
    
    '''Applying padding on the sequence to make them of uniform length'''
    data=pad_sequences(data, maxlen=max_len, truncating=trunc_type, padding=pad_type)
    
    '''Loading the trained model which was saved in the same level as main.py'''
    model=tf.keras.models.load_model("my_model")
    
    '''Making prediction for the input text'''
    y_hat=model.predict(data)[0][0]
    
    '''Returning a prediction corresponding to the predicted value'''
    if(y_hat>0.5):
        return {'Sentiment':"Positive review"}
    else:
        return {'Sentiment':"Negative review"}


'''
Creating a route with post method, where the text is given as a query.
A JSON response will be returned which would classify the text as
positive or negative
'''
@app.post('/classify')
def classify_sentiment(data):
    '''loading the tokenizer which was saved as a pickle file in the same level
    as the main.py
    The tokenizer is later applied on the input text which converts it into
    numbers suitable for feeding in to the model'''
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    '''Preprocessing the input text'''
    def preprocess(s):
        '''Removing the tags and hashtags'''
        s = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#+)", " ", s).split())
        
        '''Removing the urls from the text'''
        s = ' '.join(re.sub("(\w+:\/\/\S+)", " ", s).split())
        
        '''Converting the emojis into text'''
        s = emoji.demojize(s)
        s = s.replace(":"," ")
        s = s.replace("_"," ")
        s = ' '.join(s.split())
        s = s.lower()
        
        return s
    
    '''Converting the input data into list'''
    data=[preprocess(data)]
    
    '''Applying the tokenizer on the list data to convert it from
    texts to sequences'''
    data=tokenizer.texts_to_sequences(data)
    
    '''Applying padding on the sequence to make them of uniform length'''
    data=pad_sequences(data, maxlen=max_len, truncating=trunc_type, padding=pad_type)
    
    '''Loading the trained model which was saved in the same level as main.py'''
    model=tf.keras.models.load_model("my_model")
    
    '''Making prediction for the input text'''
    y_hat=model.predict(data)[0][0]
    
    '''Returning a prediction corresponding to the predicted value'''
    if(y_hat>0.5):
        return {'Sentiment':"Positive review"}
    else:
        return {'Sentiment':"Negative review"}
    

'''Creating the custom swagger api documentation'''
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentiment Analysis",
        version="1.0.0",
        description="This is an API which classifies Airllines review from twitter as positive or negative",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

'''Running the API on the desired port'''
if(__name__=="__main__"):
    uvicorn.run(app, host='127.0.0.1', port=8000)



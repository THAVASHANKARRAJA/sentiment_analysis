
import re
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
#import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle

def image_path(selected_product):
    if selected_product == "American Tourist Bag":
        return "products/products\American Tourist Bag.png"
    if selected_product == "Iphone 15 Pro":
        return "products/products\Iphone 15 Pro.jpg"
    if selected_product == "Refrigerator (Samsung)":
        return "products/products\Refrigerator (Samsung).png"
    if selected_product == "Amazon Echo":
        return "products/products\Amazon Echo.jpg"
    if selected_product == "Dell XPS Laptop":
        return "products/products\Dell XPS laptop.jpg"
    if selected_product == "Test Product":
        return "products/products\product.png"

def preprocess_reviews(review,max_length,token):
    # Pre-process input
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)
    print('Cleaned: ', review)
    
    english_stops = set(stopwords.words('english'))
    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]
    print('Filtered: ', filtered)
    
    tokenize_words = token.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')

    return tokenize_words

def show_review(selected_product):
    pass
    
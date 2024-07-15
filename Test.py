from tensorflow.keras.models import load_model 
from preprocessing import preprocess_reviews
import pickle

model = load_model('models/LSTM.h5')
token = pickle.load((open('token.pkl', 'rb')))

max_len = 130
review = "This product is awesome"

review_encoded = preprocess_reviews(review, max_len,token)

result = model.predict(review_encoded)
print(result)
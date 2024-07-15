from tensorflow.keras.models import load_model 
from preprocessing import preprocess_reviews
import streamlit as st
import pickle
model = load_model('models/LSTM.h5')
token = pickle.load((open('token.pkl', 'rb')))
max_len = 130

def home():
    st.title("")
    st.write("Welcome to the Home Page!")

    

def find_star(result):
    if result >= 0.8:
        st.markdown("<span style='color:green;font-size: 80px'>* * * * *</span>", unsafe_allow_html=True)
    elif result>0.65:
        st.markdown("<span style='color:grreen;font-size: 80px'>* * * *</span>", unsafe_allow_html=True)
    elif result>0.5:
        st.markdown("<span style='color:white;font-size: 80px'>* * *</span>", unsafe_allow_html=True)
    elif result>0.3:
        st.markdown("<span style='color:red;font-size: 80px'>* *</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;font-size: 80px'>*</span>", unsafe_allow_html=True)
        

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
  background-size: cover;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)

#st.title("")
st.markdown("<h1 style='text-align: center; color: yellow;'>Product Review Analysis</h1>", unsafe_allow_html=True)

#st.markdown("<h2 style='text-align: center; color: black;'>Smaller headline in black </h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Original image</p>'
    #review = st.markdown(original_title, unsafe_allow_html=True)
    st.markdown("<span style='color:white;font-size: 20px'>Enter the review1</span>", unsafe_allow_html=True)
    #st.text_area("This is the text area")
    review1 = st.text_area("Enter the review1",label_visibility = 'collapsed')
    review1 = str(review1)
    
with col2:
    st.markdown("<span style='color:white;font-size: 20px'>Enter the review2</span>", unsafe_allow_html=True)
    review2 = st.text_area("Enter the review2",label_visibility = 'collapsed')
    review2 = str(review2)


if st.button("Find Rating "):
    review_encoded1 = preprocess_reviews(review1, max_len,token)
    review_encoded2 = preprocess_reviews(review2, max_len,token)
    result1 = model.predict(review_encoded1)
    result2 = model.predict(review_encoded2)
    with col1:
        find_star(result1)
    with col2:
        find_star(result2)

        
        
        
        




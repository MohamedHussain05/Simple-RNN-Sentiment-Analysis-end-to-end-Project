import streamlit as st
import numpy as np
from tensorflow.keras .datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN

word_index=imdb.get_word_index()
real={ value:key for key,value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')


#converted1 = ' '.join([real.get(i-3,'not_found') for i in sample])
#converted1 = [real[i] for i in sample]

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,0)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preproceesed_input=preprocess_text(review)
    prediction=model.predict(preproceesed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

#Streamlit development
st.title('IMDB Movie Review Sentiment Analysis')

user_input=st.text_area('Type the Review')

if st.button('classify'):
    preprocess_input=predict_sentiment(user_input)
    
    st.write(preprocess_input)
    
else:
    st.write('Please Type movie name')
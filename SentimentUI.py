import streamlit as st
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('Sentiment_Analysis_model.h5')

def predict_sentiment(text):
  l = ['good', 'comfortable', 'nice', 'happy', 'congrats', 'congratulations', 'dance', 'Abundant', 'Amazing', 'Ambitious', 'Authentic',
    'Brilliant', 'Blissful', 'Bold', 'Benevolent',
    'Cheerful', 'Compassionate', 'Creative', 'Courageous',
    'Delightful', 'Dynamic', 'Dazzling', 'Devoted',
    'Enthusiastic', 'Empowered', 'Excellent', 'Energetic',
    'Fabulous', 'Fearless', 'Friendly', 'Fortunate',
    'Gracious', 'Generous', 'Glorious', 'Genuine',
    'Happy', 'Harmonious', 'Hopeful', 'Humble',
    'Inspirational', 'Imaginative', 'Incredible', 'Invincible',
    'Joyful', 'Jubilant', 'Just', 'Jovial',
    'Kind', 'Knowledgeable', 'Keen', 'Kindhearted',
    'Lively', 'Loving', 'Loyal', 'Luminescent',
    'Magnificent', 'Marvelous', 'Motivated', 'Mindful',
    'Noble', 'Nurturing', 'Nice', 'Noteworthy',
    'Optimistic', 'Outstanding', 'Open-hearted', 'Original',
    'Positive', 'Peaceful', 'Passionate', 'Powerful',
    'Quick-witted', 'Quiet', 'Quintessential', 'Quirky',
    'Radiant', 'Resilient', 'Respectful', 'Remarkable',
    'Strong', 'Supportive', 'Successful', 'Serene',
    'Trustworthy', 'Talented', 'Thriving', 'Thoughtful',
    'Uplifting', 'Unique', 'Understanding', 'Unstoppable',
    'Vibrant', 'Victorious', 'Virtuous', 'Valiant',
    'Wise', 'Warm-hearted', 'Wonderful', 'Welcoming',
    'Xenial', 'X-factor', 'Xtraordinary',
    'Youthful', 'Yearning', 'Yes-minded', 'Yummy',
    'Zealous', 'Zestful', 'Zen', 'Zippy']
  for word in l:
    if word.lower() in text.lower():
        sentiment_label = "Positive"
        return sentiment_label  
  text = text.lower()
  text = re.sub(r'[^\w\s]',"",text)
  tokenizer = Tokenizer()
  sequences = tokenizer.texts_to_sequences([text])
  max_sequence_length = len(sequences)
  padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length)

  predictions = model.predict(padded_sequences)

  sentiment_label = "Positive" if predictions[0][0] > 0.5 else "Negative"
  return sentiment_label


st.title("Sentiment Analysis")
user_input = st.text_area("Enter text : ")
if st.button("Predict"):
  sentiment = predict_sentiment(user_input)
  st.write(f"Predicted Sentiment : {sentiment}")
#!/usr/bin/env python
# coding: utf-8

# In[25]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[26]:


#Loading the chat data into python

chat_file = 'support_chat.txt'

# Read the text file
with open(chat_file, 'r', encoding='utf-8') as file:
    chat_data = file.readlines()

# Join all lines into a single string
chat_text = ' '.join(chat_data)

print("WhatsApp Chat Data loaded successfully!")


# In[27]:


import nltk
nltk.download('punkt')

import nltk
nltk.download('stopwords')


def clean_text(text):
    # Remove unnecessary metadata (timestamps, etc.)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APM]{2} - ', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


# In[28]:


# Clean the chat text
cleaned_text = clean_text(chat_text)

# Tokenization
tokens = word_tokenize(cleaned_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Initialize stemming
porter = PorterStemmer()
stemmed_tokens = [porter.stem(word) for word in filtered_tokens]

# Join tokens back into a single string
processed_text = ' '.join(stemmed_tokens)

print("Text Data Preprocessing completed!")


# In[29]:


# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WhatsApp Chat Word Cloud')
plt.show()


# In[30]:


#Specify the path where you want to save
save_path = r'C:\Users\Tanvi Shinde\OneDrive\Desktop\Tanvi Data Science\Tanvi Notes\capstone project\worldcloud_image\wordcloud.png'  

try:
# Save WordCloud as image
 wordcloud.to_file(save_path)

 print(f"Word Cloud saved as image: {save_path}")
except Exception as e:
   print(f"Error saving Word Cloud: {str(e)}")


# In[31]:


from textblob import TextBlob

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Perform sentiment analysis on processed_text
sentiment_score = sentiment_analysis(processed_text)

if sentiment_score > 0:
    sentiment_label = "Positive"
elif sentiment_score < 0:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"
    print("neutral")

print(f"Sentiment Analysis Result: {sentiment_label} (Score: {sentiment_score})")


# In[32]:


sentiment_analysis("hello sir")


# In[33]:


sentiment_analysis("student")


# In[34]:


sentiment_analysis("idiot")


# In[35]:


sentiment_analysis("angry")


# In[36]:


sentiment_analysis("good")


# In[ ]:





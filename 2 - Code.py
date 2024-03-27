# Importing necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Loading data
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, usecols=[0, 5])
df.columns = ['sentiment', 'text']
df = df.sample(100000)  # Taking a sample of 100,000 tweets for faster processing

# Preprocessing function to clean and tokenize tweets
def preprocess_tweet(tweet):
    # Remove all the URLs
    tweet = re.sub(r"http\S+", "", tweet)
    # Remove all the special characters
    tweet = re.sub(r'\W', ' ', tweet)
    # Remove all single characters
    tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
    # Remove single characters from the start
    tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', tweet)
    # Substituting multiple spaces with single space
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
    # Removing prefixed 'b'
    tweet = re.sub(r'^b\s+', '', tweet)
    # Converting to Lowercase
    tweet = tweet.lower()
    # Tokenizing
    tweet = tweet.split()
    # Removing stopwords
    stop_words = stopwords.words('english')
    tweet = [word for word in tweet if word not in stop_words]
    # Joining tokens back into a single string
    tweet = ' '.join(tweet)
    return tweet

# Applying the preprocessing function to each tweet
df['processed_text'] = df['text'].apply(preprocess_tweet)

# Creating document-term matrix using CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = cv.fit_transform(df['processed_text'])

# Running LDA model to identify topics in the tweets
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(doc_term_matrix)

# Displaying top words in each topic
for i,topic in enumerate(lda_model.components_):
    print(f'Top words in topic #{i}:')
    print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])
    print('\n')

# Topic modeling using LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA

lda_model = LDA(n_components=num_topics, 
                 learning_method='online', 
                 random_state=42, 
                 n_jobs=-1)

lda_output = lda_model.fit_transform(data_vectorized)

# Print the topics found by the LDA model
print("Top " + str(num_words) + " topics:")
print_topics(lda_model, tfidf_vectorizer.get_feature_names(), num_words)

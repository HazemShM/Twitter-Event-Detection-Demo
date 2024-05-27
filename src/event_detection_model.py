"""# **Importing Libraries:**"""

# For Preprocessing Module
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
from dateutil.parser import parse
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import pandas as pd

# !pip install langdetect
# from langdetect import detect
import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
contraction_dict = {"ain't": "is not","aint":"is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not","didnt": "did not",  "doesn't": "does not", "doesnt": "does not","don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                    "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                    "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "isnt": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
                    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have","today's": "today is","tomorrow's":"tomorrow is", "wasn't": "was not",
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have","&amp;":"and","&lt;":"<","&gt;":">","&le;":"=<","&ge;":">="}
stop_words=set(stopwords.words('english'))


# For Filtering Module
from pyLSHash import LSHash
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# For Clustering
from umap import UMAP
# Using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import itertools
# Using HDBSCAN
from hdbscan import HDBSCAN

# For Sentiment Analysis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# For Related Topics
from transformers import TFAutoModelForSequenceClassification
from scipy.special import expit

# For Event Summarization
from sklearn.feature_extraction.text import CountVectorizer

# More stuff
import subprocess
import json
import os
from collections import Counter



"""# **Preprocessing Module:**"""

def categorize_tweets(df, window_size='4H'):
    if 'H' in window_size:
        time_unit = 'h'
    elif 'M' in window_size:
        time_unit = 'min'
    elif 'S' in window_size:
        time_unit = 's'
    else:
        raise ValueError("Invalid window size format. Use '2H' for hours or '2M' for minutes.")

    time_value = int(window_size[:-1])
    if(time_unit=='h'):
        window_length = pd.Timedelta(hours = time_value)
    elif(time_unit=='min'):
        window_length = pd.Timedelta(minutes = time_value)
    else:
        window_length = pd.Timedelta(seconds = time_value)
    categorized = OrderedDict()
    categorized_indexes = OrderedDict()

    # df['Date'] = pd.to_datetime(df['TweetDate'])
    df['Date'] = pd.to_datetime(df['TweetDate'], format='%a %b %d %H:%M:%S %z %Y')
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(window_length)

    current_time = df['Date'].min().floor(time_unit)
    last_time = df['Date'].max().floor(time_unit)
    # print(current_time)
    # print(last_time)
    df['Group'] = None

    while current_time <= last_time:
        categorized[current_time] = []
        categorized_indexes[current_time] = []
        current_time += window_length

    for index, row in df.iterrows():
        tweet_time = row['Date'].floor(time_unit)
        window_start = tweet_time - (tweet_time - current_time) % window_length
        df.loc[index, 'Group'] = window_start
        categorized[window_start].append(row['Text'])
        categorized_indexes[window_start].append(index)
        # if (index + 1) % 5000 == 0:
        #     print(f"Finished grouping {index + 1} rows")

    return categorized, categorized_indexes

tt = TweetTokenizer()

def removestopwords(line):
    words=line.split(" ")
    wordslist=[]
    if "off" in stop_words:
        stop_words.remove("off")
    while "rt" in words:
        words.remove("rt")
    while "sl" in words:
        words.remove("sl")
    for word in words:
            if not word in stop_words:
                wordslist.append(word)
    return ' '.join(wordslist)

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(tweet):
    nltk_tagged = nltk.pos_tag(tt.tokenize(tweet))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)

    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return ' '.join(res_words)

def removeUnnecessaryWords(x):
    words=x.split(" ")
    result = []
    for word in words:
        if not len(word) < 2:
            result.append(word)

    if len(result)>2:
        return ' '.join(result)
    else:
        return ""
def remove_duplicates(tweet):
    words = tweet.split(" ")
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)
    return ' '.join(unique_words)
def remove_only_hashtags_tweets(tweet):
    temp = re.sub(r'#\w+', '', tweet)
    temp = re.sub(r'@\w+', '', temp)
    temp=re.sub(r' +',' ',temp)
    temp=re.sub(r' +[a-zA-Z] +',' ',temp)
    temp=temp.strip()
    if temp == "":
        return temp
    return tweet
#     words=temp.split(" ")
#     if(len(words) > 1):
#         return tweet
#     else:
#         return ""

def remove_bad_tweets(tweet):
    bad_words =["ãusgsã","fck","stupidity","fuckkkk",
                "yikes","nooooooooooooo","nooooo","noooo","nooooooooo","elf","cantwell",
                "booooooooooooooooooooo","madness","ummm","omg","ummmm","lolz",
                "sexenio","stupid","dumbest","idiot","stupidtweets","dumb","pleasepleaseplease","bahh"
                "cantwell","adak","scum",
                "aniak","bethel","hoonah" ,
                "wtf","gop","dtn","tou","woul","sup","supe","expe",
                "dog","aaaawwwww","aaaawwww","awwwww","awwwwwwwww","awwwwh",
                "crap","crazy","rtâ","ryâ","welpâ","lame",
                "fuck", "fucked","fucking","fuuuucckk","shittttt","sucks","suck",
                "fucking","lmao","lol","dick","ass","nigga","bitch","bitches","shit",
                "shitheads","hell","fuckin"]

    words = tweet.split(" ")
    for word in words:
        if word in bad_words:
            return ""
    return tweet
def preprocess(tweets):
    cleaned_tweets = []
    count = 0
    for tweet in tweets :
        tweet = tweet.lower()
        tweet=replace_contractions(tweet)
        # Remove URLs
        tweet = re.sub(r'(via:)? +https?:\/\/.*','', tweet)
        tweet = re.sub(r'https?://\S+', '', tweet)
        # Remove user mentions
#         tweet = re.sub(r'@\w+', '', tweet)
#         tweet = re.sub(r'#\w+', '', tweet)
        # Remove punctuation except hashtags
#         tweet = re.sub(r'[^\w\s#]', ' ', tweet)
        # Remove punctuation except hashtags , - , @
        tweet = re.sub(r'[^\w\s#@%]', ' ', tweet)
        tweet = tweet.replace('\n', ' ')
#         tweet = re.sub(r'(?<!\d)-|-(?!\d)', ' ', tweet)
#         tweet = re.sub(r'[^\w\s]', ' ', tweet)
        # Remove special characters
#         tweet=re.sub(r'[:-?;&)(!"*%_+$~/\[\]]',' ',tweet)
        # tweet=lemmatize(tweet)
        tweet=removestopwords(tweet)
        tweet=re.sub(r' +',' ',tweet)
        tweet=re.sub(r' +[a-zA-Z] +',' ',tweet)
        tweet =remove_duplicates(tweet)
        tweet =remove_only_hashtags_tweets(tweet)
        tweet =remove_bad_tweets(tweet)
        tweet=removeUnnecessaryWords(tweet)
        tweet=tweet.strip()
        cleaned_tweets.append(tweet)
        count+=1
        # if count % 5000 == 0:
        #     print(f"Preprocessed {count} rows")
    # tokens = tweet.split()
    return cleaned_tweets

def preprocess_dataframe(df) :
    df.drop_duplicates(subset=["TweetText"],inplace=True)
    df.drop(df[df['TweetText'].str.startswith('RT @')].index, inplace=True)
    df.drop(df[df['TweetText'].str.startswith('RT@')].index, inplace=True)
    
    df.replace({'TweetText': {'': np.nan}}, inplace=True)
    df.replace({'TweetDate': {'': np.nan}}, inplace=True)
    df.dropna(subset=['TweetDate', 'TweetText'], inplace=True)
    df['Text'] = preprocess(df['TweetText'])
    # df['Text'].replace('', np.nan, inplace=True)
    df.replace({'Text': {'': np.nan}}, inplace=True)
    df.dropna(subset=['Text'], inplace=True)
    df.drop_duplicates(subset=["Text"],inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Number of Rows After Preprocessing: {len(df)}")
    return df

def compute_tfidf_matrix(tweets):
    vectorizer = TfidfVectorizer(preprocessor=None, tokenizer=None, token_pattern=r'[^ ()]+'  )
    tfidf_matrix = vectorizer.fit_transform(tweets)
    return tfidf_matrix, vectorizer.get_feature_names_out()

"""# **Filtering Module**"""

def cosine_sim(x, y):
    if(x.shape[0]!=y.shape[0]):
        # print("shape unmatched ",y)
        return 0
    else:
        return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
def euclidean_dist_square(x, y):
    if(x.shape[0]!=y.shape[0]):
        # print("shape unmatched ",y)
        diff = np.array(x) - x
    else:
        diff = np.array(x) - y
    return np.dot(diff, diff)

def filtering_module(df,tweets,tweets_indexes,tfidf_matrix , hash_size = 10, num_hashtables = 6 , num_groups = 3):

    # reduced_matrix = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine' ,random_state=42).fit_transform(tfidf_matrix)
    reduced_matrix=tfidf_matrix
    sparse = issparse(reduced_matrix)
    input_dim = reduced_matrix.shape[1]
    # lsh = LSHash(hash_size, input_dim, num_hashtables)
    # lsh.clear_storage()
    lsh = LSHash(hash_size, input_dim, num_hashtables)
    count = 0
    df['Type'] = None
    # Type = np.full(df.shape[0], 'Not Unique', dtype='S9')
    group = 0
    hashedGroups = 0
    collided_tweets_sizes = []
    # similarity_array =[]
    for index, (key, value) in enumerate(tweets_indexes.items()):
        if(index < num_groups):
            if(len(value)==0):
                num_groups = num_groups + 1
            for val in value:
                if(sparse):
                    lsh.index(reduced_matrix[val].toarray()[0], extra_data = val)
                else:
                    lsh.index(reduced_matrix[val], extra_data = val)
                df.loc[val, 'Type'] = 'Not Unique'
                hashedGroups+=1
                count+=1
                # if count % 5000 == 0:
                #     print(f"Hashed {count} rows")
            group+=1
        else:
            for val in value:
                current_tweet_tfidf = tfidf_matrix[val]
                # print(current_tweet_tfidf)
                if(sparse):
                    collision_set = lsh.query(reduced_matrix[val].toarray()[0], num_results=None )
                else:
                    collision_set = lsh.query(reduced_matrix[val], num_results=None )
                count+=1
                collided_tweets_sizes.append(len(collision_set))
                # similarity_values = []
                similarity_value = 0;
                for (collided_tweet, _) in collision_set:
                    collided_tweet_tfidf = tfidf_matrix[collided_tweet[1]]
                    if(sparse):
                        similarity = cosine_similarity(current_tweet_tfidf, collided_tweet_tfidf)[0][0]
                    else:
                        similarity = cosine_similarity(current_tweet_tfidf.reshape(1, -1), collided_tweet_tfidf.reshape(1, -1))[0][0]
                    similarity_value = max(similarity_value , similarity)
                    break
                    # similarity_values.append(similarity)
                # similarity_array.append(similarity_values)
                if similarity_value >= 0.5:
                    df.loc[val, 'Type'] = 'Not Unique'
                    # lsh.index(current_tweet_tfidf.toarray()[0], extra_data = val)
                else:
                    df.loc[val, 'Type'] = 'Unique'

                # if count % 5000 == 0:
                #     print(f"Filtered {count} rows")

            group+=1
    lsh.clear_storage()
    # print('Size of Hashed groups',hashedGroups)
    # print(count)
    # print(group)
    # print(similarity_array)
    return collided_tweets_sizes


def remove_unique_tweets(df,window_size='4H'):
    rows_to_delete = df[df['Type'] == 'Unique'].index
    df.drop(rows_to_delete, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return categorize_tweets(df, window_size)

def get_dictionary_size(dictionary):
    count =0
    for key , val in dictionary.items():
        count+=len(val)
    return count

# sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
# sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

# def preprocess_sentiment(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = '' if t.startswith('#') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

# def sentiment_analysis(text):
#     text = preprocess_sentiment(text)
#     encoded_input = sentiment_tokenizer(text, return_tensors='pt')
#     # print(encoded_input)
#     output = sentiment_model(**encoded_input)
#     # print(output)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     return scores


# def analyze_cluster_sentiments(cluster):
#     cluster_sentiments = []
#     for text in cluster:
#         # text = df.iloc[text_index]["Text"]
#         # print(text)
#         cluster_sentiments.append(sentiment_analysis(text))
#     # print(cluster_sentiments)
#     return np.mean(cluster_sentiments, axis=0)

# def get_sentiment(sentiment):
#     max_index = np.argmax(sentiment)
#     categories = ['Negative', 'Neutral', 'Positive']

# topic_model_name = f"cardiffnlp/tweet-topic-latest-multi"
# topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_name)

# topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_name)
# topic_class_mapping = topic_model.config.id2label

# def tweet_related_topics(tweet):
#     tokens = topic_tokenizer(tweet, return_tensors='pt')
#     output = topic_model(**tokens)
#     scores = output[0][0].detach().numpy()
#     scores = expit(scores)
#     return scores

# def cluster_related_topics(cluster):
#     cluster_scores = []

#     for tweet in cluster:
#         tweet_score= tweet_related_topics(tweet)
#         cluster_scores.append(tweet_score)

#     cluster_final_score = np.mean(cluster_scores, axis=0)
#     cluster_predictions = (cluster_final_score >= 0.5) * 1
#     cluster_topics = []

#     for i in range(len(cluster_predictions)):
#         if cluster_predictions[i]:
#             cluster_topics.append(topic_class_mapping[i])

#     return cluster_topics

# print("\u00e2\u0098\u0085".encode().decode('unicode-escape').encode('latin1').decode())

# print(b"\u00e2\u0098\u0085".decode('unicode-escape').encode('latin1').decode())

"""# **Clustering Module:**"""

# DBSCAN Clustering
def DBSCAN_clustering(combinations , matrix):
    scores = []
    all_labels_list = []
    # reduced_matrix = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine' ,random_state=42).fit_transform(matrix)
    reduced_matrix = matrix
    for i ,(eps , num_samples) in enumerate(combinations):
        dbscan_cluster_model = DBSCAN(eps = eps, min_samples = num_samples).fit(reduced_matrix)
        labels = dbscan_cluster_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        if -1 in labels_set:
            num_clusters -=1
        if(num_clusters < 2):
            scores.append(-10)
            all_labels_list.append('nothing')
            c = (eps, num_samples)
            # print(f"Combination {c} on iteration {i+1} of {len(combinations)} has {num_clusters} clusters. Moving on")
            continue
        scores.append(silhouette_score(reduced_matrix,labels))
        all_labels_list.append(labels)
        # print(f"iteration: {i+1}, Silhouette Score: {scores[-1]}, Number of Clusters: {num_clusters}")
    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]
    return {'best_epsilon':best_parameters[0],
            'best_min_samples' : best_parameters[1],
            'best_labels' : best_labels,
            'best_score' : best_score}

# HDBSCAN Clustering
def HDBSCAN_clustering(min_cluster_size_array, matrix):
    scores = []
    all_labels_list = []
    # reduced_matrix = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine' ,random_state=42).fit_transform(matrix)
    reduced_matrix = matrix
    # print(reduced_matrix.shape)
    for i, min_cluster_size in enumerate(min_cluster_size_array):
        # hdbscan_cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True).fit(reduced_matrix.toarray())
        if issparse(reduced_matrix) :
            hdbscan_cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom').fit(reduced_matrix.toarray())
        else:
            hdbscan_cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom').fit(reduced_matrix)
        labels = hdbscan_cluster_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        if -1 in labels_set:
            num_clusters -=1
        if num_clusters < 2:
            scores.append(-10)
            all_labels_list.append('nothing')
            # print(f"Min cluster size {min_cluster_size} on iteration {i+1} of {len(min_cluster_size_array)} has {num_clusters} clusters. Moving on")
            continue
        scores.append(silhouette_score(reduced_matrix, labels))
        all_labels_list.append(labels)
        # print(f"Iteration: {i+1} , Silhouette Score: {scores[-1]}, Number of Clusters: {num_clusters}")
    best_index = np.argmax(scores)
    best_parameters = min_cluster_size_array[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]
    return {
        'best_min_cluster_size': best_parameters,
        'best_labels': best_labels,
        'best_score': best_score}

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(matrix, minRange = 5 , maxRange = 200 ,step=5):
    silhouette_scores = []
    all_labels_list = []
    num_clusters =[]
    range_clusters = range(minRange, maxRange, step)
    for n_clusters in range_clusters:
        kmeans = KMeans(n_clusters=n_clusters,n_init="auto" , random_state=42)
        cluster_labels = kmeans.fit_predict(matrix)
        silhouette_avg = silhouette_score(matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        all_labels_list.append(cluster_labels)
        num_clusters.append(n_clusters)
    best_labels = all_labels_list[np.argmax(silhouette_scores)]
    best_index = np.argmax(silhouette_scores)
    best_parameters = num_clusters[best_index]
    print("Cluster labels:", best_parameters)
    best_labels = all_labels_list[best_index]
    best_score = silhouette_scores[best_index]

    return {'best_numClusters': best_parameters,
            'best_labels': best_labels,
            'best_score': best_score}

"""# **Event Summarization Module:**"""

def get_cluster_top_keywords( k ,cluster):
    vec = TfidfVectorizer(preprocessor=None, tokenizer=None, token_pattern=r'[^ ()]+')
    matrix = vec.fit_transform(cluster)
    feature_names = np.array(vec.get_feature_names_out())
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    sorted_indices = np.argsort(-frequencies)
    top_keywords = feature_names[sorted_indices]
    return top_keywords.tolist()

def get_representative_headline(k ,cluster):
    keywords = get_cluster_top_keywords( k , cluster )
    cluster_keywords = ' '.join(keywords)
    vec = TfidfVectorizer(preprocessor=None, tokenizer=None, token_pattern=r'[^ ()]+' )
    cluster_tfidf_matrix = vec.fit_transform(cluster)
    cluster_keywords_vector = vec.transform([cluster_keywords])
    similarities = cosine_similarity(cluster_keywords_vector, cluster_tfidf_matrix)
    representative_idx = np.argmax(similarities[0])
    representative_headline = cluster[representative_idx]
    return representative_idx , keywords

def get_representative_tweet(keywords,cluster):
    cluster_text = ' '.join(keywords)
    vec = TfidfVectorizer()
    cluster_tfidf_matrix = vec.fit_transform(cluster)
    cluster_text_vector = vec.transform([cluster_text])
    similarities = cosine_similarity(cluster_text_vector, cluster_tfidf_matrix)
    representative_idx = np.argmax(similarities[0])
    representative_tweet = cluster[representative_idx]
    return representative_tweet

"""# **Events Detection Module:**"""

def get_tweets_entry_rate(clustering_type ,df , tweets_indexes , num_clusters ,window_size = '4H'):
    tweets_entry_rate = OrderedDict()
    clusters_tweets = OrderedDict()
    clusters_tweets_original = OrderedDict()
    clusters_tweets_indexes = OrderedDict()
    for i in range (num_clusters):
        tweets_entry_rate[i] = np.zeros(len(tweets_indexes))
        clusters_tweets[i] = []
        clusters_tweets_original[i] = []
        clusters_tweets_indexes[i] = []

    for i , (key , values) in enumerate(tweets_indexes.items()):
        for value in values:
            cluster = df.iloc[value][clustering_type]
            if(cluster == -1):
                continue
            tweets_entry_rate[cluster][i]+=1
            clusters_tweets[cluster].append(df.iloc[value]["Text"])
            clusters_tweets_original[cluster].append(df.iloc[value]["TweetText"])
            clusters_tweets_indexes[cluster].append(value)

#     window = int(window_size[:-1])
#     for cluster, entry_rates in tweets_entry_rate.items():
#         tweets_entry_rate[cluster] = entry_rates/window

    return tweets_entry_rate ,clusters_tweets ,clusters_tweets_original, clusters_tweets_indexes

def get_clusters_sorted(tweets_entry_rate):
    total_increase = {}
    for cluster, entry_rates in tweets_entry_rate.items():
        increase = sum(max(entry_rates[i] - entry_rates[i-1], 0) for i in range(1, len(entry_rates)))
        total_increase[cluster] = increase
    sorted_clusters = sorted(total_increase.items(), key=lambda x: x[1], reverse=True)
    # print("Clusters ranked by total increase in tweet entry rate:")
    # for rank, (cluster, increase) in enumerate(sorted_clusters, 1):
    #     print(f"Rank {rank}: Cluster {cluster} (Total Increase: {increase})")
    #     if(rank ==15):
    #         break;
    return sorted_clusters


def get_events(sorted_clusters ,clusters_tweets ,clusters_tweets_original ,Top_keywords=6,Number_Events=15):
    for i in range(len(sorted_clusters)):
        if( i>= Number_Events):
            break
        print(f"Event {i+1}: ")
        cluster = clusters_tweets[sorted_clusters[i][0]]
        cluster_original = clusters_tweets_original[sorted_clusters[i][0]]
        tweet , keywords = get_representative_headline(Top_keywords ,cluster)
        sentiment_values = analyze_cluster_sentiments(cluster)
        sentiment_category = get_sentiment(sentiment_values)
        related_topics = cluster_related_topics(cluster)
        print(f"  Top {Top_keywords} Keywords : {keywords}")
        print(f"  Representative Tweet : {cluster_original[tweet]}")
        print(f"  Sentiment : {sentiment_category}")
        print(f"  Related Topics : {related_topics}")
        print()
def remove_duplicates_from_end(arr):
    seen = set()
    result = []
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] not in seen:
            result.append(arr[i])
            seen.add(arr[i])
    result.reverse()
    return result

def get_events_keyword(sorted_clusters ,clusters_tweets ,clusters_tweets_original ,Top_keywords):
    # final_keywords=[]
    # final_tweets = []
    final_events = []
    for i in range(len(Top_keywords)):
        if(i+1>len(sorted_clusters)):
            break
        cluster = clusters_tweets[sorted_clusters[i][0]]
        cluster_original = clusters_tweets_original[sorted_clusters[i][0]]
        keywords = get_cluster_top_keywords(200 ,cluster)
        # filtered_keywords = [word for word in keywords if not word.startswith("#")]
        filtered_keywords = keywords
        # filtered_keywords = [re.sub(r'[^\w\s@-]', '', word) for word in keywords]
        resulted_keywords = filtered_keywords[:Top_keywords[i]]

        # test1 , test2 = get_representative_headline(200 ,cluster)
        # tweet = cluster[test1].split(" ")
#         tweet = get_representative_tweet(cluster).split(" ")
        tweet = get_representative_tweet(resulted_keywords,cluster_original)
#         print(tweet)
#         filtered_tweet = [word for word in tweet if not word.startswith("#")]
#         filtered_tweet = remove_duplicates_from_end(filtered_tweet)
#         filtered_tweet = ' '.join(filtered_tweet)
#         filtered_tweet = preprocess([filtered_tweet])[0].split(" ")
        # final_keywords.append(resulted_keywords)
        # final_tweets.append(tweet)
        final_events.append({
            'keywords': resulted_keywords,
            'tweet': tweet
        })
    return final_events

"""# **Evaluation Module:**"""

def count_words_in_line(line):
    line = re.sub(r'\[[^\]]+\]', '<>', line)
    line = line.replace(';', ' ')
    words = line.split()
    words = [word.replace('<>', '[') for word in words]
    word_count = len(words)
    return word_count
# def read_text_file_and_count_words(filename):
#     word_counts = []
#     with open(filename, 'r',encoding='latin-1') as file:
#         for line in file:
#             line_words = count_words_in_line(line)
#             if line_words > 0:
#                 word_counts.append(line_words)
#     return word_counts

def read_text_file_and_count_words(filename, max_lines=None):
    word_counts = []
    lines_to_keep = []
    with open(filename, 'r', encoding='latin-1') as file:
        for i, line in enumerate(file):
            if max_lines is not None and i >= max_lines:
                break  # Stop reading lines if maximum number of lines reached
            line_words = count_words_in_line(line)
            if line_words > 0:
                word_counts.append(line_words)
            lines_to_keep.append(line)
    # Write back only the first `max_lines` to the file
    with open(filename, 'w', encoding='latin-1') as file:
        file.writelines(lines_to_keep)
    return word_counts

# Detect Events for the Dataset
def detect_events(df, numEvents=5):
    useFiltering = False
    useUMAP = False
    topics_num = numEvents
    number_keywords = 10
    window_size='5S' 
    num_groups = 6 
    # Ensure the DataFrame has 'TweetText' and 'TweetDate' columns
    if 'TweetText' not in df.columns or 'TweetDate' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'TweetText' and 'TweetDate' columns")

    preprocess_dataframe(df)
    categorized_tweets , categorized_tweets_indexes  = categorize_tweets(df,window_size)
    tfidf_matrix , features = compute_tfidf_matrix(df['Text'])

    if useFiltering :
        if useUMAP :
            reduced_matrix = UMAP(n_neighbors=100, n_components=5, min_dist=0.15, metric='cosine' ).fit_transform(tfidf_matrix)
            collided_tweets = filtering_module(df,categorized_tweets,categorized_tweets_indexes,reduced_matrix ,num_groups = num_groups)
        else:
            collided_tweets = filtering_module(df,categorized_tweets,categorized_tweets_indexes,tfidf_matrix ,num_groups = num_groups)
        num_unique_rows = df[df['Type'] == 'Not Unique'].shape[0]
        num_not_unique_rows = df[df['Type'] == 'Unique'].shape[0]

        categorized_tweets , categorized_tweets_indexes = remove_unique_tweets(df ,window_size)
        tfidf_matrix , features = compute_tfidf_matrix(df['Text'])

    if useUMAP :
        reduced_matrix = UMAP(n_neighbors=10, n_components=100, min_dist=0.2, metric='cosine' ).fit_transform(tfidf_matrix)
    else:
        reduced_matrix = tfidf_matrix

    matrix = tfidf_matrix

    if df.shape[0] < 1300 :
        epsilons = [0.9]
        min_samples = [3]
    elif df.shape[0] < 11000:
        epsilons = [0.8]
        min_samples = [6]
    elif df.shape[0] < 15000:
        epsilons = [0.7]
        min_samples = [8]
    else:
        epsilons = [0.6]
        min_samples = [10]
    
    # epsilons = [0.5 ,0.65 ,0.8,0.9]
    # min_samples = [2 ,5 , 7]
    
    # DBSCAN Results
    combinations = list(itertools.product(epsilons,min_samples))
#         cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    result = DBSCAN_clustering(combinations , matrix)
    best_labels = result['best_labels']
    num_clusters = np.amax(list(set(best_labels)))+1
    df['dbscan'] = result['best_labels']
    tweets_entry_rate, clusters_tweets,clusters_tweets_original ,clusters_tweets_indexes = get_tweets_entry_rate("dbscan",df , categorized_tweets_indexes,num_clusters ,window_size)

    sorted_clusters = get_clusters_sorted(tweets_entry_rate)

    DBSCAN_events_to_detect = [number_keywords] * topics_num
    events = get_events_keyword(sorted_clusters, clusters_tweets, clusters_tweets_original, DBSCAN_events_to_detect)

    return events

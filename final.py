import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import torch
import numpy as np
import datetime as dt
from datetime import date
import requests

def bitcoin():
    # Gets todays date
    today = date.today()

    #Searches Twitter for 250 tweets relating to Bitcoin over 10 likes
    tweets = SentimentAnalysis("Bitcoin (Bitcoin OR bitcoin OR BTC OR BITCOIN OR btc) min_faves:10 until:2023-12-30 since:" + str(today))

    # finds the number of positive and negative tweets
    results = transform(tweets)

    # Finds the curret price of bitcoin
    url = 'https://api.coinbase.com/v2/prices/BTC-USD/spot'
    response = requests.get(url)
    data = response.json()
    price = data['data']['amount']

    #Using a linear model calculates the percent change based on the number of positive and negative tweets
    change = (0.00195426 * results[0]) + (-0.00131052*results[1]) - 0.2406282337181418
    while change <= -1 or change >= 1:
        change /= 10
    return round(float(price)*change + float(price), 2)

def eth():
    # Gets todays date
    today = date.today()

    #Searches Twitter for 250 tweets relating to Bitcoin over 10 likes
    tweets = SentimentAnalysis("ethereum (ethereum OR Ethereum) (#ethereum OR #Ethereum) min_faves:8 until:2023-01-21 since:"+ str(today))

    # finds the number of positive and negative tweets
    results = transform(tweets)

    # Finds the curret price of Ethereum
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {
        'symbol': 'ETH',
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '3db968fe-dc05-4a58-8722-90407f6d6fa9'
    }

    response = requests.get(url, headers=headers, params=parameters)

    data = response.json()
    price = data['data']['ETH']['quote']['USD']['price']

    #Using a linear model calculates the percent change based on the number of positive and negative tweets
    change = (0.04408891 * results[0]) + (-0.27945181*results[1]) + -2.0871789465107735
    while change <= -1 or change >= 1:
        change /= 10
    return round(float(price)*change + float(price), 2)

def transform(data_df):
    data_df = data_df.reset_index()
    dates = set()
    for index, row in data_df.iterrows():
        set.add(dates, row["Date"])
    dates = list(dates)
    num_pos = [0] * len(dates)
    num_neg = [0] * len(dates)
    i = 0
    for date in dates:
        for index, row in data_df.iterrows():
            if row["Positive Score"] > row["Negative Score"] and row["Date"] == date:
                num_pos[i] += 1
            if row["Negative Score"] > row["Positive Score"] and row["Date"] == date:
                num_neg[i] += 1
        i += 1
    avgp = sum(num_pos)/len(num_pos)
    avgn = sum(num_neg)/len(num_neg)
    return [avgp,avgn]

def SentimentAnalysis(query: str):
    """
    Takes what ever the query is and finds 1000 tweets relating to that string and uses the Roberta model to create a cs
    v that shows the positve, negative, and nuetral scores

    :param query:
    """
    tweets = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        # Gets the 100 Tweets related to the users query
        if len(tweets) == 250:
            break
        elif tweet.lang == "en":
            tweets.append([tweet.user, tweet.date, tweet.content, tweet.likeCount, tweet.replyCount])
    df = pd.DataFrame(tweets, columns=['User', 'Date', 'Text', 'Likes', 'Replys'])

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Creating three columns for the differnet sentiment scores of the tweet
    positive_score = []
    negative_score = []
    nuetral_score = []

    # sentiment analysis
    for tweet in tweets:
        encoded_tweet = tokenizer(tweet[2], return_tensors='pt')
        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        positive_score.append(scores[2]*100)
        negative_score.append(scores[0]*100)
        nuetral_score.append(scores[1]*100)

    # Adds the colums of postive negative and nutral sentiment scores
    df["Positive Score"] = positive_score
    df["Negative Score"] = negative_score
    df["Nuetral Score"] = nuetral_score

    return df

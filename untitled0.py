# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:18:17 2017

@author: vandit
"""
import tweepy
import csv
import pandas as pd
####input your credentials here

consumer_key = "qiTr6uMacBJ4XOQv939EjJMix"
consumer_secret = "I5DMuK0z5lKFpZkknEDMLq8PSxkt4vFwHwEKdDJ5Xa7qp9hO29"
access_token = "870951075150299137-oWH1sR4yOcdDuWyYi0Y11DlTe5qxvu1"
access_token_secret = "JvWI4hvyiVGEwqI0ZNqcvlcQABYByLY4UJw3e4tvOb5dx"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('de.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

max_items = 10
query = 'Demonetization'#raw_input()
for tweet in tweepy.Cursor(api.search,q=query,count=100,lang="en").items(max_items):
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'), tweet.retweet_count])

data = pd.read_csv("de.csv")
df = pd.DataFrame(data)

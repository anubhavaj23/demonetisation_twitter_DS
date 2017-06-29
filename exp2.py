# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:57:57 2017


@author: anubh
"""
'''

#For static data:
import pandas as pd
import json
file = 'demonetisation-tweets.json'

import codecs
with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as train_file:
    dict_train = json.load(train_file)
data = pd.DataFrame(dict_train)
'''

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

# Open/Create a file to build data
csvFile = open('de.csv', 'wb')
#Use csv Writer
csvWriter = csv.writer(csvFile)

#writer.writeheader()
fieldnames = ["created", "text", "retweetCount"]
writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
writer.writeheader()

max_items = 5000
query = 'Demonetisation'#raw_input()
print 'Loading 5000 tweets on Demonetisation. Please wait...'
for tweet in tweepy.Cursor(api.search, q=query, count=100, lang="en").items(max_items):
    writer.writerow({"created": tweet.created_at, "text": tweet.text.encode('utf-8'), "retweetCount": tweet.retweet_count})
csvFile.close()

#read the data in a dataframe from the stored CSV
data = pd.read_csv("de.csv")

from textblob import TextBlob
import re

#Remove unwanted characters from tweets
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'
    
def get_tweet_sentiment_score(tweet):
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment score
    return analysis.sentiment.polarity

#Create a new dataframe to perform the experiment on it
data_work = pd.DataFrame()
data_work['sentiment'] = [get_tweet_sentiment(tweet) for tweet in data['text']]
temp = data_work[['sentiment']].values
#Convert sentiment dataframe to list to perform analysis
z = temp.ravel()

import matplotlib.pyplot as plt

width = 0.2       # the width of the bars

fig, ax = plt.subplots()
#Separate out the lists for three types of sentiments
a = temp[z == 'positive',0]
b = temp[z == 'neutral',0]
c = temp[z == 'negative',0]
temp = None #get rid of temp

ind = 1
rects1 = ax.bar(ind, len(a), width, color='g')
rects2 = ax.bar(ind + width*2, len(b), width, color='b')
rects3 = ax.bar(ind + width*4, len(c), width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('No. of tweets')
ax.set_title('Sentiment Analysis')
ticks = [ind, ind+width*2, ind+width*4]
ax.set_xticks(ticks)
ax.set_xticklabels(('Positive', 'Neutral', 'Negative'))

def autolabel(rects):
    #Attach a text label above each bar displaying its height
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.legend((rects1[0], rects2[0], rects3[0]), ('Positive', 'Neutral', 'Negative'))
plt.show()

#Printing pos neg and neu tweets
print '\n\nPercentages of each tweets:\nPositive tweets:',len(a)*100./len(data),'%'
print 'Negative tweets:',len(c)*100./len(data),'%'
print 'Neutral tweets:',len(b)*100./len(data),'%\n\n'
a = None
b = None
c = None
        
#Work done to plot score vs time plot for a day                        
import datetime
#dataframe for score vs time plot
#datas = pd.DataFrame()
data_work['score'] = [get_tweet_sentiment_score(tweet) for tweet in data['text']]
data_work['time'] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").time() for t in data['created']]

#Plot sentiment score vs time
plt.plot(data_work['time'],data_work['score'])
plt.title('Time vs Sentiment Score')
plt.ylabel('Sentiment Score')
plt.xlabel('Time')
plt.show()

#Clustering part
print '\n\n'

data_work['sen_score'] = [1 if temp1 == 'positive' else (-1 if temp1 == 'negative' else 0) for temp1 in z]
temp1 = None
values_work = data_work[['time','sen_score']].values
    
plt.scatter(values_work[z == 'positive',0], values_work[z == 'positive',1], s=10, c='green', label='postive')
plt.scatter(values_work[z == 'negative',0], values_work[z == 'negative',1], s=10, c='red', label='Negatve')
plt.scatter(values_work[z == 'neutral',0], values_work[z == 'neutral',1], s=10, c='blue', label='Neutral')
plt.title('Clustering of tweets')
plt.xlabel('Time')
plt.ylabel('Sentiment Score')
plt.legend()
plt.show()


from datetime import datetime, timedelta

#data of one day
data_tsplot = pd.DataFrame()
#calculate data for previous to latest day to calculate 24-hour day
data_tsplot['retweetCount'] = [
                                row['retweetCount'] 
                                for index, row in data.iterrows() 
                                if (
                                        datetime.strptime(row['created'], "%Y-%m-%d %H:%M:%S").date() 
                                        == datetime.now().date() - timedelta(days=1))
                                ]   
data_tsplot['hour'] = [
                        datetime.strptime(row['created'], 
                        "%Y-%m-%d %H:%M:%S").time().hour 
                        for index, row in data.iterrows() 
                        if (
                                datetime.strptime(row['created'], 
                                "%Y-%m-%d %H:%M:%S").date() 
                                == datetime.now().date() - timedelta(days=1))
                      ]


#Calculate retweets in each hour
time_current = int(data_tsplot['hour'][0:1])
s=0
rtsum = []
for i in range(0,len(data_tsplot)): 
    if int(data_tsplot['hour'][i:i+1]) <> time_current:
        rtsum.append(s)
        s=0
        time_current = int(data_tsplot['hour'][i:i+1])
    else:
        s += int(data_tsplot['retweetCount'][i:i+1])
rtsum.append(s)
time_current, s = None, None

#Retweet Per Hour vs Time for a particular day per hour
import numpy as np
import seaborn as sns
print '\n\n'
sns.plt.title('Retweets Per Hour vs Time')
sns.plt.xlabel('Hours')
sns.tsplot(np.array(rtsum),color = 'g', value="Retweets")
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:52:15 2017

@author: anubh
"""

import pandas as pd
import json
file = 'demonetisation-tweets.json'

import codecs
with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as train_file:
    dict_train = json.load(train_file)
data = pd.DataFrame(dict_train)

x = data.iloc[:,[3,4,6,10]].values
x1 = x.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x.astype(int))

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_means = kmeans.fit_predict(x)

import numpy as np
import matplotlib.pyplot as plt


width = 0.2       # the width of the bars

fig, ax = plt.subplots()
a = x1[y_means == 0,3]
b = x1[y_means == 1,3]
c = x1[y_means == 2,3]
#colors = ['r','g','b']
#plt.bar(a,b,c, width, color=colors)
ind = 1
rects1 = ax.bar(ind, len(a), width, color='g')
rects2 = ax.bar(ind + width*2, len(b), width, color='b')
rects3 = ax.bar(ind + width*4, len(c), width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Retweets')
ax.set_title('Sentiment Analysis')
ticks = [ind, ind+width*2, ind+width*4]
ax.set_xticks(ticks)
ax.set_xticklabels(('Positive', 'Neutral', 'Negative'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
ax.legend((rects1[0], rects2[0], rects3[0]), ('Positive', 'Neutral', 'Negative'))
plt.ylim(0,12000)
plt.show()

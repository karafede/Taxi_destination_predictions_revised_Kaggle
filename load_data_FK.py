
import csv
import os
cwd = os.getcwd()
# change working directoy
os.chdir('C:/ENEA_CAS_WORK/Taxi_destination_predictions')
cwd = os.getcwd()

import pickle
import csv
import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# from scipy.interpolate import spline
from scipy import interpolate
from IPython.core.display import display_html
from keras.models import load_model
from utils import np_haversine, density_map, get_clusters, plot_embeddings
from data import load_data
from training import start_new_session, process_features, create_model
from training import full_train

# Display plots inline
# %matplotlib inline
plt.interactive(True)

# check version of a module or package
import tensorflow
import keras
print(tensorflow.__version__)
print(keras.__version__)

# Fix random seed for reproducibility
np.random.seed(42)

data = load_data()

# n_epochs = 100
full_train(n_epochs=1, batch_size=200, save_prefix='mymodel')

print(data.train.shape)
print(data.validation.shape)
print(data.test.shape)

data.train.head(3)

# One key feature is the ORIGIN_STAND column, i.e. the unique identifiers of official stands where customers
# may catch a taxi from. Roughly 48% of taxi rides are in fact started from official taxi stands,
# which is pretty significant:

# Percentage of taxi rides started at taxi stands
100 * pd.notnull(data.train['ORIGIN_STAND']).sum() / float(data.train.shape[0])

# Here is how the ORIGIN_STAND feature is distributed across the entire dataset:
plt.figure(figsize=(7.5,4))
plt.xticks(rotation=90, fontsize=7)
sns.countplot(data.train['ORIGIN_STAND'].dropna().astype(int))
plt.show()
plt.savefig('origin_TAXI_stand.png')

for stand_id in [15, 57]:
    lat, long = data.train[data.train['ORIGIN_STAND'] == stand_id][['START_LAT', 'START_LONG']].mean()
    display_html(
        '<a href="https://www.google.com/maps/?q={lat},{long}" target="_blank">Stand #{stand_id}</a>'.format(
            lat=lat, long=long, stand_id=stand_id), raw=True)


# The time at which each taxi ride started), respectively: week of the year (from 0 to 51), day of the week
# (from 0 to 6), and quarter hour of the day (from 0 to 95).

datetime_index = pd.DatetimeIndex(data.train['TIMESTAMP'])
data.train['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
data.train['DAY_OF_WEEK'] = datetime_index.dayofweek
data.train['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15

# number of taxi trips are distributed across each week of the year

plt.figure(figsize=(7.5,4))
sns.countplot(data.train['WEEK_OF_YEAR'])
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xlabel('Week of the year')
plt.show()
plt.savefig('trip_TAXI_year.png')


# looking at the distribution of taxi trips across each day of the week:

plt.figure(figsize=(7.5,4))
sns.countplot(data.train['DAY_OF_WEEK'])
plt.gca().set_xticklabels(calendar.day_name)
plt.xticks(fontsize=8)
plt.xlabel('Day of the week')
plt.show()
plt.savefig('trip_TAXI_weeks.png')


# .. and across each quarter hour of the day:
plt.figure(figsize=(7.5,4))
sns.countplot(data.train['QUARTER_HOUR'], color='royalblue')
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xticks(fontsize=9)
plt.xlabel('Quarter hour of the day')
plt.show()

# peaks of taxi usage on Fridays and Saturdays (presumably by people going out for entertainment)
# and around 10am and early evening (presumably by people going to and from work).

# durations of trips, the majority of which are around 8 minutes:
plt.figure(figsize=(7.5,4))
bins = np.arange(60, data.train.DURATION.max(), 60)
binned = pd.cut(data.train.DURATION, bins, labels=bins[:-1]/60, include_lowest=True)
sns.countplot(binned, color='royalblue')
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xlim(-1, 40)
plt.xticks(fontsize=9)
plt.xlabel('Duration (in minutes)')
plt.show()
plt.savefig('TAXI_trip_duration.png')


# visual representation of the dataset's spatial distribution (or density
all_coords = np.concatenate(data.train['POLYLINE_FULL'].as_matrix())
density_map(all_coords[:,0], all_coords[:,1])
plt.savefig('TAXI_heatmap_train.png')

all_coords = np.concatenate(data.test['POLYLINE_FULL'].as_matrix())
density_map(all_coords[:,0], all_coords[:,1])
plt.savefig('TAXI_heatmap_test.png')

# model that won the competition is in fact pretty simple. It is a neural network with
# a single hidden layer of 500 neurons and a Rectifier Linear Unit (ReLU) activation function.
# The input layer is comprised of embedding vectors learned for each key feature (quarter hour of the day,
# day of the week, week of the year, the client IDs, the taxi IDs and the stand IDs),
# as well as the first 5 and latest 5 recorded GPS coordinates for each taxi trip

# 1) Before training the model, do a bit of pre-processing by estimating the most popular destination points
# (a few thousands of them) using a mean-shift clustering algorithm.

# 2) Use the softmax activation function as the second-to-last output layer to determine the probabilities
# of the destination being any of the calculated clusters.

# 3) In the last output layer, multiply the probabilities with the clusters' coordinates to obtain the destination
# prediction.

### USE KERAS library

# reduce the spatial dataset by rounding up the coordinates to the 4th decimal (i.e. 11 meters

# use the mean-shift algorithm available in the scikit-learn library to further reduce the number of clusters
# (Note: the quantile parameter was tuned to find a significant and reasonable number of clusters)


# https://github.com/jphalip/ECML-PKDD-2015
# this will do:
# 1) load initial data
# 2) Estimate the GPS clusters (The labels are the last polyline coordinates, i.e. the trips' destinations)
# 3) Create the model with keras (ANN)
# 4) save data
full_train(n_epochs=100, batch_size=200, save_prefix='mymodel')

# Estimate clusters from all destination points
clusters = get_clusters(data.train_labels)
print("Number of estimated clusters: %d" % len(clusters))

plt.figure(figsize=(6,6))
plt.scatter(clusters[:,1], clusters[:,0], c='#cccccc', s=2)
plt.axis('off')
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().autoscale_view('tight')
plt.savefig('TAXI_clusters.png')


# another visualization of the clusters
plt.figure(figsize=(6,6))
plt.scatter(clusters[:,1], clusters[:,0], c='#99cc99', edgecolor='None', alpha=0.7, s=40)
plt.scatter(data.train_labels[:,1], data.train_labels[:,0], c='k', alpha=0.2, s=1)
plt.grid('off')
plt.axis('off')
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().autoscale_view('tight')


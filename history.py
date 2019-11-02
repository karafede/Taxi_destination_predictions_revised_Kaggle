
import os
import pickle
cwd = os.getcwd()
# change working directoy
os.chdir('C:\\ENEA_CAS_WORK\\Taxi_destination_predictions\\cache')
cwd

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
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adagrad
from keras import backend as K
from keras.layers.embeddings import Embedding
# from keras.layers.core import Dense, Reshape, Merge, Activation, Dropout
from keras.layers.core import Dense, Reshape, Activation, Dropout, Flatten
# from keras.layers.core import Dense, Reshape, Activation, Dropout
from keras.layers import Concatenate, Add, concatenate, add, Input
from keras.callbacks import ModelCheckpoint
from utils import tf_haversine
from data import load_data
from utils import get_clusters
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import scipy as sp
from scipy.interpolate import interp1d

'''
# Interpolate a smooth curve from the raw validation loss
n_epochs = len(history['val_loss'])
x_smooth = np.linspace(0, n_epochs-1, num=10)
y_smooth = sp.interpolate.interp1d(range(n_epochs), history['val_loss'], x_smooth)

plt.figure(figsize=(7.5,4))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.plot(x_smooth, y_smooth)
plt.title('Evolution of loss values during training')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(fontsize=9)
plt.axes().xaxis.set_major_locator(MultipleLocator(10))
plt.legend(['train', 'validation', 'smoothened validation'], loc='upper right')
plt.show()
'''

###########################################################################################################
## selected the weights learned at epoch #70 for my final model, which can now be loaded again with Keras:

start_new_session()

# load data and generate clusters
np.random.seed(42)
os.chdir('C:/ENEA_CAS_WORK/Taxi_destination_predictions')
data = load_data()
clusters = get_clusters(data.train_labels)

# load the model of the run #1
os.chdir('C:\\ENEA_CAS_WORK\\Taxi_destination_predictions\\cache')
model = create_model(data.metadata, clusters)
model.load_weights('mymodel-001-2.2026.hdf5')

WWW = model.weights
print(WWW[1].shape)
# Out[139]: TensorShape([7, 10]) ....7 feature of each of the 10 lat, lon (first and last coordinates)

processed = process_features(data.validation)
print(len(processed))
print(processed[6].shape)
# Out[155]: (16444, 20)  # lat, lon

# see the exact MEAN LOSS for our custom validation dataset:
validation_predictions = model.predict(process_features(data.validation))
print(validation_predictions.shape)
# (16444, 2)
len(validation_predictions)
print(validation_predictions)
np_haversine(validation_predictions, data.validation_labels).mean()
# 2.2025578558303396 km (distance between real and predicted destination)

#... and for our custom test dataset:
test_predictions = model.predict(process_features(data.test))
np_haversine(test_predictions, data.test_labels).mean()
# Out[63]: 2.1969294162582598 km

######################################
# Plot trajectories ##################
######################################

# chose the first trajectory of the data.test dataframe
sample_porto_trj = data.test['POLYLINE_FULL'].iloc[2400]
len(sample_porto_trj)
type(sample_porto_trj)

# transform list into and array
sample_porto_trj = np.array(sample_porto_trj)
x, y = sample_porto_trj.T
plt.scatter(x,y)
plt.show()

# chose the first trajectory of the test_predictions dataset
modeled_sample_porto_trj = np.array(test_predictions[2400])
x, y = modeled_sample_porto_trj.T
plt.scatter(x,y)
plt.scatter(x,y)


#################################################################
# import street network (edges) from Porto City as shp file
import os
os.getcwd()
os.chdir('C:/ENEA_CAS_WORK/Taxi_destination_predictions')
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.cm as cm

'''
# filter residential and unclassified roads road
filter = ('["highway"!~"residential|unclassified|living_street|track|abandoned|path|footway|service|pedestrian|road|'
          'raceway|cycleway|steps|construction"]')

# load OSM map from Port with a radious of ~ 30km
Porto = ox.graph_from_address('Porto, Portugal',
                                       distance=30000,
                                       network_type='drive',
                                       custom_filter=filter)

# check what is inside the edges (type of roads)
edge_file = Porto.edges(keys=True,data=True)
ox.plot_graph(Porto)

Porto_shp = ox.gdf_from_place('Porto, Portugal')
ox.save_gdf_shapefile(Porto_shp)

# save street network as GraphML file
ox.save_graphml(Porto, filename='network_Porto_30km_epgs4326.graphml')

# save street network as ESRI shapefile (includes NODES and EDGES)
ox.save_graph_shapefile(Porto, filename='networkPorto_30km__shape')
'''

import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

# load street map
street_map =gpd.read_file("data\\networkPorto_30km__shape\\edges\\edges.shp")
fig , ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax)


# create a dataframe with lat, lon
sample_porto_trj = pd.DataFrame({'lat': sample_porto_trj[:, 0], 'lon': sample_porto_trj[:, 1]})
crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(sample_porto_trj["lon"], sample_porto_trj["lat"])]
print(geometry[:3])
# create a Geo DataFrame
geo_df = gpd.GeoDataFrame(sample_porto_trj,
                          crs = crs,
                          geometry = geometry)


# create an ID field
geo_df['ID']=1
geo_df_line = geo_df.groupby('ID')['geometry'].apply(lambda x: LineString(x.tolist()))
geo_df_line = gpd.GeoDataFrame(geo_df_line,
                               crs = crs,
                               geometry='geometry')


# create a dataframe with lat, lon form the modelled destination coordinate
columns=["lat", "lon"]
modeled_sample_porto_trj = pd.DataFrame(modeled_sample_porto_trj.reshape(-1, len(modeled_sample_porto_trj)),columns=columns)
crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip(modeled_sample_porto_trj["lon"], modeled_sample_porto_trj["lat"])]
print(geometry[:3])
# create a Geo DataFrame
geo_df_modeled = gpd.GeoDataFrame(modeled_sample_porto_trj,
                          crs = crs,
                          geometry = geometry)


# plot OSM map and points (as points!)
fig , ax = plt.subplots(figsize = (15,15))
street_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_df_line.plot(axes=ax)
geo_df_modeled.plot(axes = ax)



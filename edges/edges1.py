import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, LSTM, add
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.initializers import Constant
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os
import matplotlib.pyplot as plt
import warnings
import csv
# from text_unidecode import unidecode
from collections import deque

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.patches as mpatches
# import seaborn as sns
from node2vec import Node2Vec

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder


# load data (figure out how to generate a traffic using flows.py and the connectivity dataset

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


# load data

data_dir = './test-data'
rootX = os.path.join(data_dir, 'x')
pathsX = sorted(listdir_nohidden(rootX))

rootY = os.path.join(data_dir, 'y')
pathsY = sorted(listdir_nohidden(rootY))

rootAdj = os.path.join(data_dir, 'adj')
pathsAdj = sorted(listdir_nohidden(rootAdj))

loadX = False
loadY = False
loadAdj = False

for x in pathsX:
    if not loadX:
        X = np.loadtxt(os.path.join(rootX, x))
        loadX = True
    else:
        X = np.vstack([X, np.loadtxt(os.path.join(rootX, x))])

for y in pathsY:
    if not loadY:
        Y = np.loadtxt(os.path.join(rootY, y))
        loadY = True
    else:
        Y = np.vstack([Y, np.loadtxt(os.path.join(rootY, y))])

for adj in pathsAdj:
    if not loadAdj:
        Adj = np.loadtxt(os.path.join(rootAdj, adj))
        loadAdj = True
    else:
        Adj = np.vstack([Adj, np.loadtxt(os.path.join(rootAdj, adj))])

# normalize between [0, 1]

X /= np.amax(X)
Y /= np.amax(Y)

Adj_Y = np.append(Adj, Y, axis=1)
Adj_X = np.append(Adj, X, axis=1)
# generate a graph

# initialize constants
NUM_FILES = 50
EPOCHS = 50
NUM_HIDDEN = 10
NUM_NODES = 4
INPUT_LEN = 128
BATCH_SIZE = 50

embeddings_Y = np.empty(shape=(NUM_FILES, NUM_HIDDEN, INPUT_LEN))
# embeddings_X = np.empty(shape=(NUM_FILES, NUM_HIDDEN, INPUT_LEN))

# set up a counter for file savings:
iy = 0

# set up a dict for indexing:
ID = {"('0','0')": 0, "('0','1')": 1, "('0','2')": 2, "('0','3')": 3,
      "('1','1')": 4, "('1','2')": 5, "('1','3')": 6,
      "('2','2')": 7, "('2','3')": 8,
      "('3','3')": 9}

# for features
for adj in Adj_Y:
    graph = nx.DiGraph()
    for i, edge in enumerate(adj[:16]):
        if edge == 1.0:
            graph.add_edge(i // NUM_NODES, i % NUM_NODES, weight=adj[16 + i])
            graph.add_edge(i // NUM_NODES, i % NUM_NODES, weight=adj[16 + i])

    node2vec = Node2Vec(graph, walk_length=1, num_walks=1, weight_key='weight')
    # model = node2vec.fit(window=10, min_count=1)
    model = node2vec.fit(min_count=1)

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    edges_kv = edges_embs.as_keyed_vectors()

    # Save embeddings for later use
    edges_kv.save_word2vec_format("yuan-EdgeFeatures" + str(iy) + ".emb")

    with open("yuan-EdgeFeatures" + str(iy) + ".emb", "r") as graph_file:
        next(graph_file)
        reader = csv.reader(graph_file, delimiter=" ")

        embedding_matrix = np.empty((NUM_HIDDEN, INPUT_LEN))
        for row in reader:
            try:
                edge = ID[row[0] + row[1]]
                embedding_matrix[edge] = [float(item) for item in row[2:]]
            except Exception as IndexError:
                print(row)

    embeddings_Y = np.vstack((embeddings_Y, embedding_matrix[None]))
    iy += 1

# set up a counter for file savings:
ix = 0

# for labels:
# add (0,1) and (1,0) = (0,1)
# (0,2) and (2,0) = (0,2)
# (0,3) and (3,0) = (0,3)
# (1,2) and (2,1) = (1,2)
# (1,3) and (3,1) = (1,3)
# (2,3) and (3,2) = (2,3)
labels = []
for x in X:
    new01 = x[1] + x[4]
    new02 = x[2] + x[8]
    new03 = x[3] + x[12]
    new12 = x[6] + x[9]
    new13 = x[7] + x[13]
    new23 = x[11] + x[14]
    newArray = [x[0], new01, new02, new03, x[5], new12, new13, x[10], new23, x[15]]
    labels.append(newArray)

labels = np.array(labels)

# for labels
# for adj in Adj_X:
#     graph = nx.DiGraph()
#     for i, edge in enumerate(adj[:16]):
#         if edge == 1.0:
#             graph.add_edge(i // NUM_NODES, i % NUM_NODES, weight=adj[16 + i])
#
#     node2vec = Node2Vec(graph, walk_length=1, num_walks=1, weight_key='weight')
#     # model = node2vec.fit(window=10, min_count=1)
#     model = node2vec.fit(min_count=1)
#
#     # Save embeddings for later use
#     model.wv.save_word2vec_format("yuan-labels" + str(ix) + ".emb")
#
#     with open("yuan-labels" + str(ix) + ".emb", "r") as graph_file:
#         next(graph_file)
#         reader = csv.reader(graph_file, delimiter=" ")
#
#         embedding_matrix = np.empty((NUM_HIDDEN, INPUT_LEN))
#         for row in reader:
#             node = row[0]
#             embedding_matrix[int(node)] = [float(item) for item in row[1:]]
#
#     embeddings_X = np.vstack((embeddings_X, embedding_matrix[None]))
#     ix += 1

# debug
# print(embeddings_X.shape)
# print(embeddings_Y.shape)


# nx.draw_networkx(graph)
# plt.show()


# visualizations
# nodes = [x for x in model.wv.vocab]
# embeddings = np.array([model.wv[x] for x in nodes])
# tsne = TSNE(n_components=2, random_state=7, perplexity=15)
# embeddings_2d = tsne.fit_transform(embeddings)
#
# figure = plt.figure(figsize=(11, 9))
#
# ax = figure.add_subplot(111)
#
# ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
# plt.show()

features = embeddings_Y[NUM_FILES:, :, :]
# labels = embeddings_X[NUM_FILES:, :, :]

# Per normal, we split the data into training set and test set, the ratio is 0.2
# we want to keep the time steps consistent as the original, so set shuffle=False
# random_state=NUM_HIDDEN2 guarantees that same sequence of random numbers are generated each time we run the code
# X_train, X_test, y_train, y_test = train_test_split(features[0], labels[0], test_size=0.2, shuffle=False,
#                                                     random_state=NUM_HIDDEN2)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False,
                                                    random_state=42)

# try constructing shared embedding layers
layers = []
inputs = []
input_len = 128

sequence_input = Input(shape=(input_len,), dtype='int32')
# sequence_input = Input(shape=(NUM_HIDDEN, 128), dtype='int32')
for i in range(NUM_FILES):
    # initialize the embedding layer using Keras API
    embedding_layer = Embedding(input_dim=NUM_HIDDEN,
                                output_dim=INPUT_LEN,
                                # embeddings_initializer=Constant(features[i]),
                                # weights=[features[i]],
                                # input_length=NUM_HIDDEN,
                                input_length=input_len,
                                trainable=False)

    embedding 
    # sequence_input = Input(shape=(NUM_HIDDEN,), dtype='int32')
    # inputs.append(sequence_input)
    embedded_sequences = embedding_layer(sequence_input)
    layers.append(embedded_sequences)

# added = concatenate(layers)

# add 50 embedding layers
added = add(layers)

# eventually I want the shape to be 10*1
UNITS = 16

# Cuz we have 10 edges
TIME_STEP = 10

# 128 vectors
NUM_FEATURES = 128

# x = Flatten()(added)
# x = Dense(1, activation='relu')(x)
# x = Dense(1, activation='relu')(added)
x = LSTM(UNITS, activation='tanh', input_shape=(TIME_STEP, NUM_FEATURES))(added)
# x = LSTM(UNITS, activation='tanh', input_shape=(TIME_STEP, NUM_FEATURES))(x)

# x = Dense(UNITS, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

ML = Model(sequence_input, preds)
# ML = Model(inputs, preds)
# ML = Sequential()
ML.compile(loss='binary_crossentropy',
           optimizer='sgd',
           metrics=['acc'])

print(ML.summary())

# ML.fit(X_train, y_train, validation_data=(X_test, y_test),
#           epochs=EPOCHS, batch_size=BATCH_SIZE)
# ML.fit(X_train[0], y_train[0], validation_data=(X_test[0], y_test[0]),
#           epochs=EPOCHS, batch_size=BATCH_SIZE)


history = ML.fit(X_train[0], y_train[0], validation_data=(X_test[0], y_test[0]),
                 epochs=EPOCHS, batch_size=BATCH_SIZE)

# history = ML.fit(X_train, y_train, validation_data=(X_test, y_test),
#                  epochs=EPOCHS, batch_size=BATCH_SIZE)

plot_model(ML, to_file='model.png', show_shapes=True)

x_new = X_train[1]
x_pred = ML.predict(x_new)

print(x_pred)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

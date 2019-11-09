# %matplotlib inline
import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, LSTM, add
from keras.models import Model
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

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.patches as mpatches
# import seaborn as sns
from node2vec import Node2Vec


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
EPOCHS = 1
NUM_HIDDEN = 4
INPUT_LEN = 128
BATCH_SIZE = 10

embeddings_Y = np.empty(shape=(NUM_FILES, NUM_HIDDEN, INPUT_LEN))
embeddings_X = np.empty(shape=(NUM_FILES, NUM_HIDDEN, INPUT_LEN))

# set up a counter for file savings:
iy = 0

# for features
for adj in Adj_Y:
    # adj=Adj_Y[0]
    graph = nx.DiGraph()
    for i, edge in enumerate(adj[:16]):
        if edge == 1.0:
            graph.add_edge(i // NUM_HIDDEN, i % NUM_HIDDEN, weight=adj[16 + i])
            graph.add_edge(i // NUM_HIDDEN, i % NUM_HIDDEN, weight=adj[16 + i])

    node2vec = Node2Vec(graph, walk_length=1, num_walks=1, weight_key='weight')
    # model = node2vec.fit(window=10, min_count=1)
    model = node2vec.fit(min_count=1)

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    edges_kv = edges_embs.as_keyed_vectors()

    # Save embeddings for later use
    edges_kv.save_word2vec_format("yuan-EdgeFeatures" + str(iy) + ".emb")

    # Save embeddings for later use
    # model.wv.save_word2vec_format("yuan-features" + str(iy) + ".emb")

    with open("yuan-EdgeFeatures" + str(iy) + ".emb", "r") as graph_file:
        next(graph_file)
        reader = csv.reader(graph_file, delimiter=" ")

        # embedding_matrix = dict()
        embedding_matrix = np.empty((NUM_HIDDEN, INPUT_LEN))
        for row in reader:
            node = row[0]
            embedding_matrix[int(node)] = [float(item) for item in row[1:]]

    embeddings_Y = np.vstack((embeddings_Y, embedding_matrix[None]))
    iy += 1

# set up a counter for file savings:
ix = 0

# for labels
for adj in Adj_X:
    graph = nx.DiGraph()
    for i, edge in enumerate(adj[:16]):
        if edge == 1.0:
            graph.add_edge(i // NUM_HIDDEN, i % NUM_HIDDEN, weight=adj[16 + i])

    node2vec = Node2Vec(graph, walk_length=1, num_walks=1, weight_key='weight')
    # model = node2vec.fit(window=10, min_count=1)
    model = node2vec.fit(min_count=1)

    # Save embeddings for later use
    model.wv.save_word2vec_format("yuan-labels" + str(ix) + ".emb")

    with open("yuan-labels" + str(ix) + ".emb", "r") as graph_file:
        next(graph_file)
        reader = csv.reader(graph_file, delimiter=" ")

        # embedding_matrix = dict()
        embedding_matrix = np.empty((NUM_HIDDEN, INPUT_LEN))
        for row in reader:
            node = row[0]
            embedding_matrix[int(node)] = [float(item) for item in row[1:]]

    embeddings_X = np.vstack((embeddings_X, embedding_matrix[None]))
    ix += 1

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
labels = embeddings_X[NUM_FILES:, :, :]

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
for i in range(NUM_FILES):
    # initialize the embedding layer using Keras API
    embedding_layer = Embedding(input_dim=NUM_HIDDEN,
                                output_dim=INPUT_LEN,
                                weights=[features[i]],
                                input_length=INPUT_LEN,
                                trainable=False)

    sequence_input = Input(shape=(INPUT_LEN,), dtype='int32')
    inputs.append(sequence_input)
    embedded_sequences = embedding_layer(sequence_input)
    layers.append(embedded_sequences)

# combined = concatenate(layers, axis=1)

# add 50 embedding layers
added = add(layers)

OUTPUT_SHAPE = 128


# x = Conv1D(64, NUM_HIDDEN, activation='relu')(combined)
x = Conv1D(OUTPUT_SHAPE, NUM_HIDDEN, activation='relu')(added)
x = MaxPooling1D(NUM_HIDDEN)(x)
x = Conv1D(OUTPUT_SHAPE, NUM_HIDDEN, activation='relu')(x)
x = MaxPooling1D(NUM_HIDDEN)(x)
x = Conv1D(OUTPUT_SHAPE, NUM_HIDDEN, activation='relu')(x)
x = MaxPooling1D(NUM_HIDDEN)(x)  # global max pooling
x = Flatten()(x)
x = Dense(OUTPUT_SHAPE, activation='relu')(x)
preds = Dense(OUTPUT_SHAPE, activation='sigmoid')(x)

model = Model(inputs, preds)

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['acc'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# Per normal, we split the data into training set and test set, the ratio is 0.2
# we want to keep the time steps consistent as the original, so set shuffle=False
# random_state=NUM_HIDDEN2 guarantees that same sequence of random numbers are generated each time we run the code
# X_train, X_test, y_train, y_test = train_test_split(features[0], labels[0], test_size=0.2, shuffle=False,
#                                                     random_state=NUM_HIDDEN2)
# # initialize the embedding layer using Keras API
# embedding_layer = Embedding(input_dim=NUM_HIDDEN,
#                             output_dim=INPUT_LEN,
#                             weights=[features[0]],
#                             input_length=INPUT_LEN,
#                             trainable=False)
#
# sequence_input = Input(shape=(INPUT_LEN,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
#
# # shared_lstm = LSTM(6NUM_HIDDEN)
#
# x = Conv1D(INPUT_LEN, NUM_HIDDEN, activation='relu')(embedded_sequences)
# x = MaxPooling1D(NUM_HIDDEN)(x)
# x = Conv1D(INPUT_LEN, NUM_HIDDEN, activation='relu')(x)
# x = MaxPooling1D(NUM_HIDDEN)(x)
# x = Conv1D(INPUT_LEN, NUM_HIDDEN, activation='relu')(x)
# x = MaxPooling1D(NUM_HIDDEN)(x)  # global max pooling
# x = Flatten()(x)
# x = Dense(INPUT_LEN, activation='relu')(x)
# preds = Dense(INPUT_LEN, activation='sigmoid')(x)
#
# model = Model(sequence_input, preds)
# model.compile(loss='binary_crossentropy',
#               optimizer='sgd',
#               metrics=['acc'])
#
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
#           epochs=50, batch_size=INPUT_LEN)
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


# # continue with the TF approach:
# # features and labels?
#
# # Some hyperparameters
#
# # lets have 50 iterations first
# epochs = 50
#
# # self-explanatory
# batch_size = 1
#
# # shape of place holder when training (<batch size>, feature-size)
# # xplaceholder= tf.placeholder('float', [None, n_features])
#
# # shape of place holder when training (<batch size>,)
# yplaceholder = tf.placeholder('float')
#
#
# # Construct the variables for the softmax
# weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, INPUT_LEN],
#                                           stddev=1.0 / math.sqrt(INPUT_LEN)))
# biases = tf.Variable(tf.zeros([NUM_HIDDEN]))
# hidden_out = tf.matmul(embedding_layer, tf.transpose(weights)) + biases
#
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hidden_out, labels=yplaceholder))
#
#
# # AdamOptimizer has good performance on minimizing the cost
# # so does SGD
# # optimizer = tf.train.AdamOptimizer().minimize(cost)
# learning_rate = 1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
# # Here dive into a true TF session
#     with tf.Session() as sess:
#
#         # initialize all global&local variables so far
#         tf.global_variables_initializer().run()
#         tf.local_variables_initializer().run()
#
#         # define a loop using the epochs variable we set
#         for epoch in range(epochs):
#
#             # in each epoch, the epoch_loss starts from 0
#             epoch_loss = 0
#
#             i = 0
#
#             # define another loop which ends until i hits the threshold we set
#             for i in range(int(len(X_train) / batch_size)):
#                 start = i
#                 end = i + batch_size
#
#                 # assign a batch of features and labels respectively to batch_x and batch_y
#                 batch_x = np.array(X_train[start:end])
#                 batch_y = np.array(y_train[start:end])
#
#                 # MAGIC: here we tell the TF to run the subgraph necessary to compute the optimizer and the cost
#                 # by feeding the values in 'batch_x' and 'batch_y' to placeholders
#                 # the value of the optimizer would be thrown away
#                 # cost would be kept as c
#
#                 _, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
#
#
#                 # aggregates c to epoch_loss
#                 epoch_loss += c
#
#                 # i += batch_size
#
#             # print each epoch_loss in each epoch
#             print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
#

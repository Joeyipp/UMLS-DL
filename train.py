from __future__ import absolute_import, division, print_function, unicode_literals

import os
import gc
import sys
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from collections import Counter
from statistics import mean, mode
from numpy import array, asarray, zeros
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Flatten, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape, Bidirectional, CuDNNLSTM
from tensorflow.keras.optimizers import Adam
tf.enable_eager_execution()

# python3 -c 'import keras; print(keras.__version__)' (Check Keras Version)

# COMMAND LINE USAGE: python3 train.py FEATUREFILE.txt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

def save_checkpoints(epochs):
    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every X-epochs. (period=epochs)
        period=epochs)
    
    return cp_callback, checkpoint_path

# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
# class Attention(Layer):
#     def __init__(self, step_dim,
#                  W_regularizer=None, b_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         self.supports_masking = True
#         self.init = initializers.get('glorot_uniform')

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)

#         self.bias = bias
#         self.step_dim = step_dim
#         self.features_dim = 0
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         assert len(input_shape) == 3

#         self.W = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         self.features_dim = input_shape[-1]

#         if self.bias:
#             self.b = self.add_weight((input_shape[1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#         else:
#             self.b = None

#         self.built = True

#     def compute_mask(self, input, input_mask=None):
#         return None

#     def call(self, x, mask=None):
#         features_dim = self.features_dim
#         step_dim = self.step_dim

#         eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
#                         K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

#         if self.bias:
#             eij += self.b

#         eij = K.tanh(eij)

#         a = K.exp(eij)

#         if mask is not None:
#             a *= K.cast(mask, K.floatx())

#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

#         a = K.expand_dims(a)
#         weighted_input = x * a
#         return K.sum(weighted_input, axis=1)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0],  self.features_dim

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def create_embedding(vocab_size, max_length, EMBEDDING_DIM, t_atoms, word2vecFile=None):
    # Use pre-trained Word2Vec
    if (word2vecFile):
        embeddings_index = {}

        # Open pre-trained Word2Vec and create the embeddings_index
        print("\n-> Loading Word Vectors from {}".format(word2vecFile))
        with open(word2vecFile, 'r') as f:
            header = f.readline()
            line = f.readline()

            while line != "":
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

                line = f.readline()

        print("Done!")

        print("\n-> Generating Non-Trainable Embedding Matrix from {}".format(word2vecFile))
        # Create the embedding_matrix using the embeddings_index
        embedding_matrix = zeros((vocab_size, EMBEDDING_DIM))

        for word, i in t_atoms.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        print("Done!")

        # Return a non-trainable Keras Embedding Layer loaded with weights from the word2vecFile
        return Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_length, trainable=True)

    # Return a trainable Keras Embedding Layer
    print("\n-> Generating Trainable Embedding Matrix")
    print("Done!")
    return Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length)

def create_model(embedding, max_length, EMBEDDING_DIM, LSTM_NUM_HIDDEN=0, NUM_CLASSES=0):
    x = Sequential()

    # Embedding Layer
    x.add(embedding)
    #x.add(Flatten())
    #x.add(GlobalMaxPooling1D())
    #model.add(Reshape((1, max_length, EMBEDDING_DIM)))

    # # First Convolutional Layer
    # model.add(Conv1D(100, kernel_size=3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))

    # model.add(Flatten())

    # LSTM Layer
    # model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(256, return_sequences=False)))
    #model.add(Attention(max_length))

    x.add(Bidirectional(CuDNNLSTM(LSTM_NUM_HIDDEN)))
    x.add(Dropout(0.2))
    shared_model = x

    left_input = Input(shape=(max_length,), dtype='int32')
    right_input = Input(shape=(max_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

    print("\n")
    model.summary()
    shared_model.summary()

    # # First Dense Layer
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.5))

    # # Second Dense Layer
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.5))

    # # Third Dense Layer
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.5))

    # # Forth Hidden Layer
    # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))

    # # Fifth Hidden Layer
    # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.2))

    # # Output Layer (Softmax)
    # model.add(Dense(NUM_CLASSES, activation='softmax'))

    # model.compile(#optimizer=Adam(lr=0.001),
    #               optimizer='adam',
    #               loss='sparse_categorical_crossentropy',   # Accept integet targets (categorial_crossentropy takes one-hot encoding)
    #               metrics=['acc'])

    return model

# def run_model(model, padded_left, padded_right, labels):
def run_model(model, inputFile, valFile, t_atoms, max_length, trunc_type, batch_size=512):
    # Model Callbacks
    callbacks = myCallback()                                # Stop training once training accuracy > X%
    cp_callback, checkpoint_path = save_checkpoints(50)     # Save weights every X epochs
    
    # Train Model
    model.save_weights(checkpoint_path.format(epoch=0))

    training_start_time = time()
    # malstm_trained = model.fit([padded_left, padded_right], labels,
    #                             shuffle=True,
    #                             epochs=100,
    #                             batch_size=128,
    #                             validation_split=0.2,
    #                             verbose=1,
    #                             callbacks=[callbacks, cp_callback])
    
    # To-do: SPLIT, SEPARATE, and ADD VALIDATION DATA
    # SHUFFLE & Randomize ALL Data/ CUIs > SPLIT (80:20) > SEPARATE 
    # Reason for no validation_split because data is reading in line-by-line on the fly,
    # so can't effectively split because NOT ALL DATA are read into memory!
    malstm_trained = model.fit_generator(generator=generate_arrays_from_file(inputFile, t_atoms, max_length, trunc_type), 
                                         steps_per_epoch=math.ceil(28419239/batch_size), # math.ceil(dataset size: 41561928/ batch size)
                                         #use_multiprocessing=True,
                                         #workers=10,
                                         validation_data=generate_arrays_from_file(valFile, t_atoms, max_length, trunc_type),
                                         validation_steps=math.ceil(3157693/batch_size),
                                         shuffle=True,
                                         epochs=20,
                                         verbose=1,
                                         callbacks=[callbacks, cp_callback])

    training_end_time = time()
    print("Total training time: {:.2f}".format(training_end_time - training_start_time))

    # Save the model
    model.save('training/final_model.h5')
    print("\nModel successfully saved to training/final_model.h5")

    # Plot accuracy
    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('history-graph.png')

    print(str(malstm_trained.history['val_acc'][-1])[:6] +
        "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")

    #return model
    return malstm_trained

def generate_arrays_from_file(inputFile, t_atoms, max_length, trunc_type):

    ## Old Code Without Generator ##
    # encoded_left = t_atoms.texts_to_sequences(list(df["Left"].compute()))
    # padded_left = pad_sequences(encoded_left, maxlen=max_length, padding='post')

    # encoded_right = t_atoms.texts_to_sequences(list(df["Right"].compute()))
    # padded_right = pad_sequences(encoded_right, maxlen=max_length, padding='post')

    while True:
        with open(inputFile) as f:
            # The first line is the header line, ignore
            next(f)
            for line in f:
                data = line.strip().split(",")
                left, right, label = data[1], data[2], data[3]

                # create numpy arrays of input data
                # and labels, from each line in the file
                encoded_left = t_atoms.texts_to_sequences([left])
                padded_left = pad_sequences(encoded_left, maxlen=max_length, padding='post', truncating=trunc_type)

                encoded_right = t_atoms.texts_to_sequences([right])
                padded_right = pad_sequences(encoded_right, maxlen=max_length, padding='post', truncating=trunc_type)

                yield [padded_left, padded_right], [int(label)]

def predict_arrays(left, right, t_atoms, max_length):
    
    # create numpy arrays of input data
    # and labels, from each line in the file
    encoded_left = t_atoms.texts_to_sequences([left])
    padded_left = pad_sequences(encoded_left, maxlen=max_length, padding='post')

    encoded_right = t_atoms.texts_to_sequences([right])
    padded_right = pad_sequences(encoded_right, maxlen=max_length, padding='post')

    return [padded_left, padded_right]

def main():
    inputFile = sys.argv[1]     # MRCONSO_Train.txt
    valFile = sys.argv[2]       # MRCONSO_Validate.txt
    inputFile2 = sys.argv[3]    # List_of_Unique_Atoms.txt
    inputFile3 = sys.argv[4]    # bio_embedding_intrinsic.txt

    print("\n-> Reading Input File")
    
    #df = dd.read_csv(inputFile)
    #df = pd.DataFrame()
    #for chunk in pd.read_csv(inputFile, header=0, chunksize=1000):
    #    df = pd.concat([df, chunk], ignore_index=True)
    #print("Done!")

    # print("\n-> Checking Assertions")
    # assert df['Left'].shape == df['Right'].shape
    # assert len(df['Left']) == len(df['Label'])
    # print("Assertion OK!")

    print("\n-> Generating Training Data")
    df2 = dd.read_csv(inputFile2, dtype={"Atom": str})
    # df2 = pd.DataFrame()
    # for chunk in pd.read_csv(inputFile2, header=0, chunksize=1000):
    #     df2 = pd.concat([df2, chunk], ignore_index=True)

    all_unique_atoms = list(df2["Atom"].compute().astype(str))

    atoms_file_exists = os.path.isfile(os.path.join(os.getcwd(), 'atoms.pickle'))
    if (atoms_file_exists):
        print("---> Using existing atoms.pickle")
        with open('atoms.pickle', 'rb') as handle:
            t_atoms = pickle.load(handle)
    else:
        # Generate TRAINING DATA
        t_atoms = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n', oov_token="<OOV>")
        t_atoms.fit_on_texts(all_unique_atoms)

        # Saving
        print("---> Pickling Atoms into atoms.pickle")
        with open('atoms.pickle', 'wb') as handle:
            pickle.dump(t_atoms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!\n")

    # Preprocess TRAINING DATA (Padding)
    length = [len(s.split()) for s in all_unique_atoms]
    
    # Pad Sequences
    max_length = max(length)
    print("Max Length: {}".format(max_length))

    print("Mode Length: {}".format(mode(length)))

    mean_length = mean(length)
    print("Mean Length: {:.2f}".format(mean_length))

    vocab_size = len(t_atoms.word_index) + 1          # Define Vocabulary Size
    print("Vocab Size: {}".format(vocab_size))
    
    max_length = 30
    print("Using Max Length: {}".format(max_length))

    # cnt = Counter()
    # for l in length:
    #     cnt[l] += 1

    # sorted_cnt = sorted(cnt, key=cnt.__getitem__, reverse=True)

    # for i in sorted_cnt:
    #     print(i, cnt[i])
    
    # plt.bar(cnt.keys(), cnt.values())
    # plt.savefig('bar.png')

    ## RUN FIT GENERATOR FOR THIS PART ##
    print("\n-> Training Model")
    
    # Embedding Size
    EMBEDDING_DIM = 200
    LSTM_NUM_HIDDEN = 30
    trunc_type='post'

    # Create Word Embedding
    embedding = create_embedding(vocab_size, max_length, EMBEDDING_DIM, t_atoms, inputFile3)                           
    model = create_model(embedding=embedding, max_length=max_length, EMBEDDING_DIM=EMBEDDING_DIM, LSTM_NUM_HIDDEN=LSTM_NUM_HIDDEN)  # Create Model
    model = run_model(model, inputFile, valFile, t_atoms, max_length, trunc_type)
    #model = run_model(model, padded_left, padded_right, list(df["Label"].compute()))                                                # Run Model

    model = tf.keras.models.load_model('./training/final_model.h5', custom_objects={'ManDist': ManDist})

    # with open(valFile) as f:
    #         # The first line is the header line, ignore
    #         next(f)
    #         for line in f:
    #             data = line.strip().split(",")
    #             left, right, label = data[1], data[2], data[3]
    #             prediction = model.predict(predict_arrays(left, right, t_atoms, max_length))
    #             print("{}, {}".format(right, prediction))

    # #### CLASSIFICATION STEPS ####
    # training_data = []
    # cui = []
    # string = []

    # # The string/atom position in the feature file after split. 
    # # The default is 3, based off MRCONSO_Parsed.txt
    # STR_POS = 3

    # # 9730407
    # with tqdm(total=121255) as pbar:
    # # Read the input file and extract the atom strings
    #     with open(inputFile, 'r') as f:
    #         header = f.readline()
    #         line = f.readline()
    #         pbar.update(2)

    #         while line != "":
    #             data = line.strip().split(',')
    #             training_data.append("{},{}".format(data[0], data[STR_POS]))

    #             line = f.readline()
    #             pbar.update(1)

    # # Remove duplicate strings/ training data
    # unique_training_data = set(training_data)

    # for training_dat in unique_training_data:
    #     data = training_dat.split(',')
    #     cui.append(data[0])     # Extract the CUI
    #     string.append(data[1])  # Extract the strings/ atom

    # print("Done!")

    # print("\n-> Generating Training Data")

    # atoms_file_exists = os.path.isfile(os.path.join(os.getcwd(), 'atoms.pickle'))
    # if (atoms_file_exists):
    #     print("---> Using existing atoms.pickle")
    #     with open('atoms.pickle', 'rb') as handle:
    #         t_atoms = pickle.load(handle)
    # else:
    #     # Generate TRAINING DATA
    #     t_atoms = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
    #     t_atoms.fit_on_texts(string)

    #     # Saving
    #     print("---> Pickling Atoms into atoms.pickle")
    #     with open('atoms.pickle', 'wb') as handle:
    #         pickle.dump(t_atoms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # cui_file_exists = os.path.isfile(os.path.join(os.getcwd(), 'cui.pickle'))
    # if (cui_file_exists):
    #     print("---> Using existing cui.pickle")
    #     with open('cui.pickle', 'rb') as handle:
    #         t_cui = pickle.load(handle)
    # else:
    #     # Generate OUTPUT LABELS
    #     t_cui = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
    #     t_cui.fit_on_texts(cui)

    #     # Saving
    #     print("---> Pickling CUI into cui.pickle")
    #     with open('cui.pickle', 'wb') as handle:
    #         pickle.dump(t_cui, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print("Done!\n")

    # print("Total Instances: {}".format(len(training_data)))
    # print("Total Uniq. Instances: {}\n".format(len(unique_training_data)))

    # # Preprocess TRAINING DATA (Padding)
    # max_length = max([len(s.split()) for s in string]) # Pad Sequences
    # print("Max Length: {}".format(max_length))

    # vocab_size = len(t_atoms.word_index) + 1          # Define Vocabulary Size
    # print("Vocab Size: {}".format(vocab_size))

    # encoded_atoms = t_atoms.texts_to_sequences(string)
    # padded_atoms = pad_sequences(encoded_atoms, maxlen=max_length, padding='post')

    # # Number of Softmax Output Classes
    # NUM_CLASSES = len(t_cui.word_index) + 1
    # print("Class Size: {}".format(NUM_CLASSES - 1))

    # encoded_cui = t_cui.texts_to_sequences(cui)
    # labels = [label[0] for label in encoded_cui]
    # # labels = to_categorical([label[0] for label in encoded_cui], num_classes=NUM_CLASSES)

    # print("\n-> Training Model")
    
    # # Embedding Size
    # EMBEDDING_DIM = 200

    # # Create Word Embedding
    # embedding = create_embedding(vocab_size, max_length, EMBEDDING_DIM, t_atoms, 'bio_embedding_intrinsic.txt')                                
    # model = create_model(embedding, max_length, EMBEDDING_DIM, NUM_CLASSES)     # Create Model
    # model = run_model(model, padded_atoms, labels)                              # Run Model
    
    # Evaluate Model
    # loss, accuracy = model.evaluate(padded_atoms, labels, verbose=1)

    # Predict the class
    # predict_class = model.predict_classes(padded_atoms[0].reshape(1,24))
    # print(predict_class)

    # Predict the softmax classes
    # predict = model.predict(padded_atoms[0].reshape(1,24))
    # print(predict)

    # # Access the embedding layer through the constructed model 
    # # First `0` refers to the position of embedding layer in the `model`
    # embeddings = model.layers[0].get_weights()[0]

    # # `embeddings` has a shape of (num_vocab, embedding_dim) 

    # # `word_to_index` is a mapping (i.e. dict) from words to their index, e.g. `love`: 69
    # words_embeddings = {w:embeddings[idx] for w, idx in t_atoms.word_index.items()}

    # # now you can use it like this for example
    # # print(words_embeddings['love']) # possible output: [0.21, 0.56, ..., 0.65, 0.10].
    # print(words_embeddings)  

    # Load latest checkpoint
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # latest
    # model = create_model()
    # model.load_weights(latest)

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    print("\n-> All Done!")

if __name__ == '__main__':
    main()
# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dataset(sentences, labels):
    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    return sentences, labels


def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    # urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    sentences, labels = load_dataset(sentences, labels)

    # Split the sentences
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    # Split the labels
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Initialize the Tokenizer class
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Generate and pad the training sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Generate and pad the testing sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert the labels lists into numpy arrays
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    # Build the model
    lstm1 = 64
    lstm2 = 32
    dense = 15
    model_cat4 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(dense, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Print the model summary
    model_cat4.summary()

    # Compile the model
    model_cat4.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                     metrics=['accuracy'])
    num_epochs = 30

    # Train the model
    history = model_cat4.fit(training_padded, training_labels, epochs=num_epochs,
                           validation_data=(testing_padded, testing_labels), verbose=2)

    # # Plot utility
    # def plot_graphs(history, string):
    #     plt.plot(history.history[string])
    #     plt.plot(history.history['val_' + string])
    #     plt.xlabel("Epochs")
    #     plt.ylabel(string)
    #     plt.legend([string, 'val_' + string])
    #     plt.show()
    #
    # # Plot the accuracy and loss
    # plot_graphs(history, "accuracy")
    # plot_graphs(history, "loss")
    #
    return model_cat4


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
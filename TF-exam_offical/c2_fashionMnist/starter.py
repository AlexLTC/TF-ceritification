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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
import tensorflow_datasets as tfds
import math

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist


    # YOUR CODE HERE
    dataset, metadata = tfds.load("fashion_mnist", as_supervised=True, with_info=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(60000).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=math.ceil(60000 / BATCH_SIZE)
    )

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

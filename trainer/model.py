from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
#from tensorflow.python.keras import models
#import trainer.task as task

tf.logging.set_verbosity(tf.logging.INFO)

def keras_estimator(features, labels, mode):
    #if labels is None:
     #   inputs = features
    #else:
    # Change numpy array shape.
     #   inputs = (features, labels)
    inputs = features
  # Convert the inputs to a Dataset.
    #dataset = tf.data.Dataset.from_tensor_slices(inputs)
    #if mode == tf.estimator.ModeKeys.TRAIN:
    #    dataset = dataset.shuffle(1000).repeat().batch(100)
    #if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
     #   dataset = dataset.batch(100)
    #inputs = dataset.make_one_shot_iterator().get_next()
    #Convolutional layer, takes in the input layer
    #inputs = tf.cast(tf.reshape(features["x"], [-1, 28, 28, 1]), tf.float32)
    #inputs = np.reshape(np.array(features["x"]), [-1, 28, 28, 1])
    #X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #inputs = np.array(features["x"]).reshape(-1,28,28,1)
    inputs = tf.reshape(features["x"], [-1,28,28,1])
    #y = tf.placeholder(tf.float32, [None, 10])
    #inputs = features
    #function conv2d that performs our convolution takes in the following inputs:
    #inputs: the first convolutional layer takes in the actual input, the following 
    #convolutional layers take in the output of the preceding layer which is normally the pooling layer 
    #filters: 
    #kernel_size:
    #padding: 'same' means the output shall be padded so that it remains same size as input
    #activation_size
    convolutional_layer1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
    #Pooling performs subsampling, inorder to reduce dimensionality with a choice of using either max subsampling or average subsampling
    #while we slide over the regions we either pick the max or average of the region depending on our method of method
    #pool size:
    #strides: This determines 
    #Pooling layer, takes in input from the first convolutional layer
    pooling_layer1 = tf.layers.max_pooling2d(inputs=convolutional_layer1, pool_size=[2,2], strides=2)

    #Convolutional layer
    convolutional_layer2 = tf.layers.conv2d(inputs=pooling_layer1, filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
    
    #Pooling layer
    pooling_layer2 = tf.layers.max_pooling2d(inputs=convolutional_layer2, pool_size=[2,2], strides=2)
    
    #Dense layer
    #print(pooling_layer2.size)
    #flatten = np.reshape(np.array(pooling_layer2), [-1, 7 * 7 * 64])
    #flatten = np.array(pooling_layer2).reshape(-1, 7*7*64)
    flatten = tf.reshape(pooling_layer2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
    
    #Dropouts
    dropout_layer = tf.layers.dropout(inputs=dense, rate=0.4, training=mode==tf.estimator.ModeKeys.TRAIN)
    logits_layer = tf.layers.dense(inputs=dropout_layer, units=10)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits_layer, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits_layer, name="softmax_tensor")
  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_layer)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Expects already tokenized and padded representation of sentences

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, 784])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)
#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
        Evaluates output of a NN
    """
    _ = tf.Variable(initial_value='fake_variable')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        sess.run(tf.variables_initializer(all_variables))
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collecction("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        y_ = sess.run(y_pred, {x: X, y: Y})
        ac = sess.run(accuracy, {x: X, y: Y})
        loss = sess.run(loss, {x: X, y: Y})
        return y_, ac, loss

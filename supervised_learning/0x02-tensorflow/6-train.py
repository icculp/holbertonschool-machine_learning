#!/usr/bin/env python3
"""
    Tensorflow project
"""
import tensorflow as tf


def train(X_train, Y_train, X_valid,
          Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
        Builds, trains, and saves a neural network classifier
    """
    create_placeholders = __import__('0-create_plac\
                                     eholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train = create_train_op(loss, alpha)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    with sess.as_default():
        for i in range(iterations):
            print(i)
            tcost = loss.eval(feed_dict={x: X_train, y: Y_train})
            tacc = accuracy.eval(feed_dict={x: X_train, y: Y_train})
            vcost = loss.eval(feed_dict={x: X_valid, y: Y_valid})
            vacc = accuracy.eval(feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(tcost))
                print("\tTraining Accuracy: {}".format(tacc))
                print("\tValidation Cost: {}".format(vcost))
                print("\tValidation Accuracy: {}".format(vacc))
            '''train = trainy.eval(feed_dict={x: X_train, y: Y_train})'''
            sess.run(train, {x: X_train, y: Y_train})
            print(i)
        saver = tf.train.Saver()
        return saver.save(sess, save_path)

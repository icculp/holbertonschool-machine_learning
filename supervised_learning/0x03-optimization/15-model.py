#!/usr/bin/env python3
"""
    Optimization project
"""
import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ Builds, trains, and saves a NN model in tf
        using Adam optimzer, mini-batch gradient descent,
        learning rate decay, and batch normalization

        Data_train is tuple containing training input and labels
        Data_valid is tuple containing validation input and labels
        layers is list of nodes in each layer
        activation is list of activatin functions for each layer
        alpha is learning rate
        beta1 is weight for first moment
        beta2 is weight for second moment
        epsilon to avoid div/0
        decary_rate is rate for inverse time decay, step is 1
        batch_size is number of data points in mini batch
        epochs
        Returns: path where saved
    """

    def create_layer(prev, n, activation):
        """ Returns tensor output of the layer """
        w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        layer = tf.layers.Dense(n, activation=activation,
                                kernel_initializer=w, name='layer')
        return layer(prev)

    def create_batch_norm_layer(prev, n, activation):
        """ Creates a batch normalization layer
            prev is activated output of previous layer
            n is number of nodes
            activation is activation function to be used
            kernal initializer
            tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
            gamma and beta initialized as vectors of 1, 0 respectively
            epsilon of 1e-8
            Returns: normlized Z matrix
        """
        k = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        base = tf.layers.Dense(n, kernel_initializer=k)
        layer = base(prev)
        mean, var = tf.nn.moments(layer, axes=[0])
        gamma = tf.Variable(tf.ones([n]), trainable=True)
        beta = tf.Variable(tf.zeros([n]), trainable=True)
        epsilon = 1e-8
        batch_norm = tf.nn.batch_normalization(layer, mean, var,
                                               beta, gamma, epsilon)
        return activation(batch_norm)

    def forward_prop(x, layer_sizes=[], activation=[]):
        """ creates forward propagation computation graph for NN """
        inp = x
        for i in range(len(layer_sizes)):
            if activation[i] is None:
                inp = create_layer(inp, layer_sizes[i], activation[i])
            else:
                inp = create_batch_norm_layer(inp, layer_sizes[i],
                                              activation[i])
        return inp

    def calculate_accuracy(y, y_pred):
        """ Calculates accuracy (cost) of a prediction """
        m = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        return tf.reduce_mean(tf.cast(m, tf.float32), name="Mean")

    def calculate_loss(y, y_pred):
        """ Calculates softmax cross-entropy loss of a prediction
            y is placeholder for labels of input data
            y_pred is tensor containing networks predictions
        """
        m = tf.losses.softmax_cross_entropy(
            y,
            y_pred)
        return m

    def learning_rate_decay(alpha, decay_rate,
                            global_step, decay_step):
        """ creates learning rate decay operation in tensorflow
            using inverse time decay
            learning_rate_decay(alpha_init, 1, i, 10)
            alpha is the original learning rate
            decay_rate is the weight dermining alpha decay
            global_step is number of gradient descent passes elapsed
            decay_step is number of passes to occur before alpha decayed again
            Returns: updated value for alpha
        """
        opt = tf.train.inverse_time_decay(alpha, global_step,
                                          decay_step, decay_rate,
                                          staircase=True)
        return opt

    def shuffle_data(X, Y):
        """ Shuffles data points in two matrices the same way
            X is ndarray with shape (m nx)
                m number of data points
                nx number of features
            Y is ndarray with shape (m nx)
                m number of data points
                nx number of features
            Returns: shuffled X, Y matrices
        """
        permute = np.random.permutation(X.shape[0])
        return X[permute], Y[permute]

    def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
        """ Creates Adam optimizer training operation in tf
            (loss, 0.001, 0.9, 0.99, 1e-8)
            loss is the loss of the NN
            alpha is the learning rate
            beta1 is first moment weight
            beta2 is second moment weight
            epsilon is small number to avoid division by zero
            Returns: Adam optimiization operation
        """
        opt = tf.train.AdamOptimizer(alpha, beta1=beta1,
                                     beta2=beta2, epsilon=epsilon)
        return opt.minimize(loss)

    with tf.Session() as sess:
        # splt data tuples into x, y
        X_train = Data_train[0]
        Y_train = Data_train[1]
        X_valid = Data_valid[0]
        Y_valid = Data_valid[1]
        # create placeholders for tensorflow
        x = tf.placeholder("float", shape=(None, X_train.shape[1]))
        y = tf.placeholder("float", shape=(None, Y_train.shape[1]))
        tf.add_to_collection('x', x)
        tf.add_to_collection('y', y)
        # m is examples I think?
        m = X_train.shape[0]
        # setting batch size and manually implementing math.ciel
        batches = m / batch_size
        if batches % 1 != 0:
            batches = int(batches) + 1
        else:
            batches = int(batches)

        # forward prop
        ''' if activations none, modify fp '''
        y_pred = forward_prop(x, layers, activations)
        tf.add_to_collection('y_pred', y_pred)

        accuracy = calculate_accuracy(y, y_pred)
        tf.add_to_collection('accuracy', accuracy)

        loss = calculate_loss(y, y_pred)
        tf.add_to_collection('loss', loss)

        ''' decay learning rate alpha '''
        global_step = tf.Variable(0)
        decay = learning_rate_decay(alpha, decay_rate, global_step, 1)
        train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)
        tf.add_to_collection('train_op', train_op)

        """ Initialize variables and run session """
        init = tf.global_variables_initializer()
        sess.run(init)

        '''trainop pass into ? not alpha, but return of decay rate function'''
        for i in range(epochs + 1):
            train_cost = loss.eval({x: X_train,
                                    y: Y_train})
            train_accuracy = accuracy.eval({x: X_train,
                                            y: Y_train})
            valid_cost = loss.eval({x: X_valid,
                                    y: Y_valid})
            valid_accuracy = accuracy.eval({x: X_valid,
                                            y: Y_valid})
            print("After {} epochs:\n".format(i) +
                  "\tTraining Cost: {}\n".format(train_cost) +
                  "\tTraining Accuracy: {}\n".format(train_accuracy) +
                  "\tValidation Cost: {}\n".format(valid_cost) +
                  "\tValidation Accuracy: {}".format(valid_accuracy))
            if i == epochs:
                ''' Done training, last epoch metrics printed '''
                break
            shuf_x, shuf_y = shuffle_data(X_train, Y_train)
            for j in range(batches):
                start = batch_size * j
                end = batch_size * (j + 1)
                '''if end > m:
                    end = None'''
                '''print("start", start, "end", end)'''
                sess.run(train_op, feed_dict={x: shuf_x[start:end],
                                              y: shuf_y[start:end]})
                if (j + 1) % 100 == 0 and j != 0:
                    step_cost = loss.eval({x: shuf_x[start:end],
                                           y: shuf_y[start:end]})
                    step_accuracy = accuracy.eval({x: shuf_x[start:end],
                                                   y: shuf_y[start:end]})
                    print("\tStep {}:\n".format(j + 1) +
                          "\t\tCost: {}\n".format(step_cost) +
                          "\t\tAccuracy: {}".format(step_accuracy))
            sess.run(tf.assign(global_step, global_step + 1))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)

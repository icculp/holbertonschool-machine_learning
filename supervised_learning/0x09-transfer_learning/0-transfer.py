#!/usr/bin/env python3
"""
    Transfer Learning
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ Preprocesses data for model
        X ndarray (m, 32, 32, 3) ontaining CIFAR 10 data
        Y ndarray (m, containing CIFAR 10 labels
        Returns X_p, Y_p
            X_p ndarray containing preprocessed X
            Y_p ndarray containing preprocessed X
        model must exceed 87% validation accuracy
        Save trained model as ./cifar10.h5
        
    """
    #X_p = X.reshape(X.shape[0], 229, 229, X.shape[3])
    #factor = 299 /
    resize = K.layers.Lambda(lambda X: K.backend.resize_images(X, 9 , 9, 'channels_last'), input_shape=(32, 32, 3))
    flat = K.layers.Flatten()
    base = K.applications.inception_v3.InceptionV3(input_shape=(288,288,3), weights='imagenet', include_top=False)#(include_top=True, weights='imagenet')#, input_shape=(32,32,3))#, input_shape=X.shape)
    model = K.Sequential()
    model.add(resize)
    model.add(base)
    model.add(flat)
    model.add(K.layers.Dense(10, activation="softmax"))
    #model.add(resize)

    #X_p = X / 255
    X_p = X# / 255 #K.applications.inception_v3.preprocess_data(X)
    Y_p = K.utils.to_categorical(Y)
    #X_p = K.applications.resnet50.preprocess_input(X)
    model.summary()
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_p, Y_p)#X_p, steps_per_epoch=1)#X_p, steps_per_epoch=1, epochs=1)
    model.save('cifar10.h5')
    return X_p, Y_p
%tensorflow_version 1.x
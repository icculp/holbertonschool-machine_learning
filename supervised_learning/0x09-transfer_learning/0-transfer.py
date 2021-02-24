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
    X_p = K.applications.inception_resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p


if __name__ == '__main__':
    import tensorflow as tf
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    inception = K.applications.DenseNet121(weights='imagenet',
                                           include_top=False)
    inputs = K.Input(shape=(32, 32, 3))
    resize = K.layers.Lambda(lambda X:
                             tf.image.resize_images(X, (299, 299)))(inputs)
    base = inception(resize, training=False)
    layer = K.layers.GlobalAveragePooling2D()(base)
    layer = K.layers.Dense(500, activation="relu")(layer)
    layer = K.layers.Dropout(.3)(layer)
    output = K.layers.Dense(10, activation="softmax")(layer)
    model = K.Model(inputs=inputs, outputs=output)
    inception.trainable = False
    model.summary()

    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(X_train_p, Y_train_p, epochs=4, batch_size=32,
              shuffle=True, verbose=1, validation_data=(X_test_p, Y_test_p))
    model.save('cifar10.h5')

    """
    '''resize = K.layers.Lambda(lambda X:
    K.backend.resize_images(X, 9 , 9,
    'channels_last'), input_shape=(32, 32, 3))'''
    resize = K.layers.Lambda(lambda X:
                             tf.image.resize_images(X, (299, 299)),
                             input_shape=(32, 32, 3))
    '''tf.image.resize(image, (299, 299))
    #flat = K.layers.Flatten()'''
    base = K.applications.InceptionResNetV2(input_shape=(299, 299, 3),
                                            weights='imagenet',
                                            include_top=False)
    '''#(include_top=True, weights='imagenet')#,
    input_shape=(32,32,3))#, input_shape=X.shape)'''
    for layer in base.layers:
        layer.trainable = False
    model = K.Sequential()
    model.add(resize)
    model.add(base)
    '''#print(len(base.layers))
    #model.add(flat)'''
    model.add(K.layers.GlobalAveragePooling2D())
    '''#model.add(K.layers.BatchNormalization())'''
    model.add(K.layers.Dense(512, activation="relu"))
    model.add(K.layers.Dropout(.3))

    '''model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation="relu"))
    model.add(K.layers.Dropout(.2))

    #model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation="relu"))
    model.add(K.layers.Dropout(.2))'''

    '''#model.add(K.layers.BatchNormalization())'''
    model.add(K.layers.Dense(10, activation="softmax"))

    '''#check = K.callbacks.ModelCheckpoint('cifar102.h5',
    save_best_only=True, monitor='val_loss')'''

    model.summary()
    adam = K.optimizers.Adam()
    '''#rms = K.optimizers.RMSprop()'''
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(X_train_p, Y_train_p, epochs=10,
              batch_size=32, shuffle=True,
              validation_data=(X_test_p, Y_test_p))
    model.save('cifar10.h5')
    """

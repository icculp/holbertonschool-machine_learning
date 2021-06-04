#!/usr/bin/env python3
"""
    Transformer Applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ creates and trains a transformer model for machine translation of
            Portuguese to English using our previously created dataset
        N the number of blocks in the encoder and decoder
        dm the dimensionality of the model
        h the number of heads
        hidden the number of hidden units in the fully connected layers
        max_len the maximum number of tokens per sequence
        batch_size the batch size for training
        epochs the number of epochs to train for
        trained with Adam optimization with
            beta_1=0.9, beta_2=0.98, epsilon=1e-9
        learning rate should scheduled using the following equation:
            lrate = (d ** 0.5) * min(step_num ** -.5,
                                     step_num * warmup_steps ** -1.5)
            with warmup_steps=4000:

        sparse categorical crossentropy loss, ignoring padded tokens

        Returns the trained model
    """

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        ''' custom schedule for learning rate '''
        def __init__(self, d_model, warmup_steps=4000):
            ''' constructor '''
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            ''' instance '''
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    data = Dataset(batch_size, max_len)
    # create_masks()
    input_v = data.tokenizer_pt.vocab_size + 2
    target_v = data.tokenizer_en.vocab_size + 2
    transformer = transformer(N, dm, h, hidden, input_v,
                              target_v, max_len, max_len)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        ''' loss '''
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        ''' acc '''
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        ''' train_step '''
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = \
            create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                  transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for e in range(epochs):
        batch = 0
        for (inp, tar) in data.data_train:
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss ' +
                      '{train_loss.result():.f} Accuracy ' +
                      '{train_accuracy.result():.f}')
            batch += 1
        print(f'Epock {epoch+1}: loss {train_loss.result():.f}')
    return model

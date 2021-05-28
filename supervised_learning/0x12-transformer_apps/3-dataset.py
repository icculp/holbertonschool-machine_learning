#!/usr/bin/env python3
"""
    Transformer Applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ loads and preps a dataset for machine translation """

    def __init__(self, batch_size, max_len):
        """
        batch_size is the batch size for training/validation
        max_len is the maximum number of tokens allowed per example sentence

        data_train, which contains the ted_hrlr_translate/pt_to
            en tf.data.Dataset train split, loaded as_supervided
        data_valid, which contains the ted_hrlr_translate/pt
            to_en tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt is the Portuguese tokenizer created
            from the training set
        tokenizer_en is the English tokenizer created
            from the training set
        """
        def filter(d, max):
            ''' filters dataset by max_len '''
            return tf.math.less(d, max)
        self.batch_size = batch_size
        train = tfds.load('ted_hrlr_translate/pt_to_en',
                          split='train', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(train)
        self.data_train = train.map(self.tf_encode)
        '''tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )'''
        self.data_train = self.data_train.filter(lambda x, y: tf.logical_and(
            tf.size(x) <= max_len, tf.size(y) <= max_len))  # filter)
        self.data_train = self.data_train.cache()
        # s = 0
        # for _ in self.data_train:
        #    buffer_size += 1
        s = sum(1 for _ in self.data_train)
        '''s = self.data_train.map(lambda x: 1,
                                num_parallel_calls=tf.data.experimental.
                                AUTOTUNE).reduce(tf.constant(0),
                                                 lambda x, _: x+1)'''
        self.data_train = self.data_train.shuffle(s)
        # self.data_train.cardinality().numpy().size)
        self.data_train = self.data_train.padded_batch(batch_size)
        # , padded_shapes=batch_size)  # , (batch_size, [None]))
        self.data_train = self.data_train.prefetch(tf.data.
                                                   experimental.AUTOTUNE)
        valid = tfds.load('ted_hrlr_translate/pt_to_en',
                          split='validation', as_supervised=True)
        self.data_valid = valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(lambda x, y: tf.logical_and(
            tf.size(x) <= max_len, tf.size(y) <= max_len))
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """ creates sub-word tokenizers for our dataset:
            data is a tf.data.Dataset whose examples are formatted as a tuple
                (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
            The maximum vocab size should be set to 2**15
            Returns: tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
        """
        STE = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = STE.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = STE.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
            The tokenized sentences should include the start and end of
                sentence tokens
            The start token should be indexed as vocab_size
            The end token should be indexed as vocab_size + 1
            Returns: pt_tokens, en_tokens
                pt_tokens is a np.ndarray containing the Portuguese tokens
                en_tokens is a np.ndarray. containing the English tokens
        """
        vocab_en = self.tokenizer_en.vocab_size
        vocab_pt = self.tokenizer_pt.vocab_size
        p = [vocab_pt] + self.tokenizer_pt.encode(pt.numpy()) + [vocab_pt + 1]
        e = [vocab_en] + self.tokenizer_en.encode(en.numpy()) + [vocab_en + 1]
        return p, e

    def tf_encode(self, pt, en):
        ''' acts as a tensorflow wrapper for the encode instance method
            set shape of return tensors
        '''
        tp, te = tf.py_function(func=self.encode,
                                inp=[pt, en],
                                Tout=[tf.int64, tf.int64])
        tp = tf.ensure_shape(tp, [None])
        te = tf.ensure_shape(te, [None])
        # tp.set_shape([None])
        # te.set_shape([None])
        return tp, te

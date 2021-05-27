#!/usr/bin/env python3
"""
    Transformer Applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ loads and preps a dataset for machine translation """

    def __init__(self):
        """ creates the instance attributes:
            data_train, which contains the ted_hrlr_translate/pt_to
                en tf.data.Dataset train split, loaded as_supervided
            data_valid, which contains the ted_hrlr_translate/pt
                to_en tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt is the Portuguese tokenizer created
                from the training set
            tokenizer_en is the English tokenizer created
                from the training set
        """
        train = tfds.load('ted_hrlr_translate/pt_to_en',
                          split='train', as_supervised=True)
        self.data_train = train.map(self.tf_encode)
        #    lambda p, e: tf.py_function(
        #    func=self.tf_encode, inp=[p, e], Tout=[tf.int64, tf.int64]))
        valid = tfds.load('ted_hrlr_translate/pt_to_en',
                          split='validation', as_supervised=True)
        self.data_valid = valid.map(self.tf_encode)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(train)
        #    lambda p, e: tf.py_function(
        #    func=self.tf_encode, inp=[p, e], Tout=[tf.int64, tf.int64]))
        # self.data_train = tf.py_function(func=self.tf_encode,
        #                                  inp=train.map(x, y),
        #                                  Tout=[tf.int64, tf.int64])
        # self.tf_encode(tfds.load('ted_hrlr_translate/pt_to_en',
        #                            split='train', as_supervised=True))
        # self.data_valid = tf.py_function(func=self.tf_encode,
        #                                  inp=tfds.load('ted_hrlr_translate/pt_to_en',
        #                            split='validation', as_supervised=True),
        #                            Tout=[tf.int64, tf.int64])
        # self.tf_encode(tfds.load('ted_hrlr_translate/pt_to_en',
        #                            split='validation', as_supervised=True))

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
        # print('vocab en', vocab_en)
        vocab_pt = self.tokenizer_pt.vocab_size
        # print('vocab pt', vocab_pt)
        # pt = [vocab_pt] + pt.numpy() + [vocab_pt + 1]
        p = [vocab_pt] + self.tokenizer_pt.encode(pt.numpy()) + [vocab_pt + 1]
        # en = [vocab_en] + en.numpy() + [vocab_en + 1]
        e = [vocab_en] + self.tokenizer_en.encode(en.numpy()) + [vocab_en + 1]
        return p, e

    def tf_encode(self, pt, en):
        ''' acts as a tensorflow wrapper for the encode instance method
            set shape of return tensors
        '''
        # print('lenpt', len(pt))
        # print('lenen', len(en))
        # p, e = self.encode(pt, en)
        # tp = tf.convert_to_tensor(p)
        # te = tf.convert_to_tensor(e)
        tp, te = tf.py_function(func=self.encode,
                                inp=[pt, en],
                                Tout=[tf.int64, tf.int64])
        # tf.ensure_shape(tp, [None])
        # tf.ensure_shape(te, [None])
        return tp, te

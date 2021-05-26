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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.\
            tokenize_dataset(self.data_train)

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

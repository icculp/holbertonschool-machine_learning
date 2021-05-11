#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from gensim.models import word2vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ creates and trains a gensim word2vec model
        sentences is a list of sentences to be trained on
        size is the dimensionality of the embedding layer
        min_count minimum number of occurrences of a word for use in training
        window maximum distance between current and predicted word
        negative is the size of negative sampling
        cbow boolean to determine the training type
            True is for CBOW; False is for Skip-gram
        iterations is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
        Returns: the trained model
    """
    # vocab = ['are', 'awesome', 'beautiful', 'cake']
    model = word2vec.Word2Vec(sentences=sentences, size=size,
                              window=window, seed=seed, negative=negative,
                              cbow_mean=int(cbow), min_count=min_count,
                              workers=workers, iter=iterations)
    # model.train(total_examples=model.corpus_count, epochs=model.epochs)
    # epochs=iterations)
    return model

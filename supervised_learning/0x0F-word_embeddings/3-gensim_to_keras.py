#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from gensim.models import word2vec


def gensim_to_keras(model):
    """ converts a gensim word2vec model to a keras Embedding layer
        model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding()

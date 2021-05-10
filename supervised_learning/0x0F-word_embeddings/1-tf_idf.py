#!/usr/bin/env python3
"""
    NLP - Word Embeddings
"""
import numpy as np
from sklearn.feature_extraction import text


def tf_idf(sentences, vocab=None):
    """ creates a TF-IDF embedding
        sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used
        Returns: embeddings, features
            embeddings ndarray (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features list of the features used for embeddings
    """
    # vocab = ['are', 'awesome', 'beautiful', 'cake']
    Vectorizer = text.TfidfVectorizer(vocabulary=vocab)
    embeddings = Vectorizer.fit_transform(sentences)
    return embeddings.toarray(), Vectorizer.get_feature_names()

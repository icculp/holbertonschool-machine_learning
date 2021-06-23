#!/usr/bin/env python3
"""
    QA Bots!
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """ performs semantic search on a corpus of documents
        corpus_path is the path to the corpus of reference
            documents on which to perform semantic search
        sentence is the sentence from which to perform
            semantic search
        Returns: the reference text of the document most
            similar to sentence
    """
    # tokenizer = BertTokenizer.from_pretrained('tokenizer_tf2_qa')
    # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-' +
    #                                          'word-masking-finetuned-squad')
    document = [sentence]
    model = hub.load('https://tfhub.dev/google/universal-sentence' +
                     '-encoder-large/5')
    for fn in os.listdir(corpus_path):
        if not fn.endswith('.md'):
            continue
        with open(corpus_path + '/' + fn, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    embeddings = model(documents)
    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:])
    similar = documents[closest + 1]
    return similar

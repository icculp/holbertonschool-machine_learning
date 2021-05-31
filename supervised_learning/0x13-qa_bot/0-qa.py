#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """ that finds a snippet of text within a reference document to answer a question:

        question is a string containing the question to answer
        reference is a string containing the reference document from which to find the answer
        Returns: a string containing the answer
            If no answer is found, return None
    """
    model = bert-uncased-tf2-qa
    bert = BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad

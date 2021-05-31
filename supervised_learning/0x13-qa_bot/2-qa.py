#!/usr/bin/env python3
"""
    QA Bots!
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """ that answers questions from a reference text:
        reference is the reference text
    """
    stop = ['exit', 'quit', 'goodbye', 'bye']
    while (1):
        question = input("Q: ")
        for word in stop:
            if word == question.lower():
                print("A: Goodbye")
                exit()
                !kill  # if in colab
        A = question_answer(question, reference)
        if A is None:
            A = 'Sorry, I do not understand your question.'
        print("A:", A)

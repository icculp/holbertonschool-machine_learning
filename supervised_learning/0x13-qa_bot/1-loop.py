#!/usr/bin/env python3
import sys
question_answer = __import__('0-qa').question_answer


stop = ['exit', 'quit', 'goodbye', 'bye']
while (1):
    question = input("Q: ")
    for word in stop:
        if word == question.lower():
            print("A: Goodbye")
            exit()
    print("A:")  # , question)

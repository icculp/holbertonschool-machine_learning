#!/usr/bin/env python3
""" 30-main """
from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """ changes all topics of a school document based on the name:
        mongo_collection will be the pymongo collection object
        name (string) will be the school name to update
        topics (list of strings) will be the list of topics approached in the school
    """
    mongo_collection.update({'name': name}, {'name': name, 'topics': topics}, {'multi': 'true'})


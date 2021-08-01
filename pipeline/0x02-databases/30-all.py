#!/usr/bin/env python3
""" 30-main """
from pymongo import MongoClient


def list_all(mongo_collection):
    """ list all docs in a mongo collection """
    return mongo_collection.find()

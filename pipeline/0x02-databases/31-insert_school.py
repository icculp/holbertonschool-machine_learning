#!/usr/bin/env python3
""" 30-main """
from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """ list all docs in a mongo collection """
    obj = mongo_collection.insert(kwargs)
    # print(dir(obj))
    # doc_id = obj.id
    return obj

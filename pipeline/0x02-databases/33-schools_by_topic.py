#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient


def schools_by_topic(mongo_collection, topic):
    """ returns list of schools containing topic """
    return mongo_collection.find({'topics': topic})

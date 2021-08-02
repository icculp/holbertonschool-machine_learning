#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx
    logs = nginx.find()
    get = nginx.find({'method': 'GET'})
    post = nginx.find({'method': 'POST'})
    put = nginx.find({'method': 'PUT'})
    patch = nginx.find({'method': 'PATCH'})
    delete = nginx.find({'method': 'DELTE'})
    status = nginx.find({'method': 'GET', 'path': '/status'})
    print("{} logs".format(logs.count()))
    print("Methods:\n" +
          "\tmethod GET: {}\n".format(get.count()) +
          "\tmethod POST: {}\n".format(post.count()) +
          "\tmethod PUT: {}\n".format(put.count()) +
          "\tmethod PATCH: {}\n".format(patch.count()) +
          "\tmethod DELETE: {}".format(delete.count()))
    print("{} status check".format(status.count()))

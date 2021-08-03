#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx
    logs = nginx.find()
    # print(dir(logs))
    # print(logs.collection)
    ips = nginx.aggregate([
                          {'$group': {'_id': '$ip', 'sum': {'$sum': 1}}},
                          {"$sort": {"sum": -1}},
                          {"$limit": 10}
                          ])
    '''pipe = [{"$unwind": "$topics"},
            {"$group":
                {"_id": '$_id',
                 "averageScore": {"$avg": "$topics.score"},
                 "name": {'$first': '$name'}
                 }
             },
            {"$sort": {"averageScore": -1}}
            ]
    logs.aggregate(pipe)'''
    # ips = logs.count('ip')#.sort("ip", -1)
    # #.limit(10)#logs.sort("ip", -1).limit(10)
    # for document in logs:
    #    print(document.keys())
    get = nginx.find({'method': 'GET'})
    post = nginx.find({'method': 'POST'})
    put = nginx.find({'method': 'PUT'})
    patch = nginx.find({'method': 'PATCH'})
    delete = nginx.find({'method': 'DELETE'})
    status = nginx.find({'method': 'GET', 'path': '/status'})
    print("{} logs".format(logs.count()))
    print("Methods:\n" +
          "\tmethod GET: {}\n".format(get.count_documents()) +
          "\tmethod POST: {}\n".format(post.count_documents()) +
          "\tmethod PUT: {}\n".format(put.count_documents()) +
          "\tmethod PATCH: {}\n".format(patch.count_documents()) +
          "\tmethod DELETE: {}".format(delete.count_documents()))
    print("{} status check".format(status.count_documents()))
    print("IPs:")
    for doc in ips:
        print("\t{}: {}".format(doc.get('_id'), doc.get('sum')))
#     +
#          "{}".format(ips)
#    )

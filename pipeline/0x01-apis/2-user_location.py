#!/usr/bin/env python3
"""
    Barely an API project.
"""
import requests
import sys
import time


if __name__ == "__main__":
    """ prints location of user specified as cli arg """
    user_url = sys.argv[1]
    i = 60
    # while i:
    r = requests.get(user_url)
    if r.status_code != 200:
        if r.status_code == 403:
            # print(r.headers)
            reset_time = int(r.headers.get('X-Ratelimit-Reset'))
            now = time.time()
            minutes = reset_time - now
            minutes = round(minutes / 60)
            # print(minutes)
            print("Reset in {} min".format(minutes))
            exit()
    user = r.json()
    #    i -= 1
    location = user.get('location')
    if location:
        print(user.get('location'))
    else:
        print("Not found")

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
            minutes = int(minutes / 60)
            # print(minutes)
            print("Reset in {} minutes".format(minutes))
            exit()
    user = r.json()
    #    i -= 1
    location = user.get('location')
    if location:
        print(user.get('location'))
    else:
        print("Not found")


def sentientPlanets():
    """ returns the list of names of the home
            planets of all sentient species
    """
    sentient_planets = []
    species = []
    url = 'https://swapi-api.hbtn.io/api/species/?format=json'
    while 1:
        # print(1)
        r = requests.get(url).json()
        species += r.get('results')
        next = r.get('next')
        if next:
            url = next
        else:
            break
    # print(species)
    for specie in species:
        # print('looping')
        try:
            if specie.get('designation') == 'sentient' or\
                    specie.get('classification') == 'sentient':
                planet_url = specie.get('homeworld')
                if planet_url:
                    name = requests.get(planet_url).json().get('name')
                    sentient_planets.append(name)
        except Exception as e:
            print(e, specie.get('name'))
            continue
    return sentient_planets

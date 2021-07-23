#!/usr/bin/env python3
"""
    Barely an API project.
"""
import requests


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

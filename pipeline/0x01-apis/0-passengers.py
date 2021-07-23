#!/usr/bin/env python3
"""
    Barely an API project.
"""
import requests


def availableShips(passengerCount):
    """ returns the list of ships that can hold a
            given number of passengers
    """
    if passengerCount < 1:
        return []
    suitable_ships = []
    starships = []
    url = 'https://swapi-api.hbtn.io/api/starships/?format=json'
    while 1:
        r = requests.get(url).json()
        starships += r.get('results')
        next = r.get('next')
        if next:
            url = next
        else:
            break
    for ship in starships:
        try:
            if int(ship.get('passengers').replace(',', '')) >= passengerCount:
                suitable_ships.append(ship.get('name'))
        except Exception as e:
            # print(e, ship.get('name'))
            continue
    return suitable_ships

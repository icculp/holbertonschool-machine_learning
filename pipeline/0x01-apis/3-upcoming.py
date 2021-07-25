#!/usr/bin/env python3
"""
    Barely an API project.
"""
import requests
import time


if __name__ == "__main__":
    """ prints location of user specified as cli arg """
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    r_api = requests.get(url)
    j = r_api.json()[0]
    launch_name = j.get('name')
    rocket_id = j.get('rocket')
    date = j.get('date_local')
    rocket_url = 'https://api.spacexdata.com/v4/rockets/' + rocket_id
    r_rocket = requests.get(rocket_url)
    rocket_name = r_rocket.json().get('name')
    launch_id = j.get('launchpad')
    launchpad_url = 'https://api.spacexdata.com/v4/launchpads/' + launch_id
    r_launchpad = requests.get(launchpad_url)
    launchpad_name = r_launchpad.json().get('name')
    launchpad_locality = r_launchpad.json().get('locality')
    print("{} ({}) {} - {} ({})".format(launch_name,
                                        date,
                                        rocket_name,
                                        launchpad_name,
                                        launchpad_locality))

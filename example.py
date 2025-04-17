#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
StarryNet: empowering researchers to evaluate futuristic integrated space and terrestrial networks.
author: Zeqi Lai (zeqilai@tsinghua.edu.cn) and Yangtao Deng (dengyt21@mails.tsinghua.edu.cn)
"""

from starrynet.sn_observer import *
from starrynet.sn_orchestrater import *
from starrynet.sn_synchronizer import *

import subprocess
import time

import numpy as np

# kill all docker containers
# print("Removing all docker containers...")
# subprocess.call("docker rm -f $(docker ps -aq)", shell=True, stdout=subprocess.DEVNULL)
# wait for 5 seconds
# time.sleep(5)

# Starlink 5*5: 25 satellite nodes, 2 ground stations.
# The node index sequence is: 25 sattelites, 2 ground stations.
# In this example, 25 satellites and 2 ground stations are one AS.

with open('config.json', 'r') as f:
    config = json.load(f)
    n_orbits = config['# of orbit']
    n_satellites = config['# of satellites']
    n_ground_stations = config['GS number']
    n_nodes = (n_orbits * n_satellites) + n_ground_stations

AS = [[1, n_nodes]]  # Node #1 to Node #27 are within the same AS.
# GS_lat_long = [[50.110924, 8.682127], [46.635700, 14.311817]
#                 ]  # latitude and longitude of frankfurt and  Austria

# List of approximate [latitude, longitude] coordinates for LEO ground stations.
# Based on publicly known/reported locations for Starlink and other providers (AWS, OneWeb).
# Coordinates are approximate.

ground_station_locations = [
    # North America
    [33.92, -118.35],  # Hawthorne, CA, USA (Starlink HQ/Gateway)
    [47.67, -122.12],  # Redmond, WA, USA (Starlink Gateway)
    [48.17, -111.94],  # Conrad, MT, USA (Starlink Gateway)
    [44.44, -90.84],   # Merrillan, WI, USA (Starlink Gateway)
    [25.99, -97.15],   # Boca Chica, TX, USA (Starlink Gateway/Starbase)
    [47.56, -52.71],   # St. John's, NL, Canada (Starlink Gateway)

    # Europe
    [50.98, 2.12],     # Gravelines, France (Starlink Gateway)
    [50.33, 8.53],     # Usingen, Germany (Near Frankfurt, Starlink Gateway)
    [50.05, -5.18],    # Goonhilly Downs, UK (Starlink/Multi-Operator Gateway)
    [40.41, -3.70],    # Madrid, Spain (Reported Starlink Gateway Area)
    [38.33, 23.56],    # Tanagra, Greece (OneWeb Gateway)
    [78.22, 15.65],    # Svalbard, Norway (OneWeb/KSAT Gateway)
    [50.110924, 8.682127], # Frankfurt, Germany (Starlink Gateway)
    [46.635700, 14.311817], # Austria (Starlink Gateway)

    # Oceania
    [-36.40, 174.66],  # Warkworth, New Zealand (Starlink Gateway)
    [-32.95, 151.65],  # Boolaroo, NSW, Australia (Starlink Gateway)

    # South America
    [-33.45, -70.67],  # Santiago, Chile (Starlink/AWS Gateway)
    [-23.55, -46.63],  # São Paulo, Brazil (AWS Gateway)

    # Asia
    [26.07, 50.55],    # Bahrain (AWS Gateway)
    [1.35, 103.82],   # Singapore (AWS Gateway)

    # Africa
    [-33.92, 18.42],   # Cape Town, South Africa (AWS Gateway)
    [-4.17, 39.45]     # Kwale, Kenya (OneWeb Gateway)
]

# Convert list of lists to numpy array first
ground_station_array = np.array(range(len(ground_station_locations)))
chosen_indices = np.random.choice(ground_station_array, 2, replace=False)
GS_lat_long = [ground_station_locations[i] for i in chosen_indices]  # Get the actual coordinates

configuration_file_path = "./config.json"
hello_interval = 10  # hello_interval(s) in OSPF. 1-200 are supported.

print('Start StarryNet.')
sn = StarryNet(configuration_file_path, GS_lat_long, hello_interval, AS)
sn.stop_emulation() # stop emulation before creating nodes
sn.create_nodes()
sn.create_links()
sn.run_routing_deamon()

print('Creating RTC nodes.')
sn.create_rtc_nodes()

# # distance between nodes at a certain time
# node_distance = sn.get_distance(node_index1, node_index2, time_index)
# print("node_distance (km): " + str(node_distance))

# # neighbor node indexes of node at a certain time
# neighbors_index = sn.get_neighbors(node_index1, time_index)
# print("neighbors_index: " + str(neighbors_index))

# # LLA of a node at a certain time
# LLA = sn.get_position(node_index1, time_index)
# print("LLA: " + str(LLA))

# ratio = 0.3
# time_index = 5
# # random damage of a given ratio at a certain time
# sn.set_damage(ratio, time_index)

sn.set_video_call(26, 27, 5)  # start video call. NOTE: these indices are not used as sender and receiver are already set.

print("Starting emulation...")
start_time = time.time()
sn.start_emulation()
print("Emulation completed in " + str(time.time() - start_time) + " seconds")
sn.stop_emulation()

# clear /tmp/* directory
print("Cleaning up /tmp directory...")
subprocess.run(['sudo', 'rm', '-rf', '/tmp/*'])
print("Cleanup complete.")


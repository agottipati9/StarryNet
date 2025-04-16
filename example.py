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
GS_lat_long = [[50.110924, 8.682127], [46.635700, 14.311817]
                ]  # latitude and longitude of frankfurt and  Austria
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

sn.start_emulation()
sn.stop_emulation()

# TODO: clean output directory, logging, plotting, etc


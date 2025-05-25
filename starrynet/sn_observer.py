#encoding: utf-8
import math
from sgp4.api import Satrec, WGS84
from skyfield.api import load, wgs84, EarthSatellite
from datetime import datetime
import numpy as np
import os
import torch
from glob import glob

from starrynet.sn_utils import *

_ = inf = 999999  # inf

# To calculate the connection between satellites and GSes in time_in
# GS_num: number of ground stations


class Observer():

    def __init__(self, file_path, configuration_file_path, inclination,
                 satellite_altitude, orbit_number, sat_number, duration,
                 antenna_number, GS_lat_long, antenna_inclination,
                 intra_routing, hello_interval, AS, tle_file_path, add_maneuvers):
        self.file_path = file_path
        self.configuration_file_path = configuration_file_path
        self.inclination = inclination
        self.satellite_altitude = satellite_altitude
        self.orbit_number = orbit_number
        self.sat_number = sat_number
        self.duration = duration
        self.antenna_number = antenna_number
        self.GS_lat_long = GS_lat_long
        self.antenna_inclination = antenna_inclination
        self.intra_routing = intra_routing
        self.hello_interval = hello_interval
        self.AS = AS
        self.tle_file_path = tle_file_path
        self.add_maneuvers = add_maneuvers

    def access_P_L_shortest(self, sat_cbf, GS_cbf, GS_num, sat_num,
                            orbit_number, sat_number, duration, fac_ll,
                            sat_lla, bound_dis, alpha, antenna_num, path, sat_vel):
        delay_matrix = np.zeros((GS_num + sat_num, GS_num + sat_num))
        
        # Create directory for logging if it doesn't exist
        # NOTE: For storing satellite specific features
        satellite_feature_dir = path + "/satellite_features"
        if not os.path.exists(satellite_feature_dir):
            os.makedirs(satellite_feature_dir)
            
        for cur_time in range(duration):
            for i in range(0, GS_num):
                access_list = {}
                fac_lat = float(fac_ll[i][0])  # latitude
                up_lat = fac_lat + alpha  # bound
                down_lat = fac_lat - alpha
                x2 = GS_cbf[i][0]
                y2 = GS_cbf[i][1]
                z2 = GS_cbf[i][2]
                
                # Store satellite info for logging
                sat_info = {}
                
                for j in range(0, sat_num):
                    if sat_lla[cur_time][j][0] >= down_lat and sat_lla[
                            cur_time][j][0] <= up_lat:
                        x1 = sat_cbf[cur_time][j][0]  # in km
                        y1 = sat_cbf[cur_time][j][1]
                        z1 = sat_cbf[cur_time][j][2]
                        dist = math.sqrt(
                            np.square(x1 - x2) + np.square(y1 - y2) +
                            np.square(z1 - z2))
                        if dist < bound_dis:
                            # [satellite index，distance]
                            access_list.update({j: dist})
                            sat_info.update({
                                j: {
                                    'distance': dist,
                                    'lla': sat_lla[cur_time][j],
                                    'velocity': sat_vel[cur_time][j],
                                    'delay': -1
                                }
                            })
                
                
                sorted_access_list = dict(sorted(access_list.items(), key=lambda item: item[1]))
                if len(access_list) > antenna_num:
                    cnt = 0
                    # NOTE: log satellite-specific features
                    for key, value in sorted_access_list.items():
                        delay = value / (17.31 / 29.5 * 299792.458) * 1000  # ms
                        sat_info[key]['delay'] = delay
                    for key, value in sorted_access_list.items():
                        cnt = cnt + 1
                        if cnt > antenna_num:
                            break
                        delay_time = value / (17.31 / 29.5 *
                                              299792.458) * 1000  # ms
                        delay_matrix[sat_num + i][key] = delay_time
                        delay_matrix[key][sat_num + i] = delay_time
                elif len(access_list) != 0:
                    for key, value in sorted_access_list.items():
                        delay_time = value / (17.31 / 29.5 *
                                              299792.458) * 1000  # ms
                        delay_matrix[sat_num + i][key] = delay_time
                        delay_matrix[key][sat_num + i] = delay_time
                # # NOTE: make sure the length of sat_info is 5
                if len(sat_info) > 5:
                    sat_info = dict(list(sat_info.items())[:5])
                elif len(sat_info) < 5:
                    for index in range(5 - len(sat_info)):
                        sat_info.update({
                            -index - 1: {
                                'distance': -1,
                                'lla': -1, 
                                'velocity': -1,
                                'delay': -1
                            }
                        })
                # Log the information
                log_file = f"{satellite_feature_dir}/gs_{i}_time_{cur_time:03d}.txt"
                with open(log_file, 'w') as f:
                    f.write(f"Ground Station {i} at time {cur_time}\n")
                    f.write(f"GS Location: {fac_ll[i]}\n\n")
                    for k, sat in enumerate(sat_info, 1):
                        f.write(f"Satellite {k}:\n")
                        f.write(f"ID: {sat}\n")
                        f.write(f"Distance: {sat_info[sat]['distance']} km\n")
                        f.write(f"LLA: {sat_info[sat]['lla']}\n")
                        f.write(f"Velocity: {sat_info[sat]['velocity']} km/s\n")
                        f.write(f"Delay: {sat_info[sat]['delay']} ms\n")
                        f.write("\n")

            for i in range(orbit_number):
                for j in range(sat_number):
                    num_sat1 = i * sat_number + j
                    x1 = sat_cbf[cur_time][num_sat1][0]  # km
                    y1 = sat_cbf[cur_time][num_sat1][1]
                    z1 = sat_cbf[cur_time][num_sat1][2]
                    num_sat2 = i * sat_number + (
                        j + 1) % sat_number
                    x2 = sat_cbf[cur_time][num_sat2][0]  # km
                    y2 = sat_cbf[cur_time][num_sat2][1]
                    z2 = sat_cbf[cur_time][num_sat2][2]
                    num_sat3 = ((i + 1) % orbit_number) * sat_number + j
                    x3 = sat_cbf[cur_time][num_sat3][0]  # km
                    y3 = sat_cbf[cur_time][num_sat3][1]
                    z3 = sat_cbf[cur_time][num_sat3][2]
                    delay1 = math.sqrt(
                        np.square(x1 - x2) + np.square(y1 - y2) +
                        np.square(z1 - z2)) / (17.31 / 29.5 *
                                               299792.458) * 1000  # ms
                    delay2 = math.sqrt(
                        np.square(x1 - x3) + np.square(y1 - y3) +
                        np.square(z1 - z3)) / (17.31 / 29.5 *
                                               299792.458) * 1000  # ms
                    delay_matrix[num_sat1][num_sat2] = delay1
                    delay_matrix[num_sat2][num_sat1] = delay1
                    delay_matrix[num_sat1][num_sat3] = delay2
                    delay_matrix[num_sat3][num_sat1] = delay2
            np.savetxt(path + "/delay/" + str(cur_time + 1) + ".txt",
                       delay_matrix,
                       fmt='%.2f',
                       delimiter=',')
            for i in range(len(delay_matrix)):
                delay_matrix[i, ...] = 0

    def to_cbf(self, lat_long,
               length):  # the xyz coordinate system. length: number of nodes
        cbf = []
        radius = 6371
        for num in range(0, length):
            cbf_in = []
            R = radius
            if len(lat_long[num]) > 2:
                R += lat_long[num][2]
            z = R * math.sin(math.radians(float(lat_long[num][0])))
            x = R * math.cos(math.radians(float(
                lat_long[num][0]))) * math.cos(
                    math.radians(float(lat_long[num][1])))
            y = R * math.cos(math.radians(float(
                lat_long[num][0]))) * math.sin(
                    math.radians(float(lat_long[num][1])))
            cbf_in.append(x)
            cbf_in.append(y)
            cbf_in.append(z)
            cbf.append(cbf_in)
        return cbf  # xyz coordinates of all the satellites

    def calculate_bound(self, inclination_angle, height):
        bound_distance = 6371 * math.cos(
            (90 + inclination_angle) / 180 * math.pi) + math.sqrt(
                math.pow(
                    6371 * math.cos(
                        (90 + inclination_angle) / 180 * math.pi), 2) +
                math.pow(height, 2) + 2 * height * 6371)
        return bound_distance

    def matrix_to_change(self, duration, orbit_number, sat_number, path,
                         GS_lat_long):
        no_fac = len(GS_lat_long)
        no_geo = 0
        duration = duration - 1
        no_leo = orbit_number * sat_number

        topo_duration = [[[0 for i in range(no_leo + no_geo + no_fac)]
                          for j in range(no_leo + no_geo + no_fac)]
                         for k in range(duration)]
        for time in range(1, duration + 1):
            topo_path = path + '/delay/' + str(time) + ".txt"
            adjacency_matrix = sn_get_param(topo_path)
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix[i])):
                    if float(adjacency_matrix[i][j]) > 0:
                        adjacency_matrix[i][j] = 1
                    else:
                        adjacency_matrix[i][j] = 0
            topo_duration[time - 1] = adjacency_matrix

        changetime = []
        Duration = []
        for i in range(duration - 1):
            l1 = topo_duration[i]
            l2 = topo_duration[i + 1]
            if l1 == l2:
                continue
            else:
                changetime.append(i)
        pretime = 0
        for item in changetime:
            Duration.append(item - pretime)
            pretime = item
        Duration.append(60)

        topo_leo_change_path = path + "/Topo_leo_change.txt"
        f = open(topo_leo_change_path, "w")
        cnt = 1
        for i in range(duration - 1):
            pre_lines = topo_duration[i]
            now_lines = topo_duration[i + 1]
            if pre_lines == now_lines:
                continue
            else:
                f.write("time " + str(i + 2) + ":\n")  # time started from 1
                f.write('duration ' + str(Duration[cnt]) + ":\n")
                cnt += 1
                f.write("add:\n")
                for j in range(no_fac):
                    prelines = pre_lines[no_geo + no_leo + j]
                    nowlines = now_lines[no_geo + no_leo + j]
                    for k in range(no_geo + no_leo + no_fac):
                        if prelines[k] == 0 and nowlines[k] == 1:
                            f.write(
                                str(k + 1) + "-" + str(no_leo + j + 1) +
                                "\n")  # index
                f.write("del:\n")
                for j in range(no_fac):
                    prelines = pre_lines[no_geo + no_leo + j]
                    nowlines = now_lines[no_geo + no_leo + j]
                    for k in range(no_geo + no_leo + no_fac):
                        if prelines[k] == 1 and nowlines[k] == 0:
                            f.write(
                                str(k + 1) + "-" + str(no_leo + j + 1) +
                                "\n")  # index
        f.write("time " + str(self.duration) + ":\n")  #
        f.write("end of the emulation! \n")  #
        f.close()
        cnt = 1

    def calculate_delay(self):
        path = self.configuration_file_path + "/" + self.file_path
        sat_cbf = []  # first dimension: time. second dimension: node. third dimension: xyz
        sat_lla = []  # first dimension: time. second dimension: node. third dimension: lla
        sat_vel = []  # first dimension: time. second dimension: node. third dimension: velocity
        GS_cbf = []  # first dimension: node. second dimension: xyz

        if os.path.exists(path + '/delay') == True:
            osstr = "rm -f " + path + "/delay/*"
            os.system(osstr)
        else:
            os.system("mkdir " + path)
            os.system("mkdir " + path + "/delay")
        if os.path.exists(path + '/position') == True:
            osstr = "rm -f " + path + "/position/*"
            os.system(osstr)
        else:
            os.system("mkdir " + path + "/position")

        ts = load.timescale()
        since = datetime(1949, 12, 31, 0, 0, 0)
        start = datetime(2022, 1, 1, 0, 0, 0)
        epoch = (start - since).days
        duration = self.duration  # second
        result = [[] for i in range(duration)]  # LLA result
        lla_per_sec = [[] for i in range(duration)]  # LLA result
        velocities_per_sec = [[] for i in range(duration)]  # velocity result
        bound_dis = self.calculate_bound(self.antenna_inclination, self.satellite_altitude) * 29.5 / 17.31
        orbit_inclination = self.inclination * 2 * np.pi / 360
        alpha = np.degrees(
        np.arccos(6371 / (6371 + self.satellite_altitude) *
                    np.cos(np.radians(orbit_inclination)))) - orbit_inclination

        # Use real TLE data
        if self.tle_file_path:
            print(f"Using TLE file: {self.tle_file_path}")
            satellites = load.tle_file(self.tle_file_path)
            num_of_orbit = self.orbit_number
            sat_per_orbit = self.sat_number
            num_of_sat = num_of_orbit * sat_per_orbit
            if len(satellites) < num_of_sat:
                # NOTE: ideally, the number of satellites in the TLE file should be equal to the number of satellites in the simulation
                # However, this is not always the case (hardware limitations), so we need to adjust the number of satellites or the TLE file
                print(f"Error: TLE file contains less than {num_of_sat} satellites. Adjust the number of satellites or the TLE file.")
                raise ValueError("TLE file contains less than the number of satellites in the simulation")
            # Use TLE data to generate satellite positions
            for i in range(num_of_sat):
                sat = satellites[i]
                cur = datetime(2025, 5, 12, 1, 0, 0)
                t_ts = ts.utc(*cur.timetuple()[:5], range(duration))
                geocentric = sat.at(t_ts)
                subpoint = wgs84.subpoint(geocentric)

                # Extract velocities in ECI coordinates
                for t in range(duration):
                    velocity = float(np.linalg.norm(geocentric.velocity.km_per_s[:, t]))
                    velocities_per_sec[t].append(velocity)

                # Get position data
                for t in range(duration):
                    lla = '%f,%f,%f\n' % (subpoint.latitude.degrees[t],
                                        subpoint.longitude.degrees[t],
                                        subpoint.elevation.km[t])
                    result[t].append(lla)
                    lla = []
                    lla.append(float(subpoint.latitude.degrees[t]))
                    lla.append(float(subpoint.longitude.degrees[t]))
                    lla.append(float(subpoint.elevation.km[t]))
                    lla_per_sec[t].append(lla)
        # Use SGP4 to generate satellite positions
        else:
            print("Using SGP4 to generate satellite positions")
            GM = 3.9860044e14
            R = 6371393
            altitude = self.satellite_altitude * 1000
            num_of_orbit = self.orbit_number
            sat_per_orbit = self.sat_number
            num_of_sat = num_of_orbit * sat_per_orbit
            p_maneuver = np.random.uniform(0.05, 0.15)
            F = 18
            for i in range(num_of_orbit):  # range(num_of_orbit)
                for j in range(sat_per_orbit):  # range(sat_per_orbit)
                    raan = i / num_of_orbit * 2 * np.pi
                    mean_anomaly = (j * 360 / sat_per_orbit + i * 360 * F /
                                    num_of_sat) % 360 * 2 * np.pi / 360
                    do_maneuver = np.random.uniform(0, 1) < p_maneuver
                    # domain randomization
                    if self.add_maneuvers and do_maneuver:
                        progress = np.random.uniform(0.2, 1.0)  # 20-100% through maneuver
                        perturbed_params = add_maneuver_perturbations(
                            base_satellite_altitude_km=self.satellite_altitude,
                            base_mean_anomaly_deg=mean_anomaly,
                            base_raan_rad=raan,
                            maneuver_progress=progress,
                        )
                    else:
                        perturbed_params = add_noise_to_sgp4_params(
                            base_satellite_altitude_km=self.satellite_altitude,
                            base_raan_rad=raan,
                            base_mean_anomaly_deg=mean_anomaly,
                        )
                    altitude = perturbed_params['altitude_km'] * 1000
                    raan = perturbed_params['raan_rad']
                    mean_anomaly = perturbed_params['mean_anomaly_deg']
                    mean_motion = np.sqrt(GM / (R + altitude)**3) * 60
                    satrec = Satrec()
                    satrec.sgp4init(
                        WGS84,  # gravity model
                        'i',  # 'a' = old AFSPC mode, 'i' = improved mode
                        i * sat_per_orbit + j,  # satnum: Satellite number
                        epoch,  # epoch: days since 1949 December 31 00:00 UT
                        2.8098e-05,  # bstar: drag coefficient (/earth radii)
                        6.969196665e-13,  # ndot: ballistic coefficient (revs/day)
                        0.0,  # nddot: second derivative of mean motion (revs/day^3)
                        0.001,  # ecco: eccentricity
                        0.0,  # argpo: argument of perigee (radians)
                        orbit_inclination,  # inclo: inclination (radians)
                        mean_anomaly,  # mo: mean anomaly (radians)
                        mean_motion,  # no_kozai: mean motion (radians/minute)
                        raan,  # nodeo: right ascension of ascending node (radians)
                    )
                    sat = EarthSatellite.from_satrec(satrec, ts)
                    cur = datetime(2022, 1, 1, 1, 0, 0)
                    t_ts = ts.utc(*cur.timetuple()[:5],
                                range(duration))  # [:4]:minute，[:5]:second
                    geocentric = sat.at(t_ts)
                    subpoint = wgs84.subpoint(geocentric)

                    # NOTE: for satellite features, we use velocity vectors from SGP4
                    # Extract velocities in ECI coordinates
                    # Get all velocities at once and compute norms
                    for t in range(duration):
                        velocity = float(np.linalg.norm(geocentric.velocity.km_per_s[:, t]))
                        velocities_per_sec[t].append(velocity)

                    # list: [subpoint.latitude.degrees] [subpoint.longitude.degrees] [subpoint.elevation.km]
                    for t in range(duration):
                        lla = '%f,%f,%f\n' % (subpoint.latitude.degrees[t],
                                            subpoint.longitude.degrees[t],
                                            subpoint.elevation.km[t])
                        result[t].append(lla)
                        lla = []
                        lla.append(float(subpoint.latitude.degrees[t]))
                        lla.append(float(subpoint.longitude.degrees[t]))
                        lla.append(float(subpoint.elevation.km[t]))
                        lla_per_sec[t].append(lla)

        for t in range(duration):
            file = path + '/position/' + '%d.txt' % t
            with open(file, 'w') as fw:
                fw.writelines(result[t])
            cbf_per_sec = self.to_cbf(lla_per_sec[t], num_of_sat)
            sat_cbf.append(cbf_per_sec)
            sat_lla.append(lla_per_sec[t])
            sat_vel.append(velocities_per_sec[t])
        if len(self.GS_lat_long) != 0:
            GS_cbf = self.to_cbf(self.GS_lat_long, len(self.GS_lat_long))

        self.access_P_L_shortest(sat_cbf, GS_cbf, len(self.GS_lat_long),
                                 self.sat_number * self.orbit_number,
                                 self.orbit_number, self.sat_number,
                                 self.duration, self.GS_lat_long, sat_lla,
                                 bound_dis, alpha, self.antenna_number, path, sat_vel)
        self.matrix_to_change(self.duration, self.orbit_number,
                              self.sat_number, path, self.GS_lat_long)

    # TODO: use model to set queue size in AlphaRTC
    def set_queue_size(self, model, queue_script='/opt/home_dir/StarryNet/adjust_queue_size.py'):
        path = self.configuration_file_path + "/" + self.file_path + "/satellite_features"
        sat_features = self.parse_satellite_features(path)
        queue_sizes = [600, 900] 
        with torch.no_grad():
            prediction = model(sat_features).argmax(dim=1).item()
        # TODO: we need to write model features to disk for future use
        queue_size = queue_sizes[prediction]
        print(f"Queue size: {queue_size}")
        os.system(f"python {queue_script} --queue_size {queue_size}")

    def parse_satellite_features(self, folder):
        # TODO: parse all 125 x 2 files for both ground stations in the satellite_features subfolder
        gs_ids = ['gs_0'] #, 'gs_1']
        sat_features = {}
        for gs_id in gs_ids:
            sat_features[gs_id] = torch.zeros(125, 35)
            features_files = glob(os.path.join(folder, 'satellite_features', f'{gs_id}*.txt'))
            features_files.sort()  # ensure files are in chronological order
            for time_step, features_file in enumerate(features_files):
                with open(features_file, 'r') as f:
                    lines = f.readlines()
                    # Skip header lines
                    satellite_features = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('LLA:'):  # NOTE: lets skip GS Location for now
                            if '[' not in line:
                                coords = [-1.0, -1.0, -1.0]
                            else:
                                line_arr = line.split(':')[1].strip().strip('[]').split(',')
                                coords = [float(coord) for coord in line_arr]
                            satellite_features.extend(coords)
                        elif line.startswith('ID:'):
                            satellite_id = int(line.split(':')[1].strip())
                            satellite_features.append(satellite_id)
                        elif line.startswith('Distance:') or line.startswith('Velocity:') or line.startswith('Delay:'):
                            value = float(line.split(' ')[1].strip())
                            satellite_features.append(value)
                    sat_features[gs_id][time_step] = torch.tensor(satellite_features)
        sat_features = torch.stack(list(sat_features.values()), dim=0)
        sat_features = self.normalize_observations(sat_features)
        sat_features = sat_features.reshape(1, 125, 35)
        return sat_features

    def normalize_observations(self, observations):
        # observations is (N x 125 x 35) tensor
        feature_columns = {
            'sat_id': [0, 7, 14, 21, 28],
            'distance': [1, 8, 15, 22, 29],
            'longitude': [2, 9, 16, 23, 30],
            'latitude': [3, 10, 17, 24, 31],
            'altitude': [4, 11, 18, 25, 32],
            'velocity': [5, 12, 19, 26, 33],
            'delay': [6, 13, 20, 27, 34]
        }
        for feature_name, columns in feature_columns.items():
            # normalize each column
            for i in columns:
                feature = observations[:, :, i]
                observations[:, :, i] = self.normalize_feature(feature, feature_type=feature_name)
        return observations

    def normalize_feature(self,feature, feature_type='sat_id'):
        if feature_type == 'sat_id':
            feature = feature / 100  # 0-100 (all positive IDs are valid)
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'distance':
            min_feature = 538.0  # (all negative values are invalid)
            max_feature = 1915.0
            feature = (feature - min_feature) / (max_feature - min_feature)
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'altitude':
            min_feature = 536.0 # (all negative values are invalid)
            max_feature = 581.0
            feature = (feature - min_feature) / (max_feature - min_feature)
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'velocity': 
            min_feature = 7.5  # (all negative values are invalid)
            max_feature = 7.6
            feature = (feature - min_feature) / (max_feature - min_feature)
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'delay': 
            min_feature = 3.0  # (all negative values are invalid)
            max_feature = 11.0
            feature = (feature - min_feature) / (max_feature - min_feature)
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'longitude':
            feature = feature / 180  # -180 to 180
            return torch.clamp(feature, -1, 1)
        elif feature_type == 'latitude':
            feature = feature / 90  # -90 to 90
            return torch.clamp(feature, -1, 1)
        else:
            raise ValueError(f"Invalid feature type: {feature_type}")
  
    def compute_conf(self, sat_node_number, interval, num1, num2, ID, Q,
                     num_backbone, matrix):
        Q.append(
            "log \"/var/log/bird.log\" { debug, trace, info, remote, warning, error, auth, fatal, bug };"
        )
        Q.append("debug protocols all;")
        Q.append("protocol device {")
        Q.append("}")
        Q.append(" protocol direct {")
        Q.append("    disabled;		# Disable by default")
        Q.append("    ipv4;			# Connect to default IPv4 table")
        Q.append("    ipv6;			# ... and to default IPv6 table")
        Q.append("}")
        Q.append("protocol kernel {")
        Q.append("    ipv4 {			# Connect protocol to IPv4 table by channel")
        Q.append(
            "        export all;	# Export to protocol. default is export none")
        Q.append("    };")
        Q.append("}")
        # Q.append("protocol kernel {")
        # Q.append("    ipv6 { export all; ")
        # Q.append("    };")
        # Q.append("}")
        Q.append("protocol static {")
        Q.append("    ipv4;			# Again, IPv6 channel with default options")
        Q.append("}")
        Q.append("protocol ospf{")
        Q.append("    ipv4 {")
        Q.append("        import all;")
        Q.append("    };")
        Q.append("    area 0 {")
        Q.append("    interface \"B%d-eth0\" {" % ID)
        Q.append("        type broadcast;		# Detected by default")
        Q.append("        cost 256;")
        Q.append("        hello " + str(interval) +
                 ";			# Default hello perid 10 is too long")
        Q.append("    };")
        Q.append("    interface \"inter_machine\" {")
        Q.append("        type broadcast;		# Detected by default")
        Q.append("        cost 256;")
        Q.append("        hello " + str(interval) +
                 ";			# Default hello perid 10 is too long")
        Q.append("    };")
        if num1 <= sat_node_number and num2 <= num_backbone and ID <= sat_node_number:  # satellite
            for peer in range(num1, num2 + 1):
                if (peer == ID) or (int(float(matrix[ID - 1][peer - 1])) == 0):
                    continue
                Q.append("    interface \"B%d-eth%d\" {" % (ID, peer))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
            if num2 > sat_node_number:
                for i in range(sat_node_number + 1,
                               num_backbone + 1):  # each ground station
                    Q.append("    interface \"B%d-eth%d\" {" % (ID, i))
                    Q.append("        type broadcast;		# Detected by default")
                    Q.append("        cost 256;")
                    Q.append("        hello " + str(interval) +
                             ";			# Default hello perid 10 is too long")
                    Q.append("    };")
        elif num1 <= sat_node_number and num2 <= num_backbone and ID > sat_node_number:  # ground station
            for peer in range(1,
                              1 + sat_node_number):  # fac and each satellite
                Q.append("    interface \"B%d-eth%d\" {" % (ID, peer))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
            Q.append("    interface \"B%d-default\" {" % (ID))
            Q.append("        type broadcast;		# Detected by default")
            Q.append("        cost 256;")
            Q.append("        hello " + str(interval) +
                     ";			# Default hello perid 10 is too long")
            Q.append("    };")
        elif num1 > num_backbone and num2 > num_backbone:  # ground users
            if ID != num1 and ID != num2:
                Q.append("    interface \"B%d-eth%d\" {" % (ID, ID - 1))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
                Q.append("    interface \"B%d-eth%d\" {" % (ID, ID + 1))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
            elif ID == num1:
                Q.append("    interface \"B%d-eth%d\" {" % (ID, ID + 1))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
            elif ID == num2:
                Q.append("    interface \"B%d-eth%d\" {" % (ID, ID - 1))
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
        else:
            return False
        Q.append("    };")
        Q.append(" }")
        return True

    def print_conf(self, sat_node_number, fac_node_number, ID, Q, remote_ftp):
        filename = self.file_path + "/conf/bird-" + \
            str(sat_node_number) + "-" + str(fac_node_number) + "/B%d.conf" % ID
        fout = open(self.configuration_file_path + "/" + filename, 'w+')
        for item in Q:
            fout.write(str(item) + "\n")
        fout.close()
        remote_ftp.put(self.configuration_file_path + "/" + filename, filename)

    def generate_conf(self, remote_ssh, remote_ftp):
        if self.intra_routing != "OSPF" and self.intra_routing != "ospf":
            return False
        if os.path.exists(self.configuration_file_path + "/" + self.file_path +
                          "/conf/bird-" +
                          str(self.orbit_number * self.sat_number) + "-" +
                          str(len(self.GS_lat_long))) == True:
            osstr = "rm -f " + self.configuration_file_path+"/"+self.file_path+"/conf/bird-" + \
                str(self.orbit_number*self.sat_number) + "-" + str(len(self.GS_lat_long)) + "/*"
            os.system(osstr)
            sn_remote_cmd(remote_ssh, "mkdir ~/" + self.file_path + "/conf")
            sn_remote_cmd(
                remote_ssh, "mkdir ~/" + self.file_path + "/conf/bird-" +
                str(self.orbit_number * self.sat_number) + "-" +
                str(len(self.GS_lat_long)))
        else:
            os.makedirs(self.configuration_file_path + "/" + self.file_path +
                        "/conf/bird-" +
                        str(self.orbit_number * self.sat_number) + "-" +
                        str(len(self.GS_lat_long)))
            sn_remote_cmd(remote_ssh, "mkdir ~/" + self.file_path + "/conf")
            sn_remote_cmd(
                remote_ssh, "mkdir ~/" + self.file_path + "/conf/bird-" +
                str(self.orbit_number * self.sat_number) + "-" +
                str(len(self.GS_lat_long)))
        path = self.configuration_file_path + "/" + self.file_path + "/delay/1.txt"
        matrix = sn_get_param(path)
        num_backbone = self.orbit_number * self.sat_number + len(
            self.GS_lat_long)
        error = True
        for i in range(len(self.AS)):
            if len(self.AS[i]) != 1:
                for ID in range(self.AS[i][0], self.AS[i][1] + 1):
                    Q = []
                    error = self.compute_conf(
                        self.orbit_number * self.sat_number,
                        self.hello_interval, self.AS[i][0], self.AS[i][1], ID,
                        Q, num_backbone, matrix)
                    self.print_conf(self.orbit_number * self.sat_number,
                                    len(self.GS_lat_long), ID, Q, remote_ftp)
            else:  # one node in one AS
                ID = self.AS[i][0]
                Q = []
                Q.append(
                    "log \"/var/log/bird.log\" { debug, trace, info, remote, warning, error, auth, fatal, bug };"
                )
                Q.append("debug protocols all;")
                Q.append("protocol device {")
                Q.append("}")
                Q.append(" protocol direct {")
                Q.append("    disabled;		# Disable by default")
                Q.append("    ipv4;			# Connect to default IPv4 table")
                Q.append("    ipv6;			# ... and to default IPv6 table")
                Q.append("}")
                Q.append("protocol kernel {")
                Q.append(
                    "    ipv4 {			# Connect protocol to IPv4 table by channel")
                Q.append(
                    "        export all;	# Export to protocol. default is export none"
                )
                Q.append("    };")
                Q.append("}")
                Q.append("protocol static {")
                Q.append(
                    "    ipv4;			# Again, IPv6 channel with default options")
                Q.append("}")
                Q.append("protocol ospf {")
                Q.append("    ipv4 {")
                Q.append("        import all;")
                Q.append("    };")
                Q.append("    area 0 {")
                Q.append("    interface \"B%d-eth0\" {" % ID)
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(self.hello_interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
                Q.append("    interface \"inter_machine\" {")
                Q.append("        type broadcast;		# Detected by default")
                Q.append("        cost 256;")
                Q.append("        hello " + str(self.hello_interval) +
                         ";			# Default hello perid 10 is too long")
                Q.append("    };")
                Q.append("    };")
                Q.append(" }")
                self.print_conf(self.orbit_number * self.sat_number,
                                len(self.GS_lat_long), ID, Q, remote_ftp)

        return error


def add_noise_to_sgp4_params(
    # Base values for the satellite/orbit
    base_satellite_altitude_km: float,
    base_mean_anomaly_deg: float,
    base_raan_rad: float, # Nominal RAAN for the current plane
    # Perturbation magnitudes/settings
    altitude_min_km: float = 550.0,
    altitude_perturbation_range_km: float = 5.0,
    mean_anomaly_perturbation_range_deg: float = 2.0,
    raan_perturbation_range_deg: float = 5.0,
    ):
    """
    Perturbs SGP4 orbital elements for domain randomization.

    Args:
        base_satellite_altitude_km: Nominal altitude for mean motion calculation.
        base_raan_rad: Nominal Right Ascension of Ascending Node in radians for the plane.
        base_mean_anomaly_deg: Nominal Mean Anomaly in degrees for the satellite.
        altitude_perturbation_range_km: Max +/- random value added to altitude.
        raan_perturbation_range_deg: Max +/- random value added to RAAN.
        mean_anomaly_perturbation_range_deg: Max +/- random value added to Mean Anomaly.

    Returns:
        A dictionary containing the perturbed SGP4 elements:
        'altitude_km', 'raan_rad', 'mean_anomaly_deg'
    """
    perturbed_params = {}
    perturbed_params['altitude_km'] = base_satellite_altitude_km + np.random.uniform(-altitude_perturbation_range_km, altitude_perturbation_range_km)
    perturbed_params['altitude_km'] = max(perturbed_params['altitude_km'], altitude_min_km)
    raan_perturbation_range_rad = np.radians(np.random.uniform(-raan_perturbation_range_deg, raan_perturbation_range_deg))
    perturbed_params['raan_rad'] = (base_raan_rad + raan_perturbation_range_rad) % (2 * np.pi)
    ma_perturb = np.random.uniform(-mean_anomaly_perturbation_range_deg, mean_anomaly_perturbation_range_deg)
    perturbed_mean_anomaly_deg = base_mean_anomaly_deg + ma_perturb
    # Wrap Mean Anomaly to [0, 360) degrees
    perturbed_params['mean_anomaly_deg'] = perturbed_mean_anomaly_deg % 360.0
    if perturbed_params['mean_anomaly_deg'] < 0: # Ensure positive if modulo gives negative
        perturbed_params['mean_anomaly_deg'] += 360.0
    return perturbed_params


def add_maneuver_perturbations(
    base_satellite_altitude_km: float,
    base_mean_anomaly_deg: float,
    base_raan_rad: float,
    maneuver_type: str = 'random',  # e.g., 'altitude_change', 'plane_change', 'phasing'
    maneuver_magnitude: float = 20.0,  # magnitude of the maneuver
    raan_maneuver_magnitude: float = 10.0,
    mean_anomaly_maneuver_magnitude: float = 2.0,
    min_altitude_km: float = 550.0,
    maneuver_progress: float = 0.5,
):
    """
    Perturbs orbital elements to simulate specific types of maneuvers.
    """
    perturbed_params = {}

    # Select random maneuver type if not specified
    if maneuver_type == 'random':
        maneuver_type = np.random.choice(['altitude_change', 'plane_change', 'phasing'])

    # Scale the maneuver magnitude based on the maneuver progress
    maneuver_magnitude = maneuver_magnitude * maneuver_progress
    raan_maneuver_magnitude = raan_maneuver_magnitude * maneuver_progress
    mean_anomaly_maneuver_magnitude = mean_anomaly_maneuver_magnitude * maneuver_progress

    if maneuver_type == 'altitude_change':
        # Larger altitude changes for orbit raising/lowering
        perturbed_params['altitude_km'] = base_satellite_altitude_km + maneuver_magnitude
        perturbed_params['altitude_km'] = max(perturbed_params['altitude_km'], min_altitude_km)
        # Smaller changes to other parameters
        perturbed_params['mean_anomaly_deg'] = base_mean_anomaly_deg + np.random.uniform(-1, 1)
        perturbed_params['raan_rad'] = base_raan_rad + np.radians(np.random.uniform(-0.5, 0.5))
    elif maneuver_type == 'plane_change':
        # Larger RAAN changes for plane changes
        perturbed_params['raan_rad'] = base_raan_rad + np.radians(raan_maneuver_magnitude)
        # Smaller changes to other parameters
        perturbed_params['altitude_km'] = base_satellite_altitude_km + np.random.uniform(-2, 2)
        perturbed_params['mean_anomaly_deg'] = base_mean_anomaly_deg + np.random.uniform(-1, 1)
    elif maneuver_type == 'phasing':
        # Larger mean anomaly changes for phasing maneuvers
        perturbed_params['mean_anomaly_deg'] = base_mean_anomaly_deg + mean_anomaly_maneuver_magnitude
        # Smaller changes to other parameters
        perturbed_params['altitude_km'] = base_satellite_altitude_km + np.random.uniform(-2, 2)
        perturbed_params['raan_rad'] = base_raan_rad + np.radians(np.random.uniform(-0.5, 0.5))
    return perturbed_params
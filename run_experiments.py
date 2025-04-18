import json
import argparse
import copy

from starrynet.sn_observer import *
from starrynet.sn_orchestrater import *
from starrynet.sn_synchronizer import *

import subprocess
import time

import numpy as np

import pandas as pd
import shutil
import os
from datetime import datetime
import re

# Define the base configuration using parameters from your example
# Note: Adjusted Altitude to 550km as a more typical LEO baseline,
# Stable case will override this. Loss/BW kept as 1%/1Gbps default.
BASE_CONFIG = {
  "Name": "starlink",
  "Altitude (km)": 550,       # Typical LEO altitude
  "Cycle (s)": 5731,          # Keep constant unless specifically testing this
  "Inclination": 53,          # Keep constant unless specifically testing this
  "Phase shift": 1,           # Keep constant unless specifically testing this
  "# of orbit": 10,           # Default number of orbits
  "# of satellites": 10,        # Default satellites per orbit (100 total)
  "Duration (s)": 125,        # Default to 2 emulation minutes (~300s)
  "update_time (s)": 1,
  "satellite link bandwidth (\"X\" Gbps)": 1, # Default bandwidth
  "sat-ground bandwidth (\"X\" Gbps)": 1,    # Default bandwidth
  "satellite link loss (\"X\"% )": 1,       # Default loss
  "sat-ground loss (\"X\"% )": 1,          # Default loss
  "GS number": 2,        # Default number of ground stations -- one for receiver, one for sender
  "antenna number": 1,
  "antenna_inclination_angle": 25, # Default elevation angle
  "remote_machine_IP": "127.0.0.1", # Placeholder
  "remote_machine_username": "agot", # REPLACE ME
  "remote_machine_password": "1234", # REPLACE ME
  "Satellite link": "grid",
  "IP version": "IPv4",
  "Intra-AS routing": "OSPF",
  "Inter-AS routing": "BGP",     # Keep constant unless testing routing protocols
  "Link policy": "LeastDelay", # Keep constant unless testing link policies
  "Handover policy": "instant handover",
  "multi-machine (\"0\" for no, \"1\" for yes)": 0
}

# Define Scenario IDs (adjust as needed)
SCENARIOS = {
    1: "Stable_LEO_Best_Case_Control",
    2: "Typical_Churn_LEO_Exp_1_Base", 
    3: "Maneuver_Scenario_Density_Exp_2_Base",
    4: "Partial_Deployment_Sparse_LEO_Exp_3_Base",
    5: "Extreme_GSL_Churn",
    6: "Extreme_ISL_Churn", 
    7: "Extreme_Loss",
    8: "Extreme_Maneuvers",
    9: "Random_Persistent_Disruptions"
}

def set_call_duration(n_nodes):
    # NOTE: we adjust call runtime based on scenario due to load on node
    # TODO: we need to test each scenario to ensure calls finish before emulation ends
    # 50 containers ~ 3 minutes to ???
    # 80 containers ~ 4 minutes to ???
    # 100 containers ~ 5 minutes to ???
    # 150 containers ~ 7 minutes to ???
    if n_nodes <= 50:
        call_duration = 180
    elif n_nodes <= 80:
        call_duration = 240
    elif n_nodes <= 100:
        call_duration = 300
    else:
        call_duration = 420
    with open('/opt/home_dir/AlphaRTC/configs/sender.json', 'r') as f:
        sender_config = json.load(f)
    with open('/opt/home_dir/AlphaRTC/configs/receiver.json', 'r') as f:
        receiver_config = json.load(f)
    sender_config['serverless_connection']['autoclose'] = call_duration
    receiver_config['serverless_connection']['autoclose'] = call_duration
    # save configs
    with open('/opt/home_dir/AlphaRTC/configs/sender.json', 'w') as f:
        json.dump(sender_config, f, indent=2)
    with open('/opt/home_dir/AlphaRTC/configs/receiver.json', 'w') as f:
        json.dump(receiver_config, f, indent=2)
    print(f"Call duration set to {call_duration} seconds")


def generate_config(scenario_id):
    """
    Generates a specific configuration dictionary based on the scenario ID.

    Args:
        scenario_id (int): The ID of the scenario to generate.

    Returns:
        dict: The configuration dictionary for the specified scenario,
              or None if the ID is invalid.
    """
    if scenario_id not in SCENARIOS:
        print(f"Error: Invalid scenario ID {scenario_id}")
        print("Available scenarios:")
        for id, name in SCENARIOS.items():
            print(f"  {id}: {name}")
        return None

    # Start with a deep copy of the base config
    config = copy.deepcopy(BASE_CONFIG)
    scenario_name = SCENARIOS[scenario_id]

    print(f"Generating config for Scenario {scenario_id}: {scenario_name}")
    
    # --- Control Cases ---
    if scenario_id == 1: # Stable LEO (Best-Case Control)
        config["Altitude (km)"] = 1200 # Higher altitude reduces GSL churn
        config["antenna_inclination_angle"] = 15 # Lower angle reduces GSL churn
        config["# of orbit"] = 15
        config["# of satellites"] = 10 # Max density within 150 limit (15x10=150)
        config["satellite link loss (\"X\"% )"] = 0.01 # Minimal loss
        config["sat-ground loss (\"X\"% )"] = 0.01    # Minimal loss
        config["satellite link bandwidth (\"X\" Gbps)"] = 10 # High BW
        config["sat-ground bandwidth (\"X\" Gbps)"] = 10    # High BW

    # --- Non-Extreme LEO Tests (Based on 10 fixed orbits) ---
    elif scenario_id == 2: # Typical Churn LEO (Exp 1 Base)
        config["Altitude (km)"] = 550 # Standard LEO altitude
        config["antenna_inclination_angle"] = 25 # Standard angle
        config["# of orbit"] = 10
        config["# of satellites"] = 10 # Moderate density (100 total)
        config["satellite link loss (\"X\"% )"] = 1 # Standard loss
        config["sat-ground loss (\"X\"% )"] = 1    # Standard loss
        config["satellite link bandwidth (\"X\" Gbps)"] = 1 # Standard BW
        config["sat-ground bandwidth (\"X\" Gbps)"] = 1    # Standard BW

    elif scenario_id == 3: # Maneuver Scenario Density (Exp 2 Base)
        # Note: Actual maneuvers require external simulation/injection.
        # This config sets the density context (~80 sats).
        config["Altitude (km)"] = 550
        config["antenna_inclination_angle"] = 25
        config["# of orbit"] = 10
        config["# of satellites"] = 8 # Lower density (80 total)
        config["satellite link loss (\"X\"% )"] = 1
        config["sat-ground loss (\"X\"% )"] = 1
        config["satellite link bandwidth (\"X\" Gbps)"] = 1
        config["sat-ground bandwidth (\"X\" Gbps)"] = 1

    elif scenario_id == 4: # Partial Deployment/Sparse LEO (Exp 3 Base)
        config["Altitude (km)"] = 550
        config["antenna_inclination_angle"] = 25  # NOTE: for extreme scenarios, this should be 45 to increase churn
        config["# of orbit"] = 10
        config["# of satellites"] = 5 # Sparse density (50 total) - high ISL churn expected, NOTE: for extreme scenarios, this should 25-30
        config["satellite link loss (\"X\"% )"] = 1  # NOTE: for extreme scenarios, this should be 2-5%
        config["sat-ground loss (\"X\"% )"] = 1
        config["satellite link bandwidth (\"X\" Gbps)"] = 1
        config["sat-ground bandwidth (\"X\" Gbps)"] = 1

    # --- Extreme LEO Tests ---
    elif scenario_id == 5: # Extreme GSL Churn
        config["Altitude (km)"] = 400 # Lowest plausible altitude
        config["antenna_inclination_angle"] = 45 # Highest plausible angle
        config["# of orbit"] = 10
        config["# of satellites"] = 5 # Sparse density (50 total)
        config["satellite link loss (\"X\"% )"] = 0.01 # Low loss to isolate churn
        config["sat-ground loss (\"X\"% )"] = 0.01    # Low loss to isolate churn
        config["satellite link bandwidth (\"X\" Gbps)"] = 10 # High BW
        config["sat-ground bandwidth (\"X\" Gbps)"] = 10    # High BW

    elif scenario_id == 6: # Extreme ISL Churn
        config["Altitude (km)"] = 550 # Standard altitude
        config["antenna_inclination_angle"] = 25 # Standard angle
        config["# of orbit"] = 5  # Fewest orbits
        config["# of satellites"] = 10 # Few sats per orbit (50 total) - Max sparsity
        config["satellite link loss (\"X\"% )"] = 0.01 # Low loss to isolate churn
        config["sat-ground loss (\"X\"% )"] = 0.01    # Low loss to isolate churn
        config["satellite link bandwidth (\"X\" Gbps)"] = 10 # High BW
        config["sat-ground bandwidth (\"X\" Gbps)"] = 10    # High BW

    elif scenario_id == 7: # Extreme Loss
        config["Altitude (km)"] = 550 # Use a moderately stable base
        config["antenna_inclination_angle"] = 25
        config["# of orbit"] = 10
        config["# of satellites"] = 10 # Moderate density (100 total)
        config["satellite link loss (\"X\"% )"] = 10 # High loss
        config["sat-ground loss (\"X\"% )"] = 10    # High loss
        config["satellite link bandwidth (\"X\" Gbps)"] = 1 # Standard BW
        config["sat-ground bandwidth (\"X\" Gbps)"] = 1    # Standard BW

    elif scenario_id == 8 or scenario_id == 9: # Extreme Disruption Base Config
        # Note: Actual disruptions require external injection (e.g., set_damage).
        # This config provides a stable base network for the disruption test.
        # Using the 'Stable LEO' config parameters as the base.
        config["Altitude (km)"] = 1200
        config["antenna_inclination_angle"] = 15
        config["# of orbit"] = 15
        config["# of satellites"] = 10 # Max density (150 total)
        config["satellite link loss (\"X\"% )"] = 0.01 # Low loss
        config["sat-ground loss (\"X\"% )"] = 0.01    # Low loss
        config["satellite link bandwidth (\"X\" Gbps)"] = 1 # Standard BW (easier to see disruption impact)
        config["sat-ground bandwidth (\"X\" Gbps)"] = 1    # Standard BW

    # Ensure total satellites doesn't exceed 150
    total_sats = config["# of orbit"] * config["# of satellites"]
    if total_sats > 150:
        print(f"Warning: Scenario {scenario_id} configuration resulted in {total_sats} satellites, exceeding the 150 limit. Adjust parameters.")
    set_call_duration(total_sats)
    return config

def run_experiment(args):
    # get the number of nodes
    with open(args.outfile, 'r') as f:
        config = json.load(f)
        n_orbits = config['# of orbit']
        n_satellites = config['# of satellites']
        n_ground_stations = config['GS number']
        n_nodes = (n_orbits * n_satellites) + n_ground_stations

    # put all nodes in the same AS
    AS = [[1, n_nodes]]
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
    hello_interval = 10  # hello_interval(s) in OSPF. 1-200 are supported.

    print('Initializing StarryNet...')
    sn = StarryNet(args.outfile, GS_lat_long, hello_interval, AS)
    sn.stop_emulation() # stop emulation before creating nodes
    sn.create_nodes()
    sn.create_links()
    sn.run_routing_deamon()

    print('Creating RTC nodes...')
    sn.create_rtc_nodes()

    # Set damage parameters based on experiment scenario
    if args.exp == 3:  # Maneuver Scenario Density scenario
        ratio = np.random.uniform(0.05, 0.1)   # 5%-10% network damage
        step = np.random.randint(10, 15)  # NOTE: for extreme maneuvers, step is 5-10s
        time_steps = []
        for i in range(10, 115, step):
            sn.set_damage(ratio, i)
            time_steps.append(i)
        recovery_time = np.random.randint(5, 10)  # NOTE: for extreme maneuvers, recovery time is 30-60s
        for time_step in time_steps:
            sn.set_recovery(time_step + recovery_time)
    elif args.exp == 7:  # Extreme Loss scenario
        ratio = np.random.uniform(0.05, 0.1)   # 5%-10% network damage
        time_index = np.random.randint(10, 115)  # Damage occurs 10-115s into simulation
        print(f'Applying {ratio*100}% network damage at t={time_index}s')
        sn.set_damage(ratio, time_index)
    elif args.exp == 8:  # Extreme Disruption scenario
        ratio = np.random.uniform(0.1, 0.5)    # 10%-50% network damage
        time_index = np.random.randint(10, 115)  # Damage occurs 10-115s into simulation
        print(f'Applying {ratio*100}% network damage at t={time_index}s')
        sn.set_damage(ratio, time_index)
    elif args.exp == 9:  # Random Persistent Disruptions scenario
        ratio = np.random.uniform(0.0, 0.1)   # 0%-1% network damage
        time_index = 10
        while time_index < 115:
            sn.set_damage(ratio, time_index)
            time_index += np.random.randint(1, 10)  # apply damage every 1-10s
    # Add other experiment-specific damage scenarios as needed

    sn.set_video_call(26, 27, 5)  # start video call. NOTE: these indices are not used as sender and receiver are already set.

    start_time = time.time()
    sn.start_emulation()
    end_time = time.time()
    print(f"Emulation time: {end_time - start_time} seconds")
    sn.stop_emulation()

    # parse output logs
    print("Parsing output logs...")
    results = parse_output_logs()
    # save results
    if not os.path.exists('/mydata/gcc_baselines/'):
        os.makedirs('/mydata/gcc_baselines/')
    # save results to output directory
    with open(f'/mydata/gcc_baselines/{args.experiment_id}.json', 'w') as f:
        json.dump(results, f, indent=2)
    # clean output directory
    print("Cleaning up /tmp and /outputs directory...")
    os.system('sudo rm -rf /opt/home_dir/outputs/*')
    os.system('sudo rm -rf /tmp/*')
    print("Cleanup complete.")

def parse_output_logs():
    output_logs = ['/opt/home_dir/outputs/receiver.log', '/opt/home_dir/outputs/sender.log']
    all_metrics = {}
    for log in output_logs:
        metrics = _parse_log(log)
        all_metrics[log] = metrics
    return all_metrics

def _parse_log(log):
    with open(log, 'r') as f:
        log_content = f.read()
    return _parse_log_webrtc_only(log_content)

def _parse_log_webrtc_only(log_content):
    """
    Parses log content to extract metrics starting with "WebRTC.",
    handling both simple values and periodic samples (min, avg, max),
    including those with a comma before 'periodic_samples'.

    Args:
        log_content: A string containing the log data.

    Returns:
        A dictionary where keys are metric names (e.g., "WebRTC.Video.InputWidthInPixels",
        "WebRTC.Video.InputFramesPerSecond.min") and values are floats.
    """
    metrics = {}
    # Regex specifically for periodic samples: captures name, samples, min, avg, max
    # FIXED: Added optional comma (,?) before potential whitespace (\s*)
    #        preceding 'periodic_samples:'
    periodic_pattern = re.compile(
        r"(WebRTC\.[\w\.]+),?\s*" # Capture name (Group 1), then optional comma, then whitespace
        r"periodic_samples:(\d+),\s*" # Capture sample count (Group 2)
        r"\{min:([\d\.-]+),\s*"       # Capture min value (Group 3)
        r"avg:([\d\.-]+),\s*"       # Capture avg value (Group 4)
        r"max:([\d\.-]+)\}"         # Capture max value (Group 5)
    )

    lines = log_content.splitlines() # Split content into lines

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # --- Only process lines containing "WebRTC." ---
        if "WebRTC." in line:
            # Attempt to match the detailed periodic pattern first
            match = periodic_pattern.search(line)
            if match:
                try:
                    metric_name = match.group(1)
                    # samples = int(match.group(2)) # Uncomment if you want to store sample count
                    min_val = float(match.group(3))
                    avg_val = float(match.group(4))
                    max_val = float(match.group(5))

                    # Store min, avg, max as separate metrics
                    metrics[metric_name + ".min"] = min_val
                    metrics[metric_name + ".avg"] = avg_val
                    metrics[metric_name + ".max"] = max_val
                    # Optionally add samples: metrics[metric_name + ".samples"] = samples

                except (ValueError, IndexError):
                     # print(f"Warning: Could not parse periodic metric values in line: {line}")
                     pass # Silently ignore parsing errors for this line
                # Successfully parsed or failed, move to the next line
                continue

            # If not periodic, try parsing as a simple WebRTC key-value metric
            else:
                try:
                    # Split based on the first occurrence of "WebRTC."
                    base_parts = line.split("WebRTC.", 1)
                    if len(base_parts) == 2:
                        metric_parts = base_parts[1].split()
                        if len(metric_parts) >= 2:
                            metric_name = "WebRTC." + metric_parts[0]
                            value_str = metric_parts[-1]
                            # Basic check if the last part looks like a number
                            if value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                                value = float(value_str)
                                metrics[metric_name] = value
                            # else: Line contained WebRTC. but didn't end like a simple metric
                except (ValueError, IndexError):
                     # print(f"Warning: Could not parse simple WebRTC metric line: {line}")
                     pass # Silently ignore parsing errors for this line
                # Continue to the next line
                continue

    return metrics

def main():
    """
    Parses command line arguments and prints the generated config.
    """
    parser = argparse.ArgumentParser(description="Generate LEO network simulation configurations.")
    parser.add_argument("--exp", type=int, required=True, choices=SCENARIOS.keys(),
                        help=f"Experiment scenario ID ({', '.join(map(str, SCENARIOS.keys()))})")
    parser.add_argument("--outfile", type=str, default="/opt/home_dir/StarryNet/config.json",
                        help="File path to save the JSON config.")

    args = parser.parse_args()

    generated_config = generate_config(args.exp)

    if generated_config:
        if args.outfile:
            try:
                with open(args.outfile, 'w') as f:
                    json.dump(generated_config, f, indent=2)
                print(f"Configuration saved to {args.outfile}")
            except IOError as e:
                print(f"Error writing to file {args.outfile}: {e}")
                print("\nGenerated Configuration JSON:")
                print(json.dumps(generated_config, indent=2)) # Print to stdout if file write fails
        else:
            # Print the generated config to standard output as JSON
            print("\nGenerated Configuration JSON:")
            print(json.dumps(generated_config, indent=2))
    else:
        print("Error: No configuration generated. Exiting.")
        exit(1)

    # unique id for the experiment
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_id = experiment_id + "_" + SCENARIOS[args.exp]
    # run experiment and save results
    run_experiment(args)

if __name__ == "__main__":
    main()

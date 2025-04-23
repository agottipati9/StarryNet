#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np
import sys
import subprocess
import re

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.returncode
    return out, err, return_code

class NetInfo(object):
    def __init__(self, net_path):
        self.net_path = net_path
        self.net_data = None

        # TODO: we need to parse the state info we just added to logs
        self.net_states = None

        self.parse_net_log()
        

    def parse_net_log(self):
        if not self.net_path or not os.path.exists(self.net_path):
            raise ValueError("Error net path")

        json_data = []
        current_packet = None
        
        with open(self.net_path, 'r') as f:
            for line in f.readlines():
                net_states = {}
                if "remote_estimator_proxy.cc" not in line:
                    continue
                
                # Extract the line number and content
                match = re.search(r'\(remote_estimator_proxy\.cc:(\d+)\):\s*(.*)', line)
                if not match:
                    continue
                
                line_num = match.group(1)
                content = match.group(2).strip()
                
                # Try to parse as JSON (packet info)
                if content.startswith('{'):
                    try:
                        json_obj = json.loads(content)
                        # We'll preserve mediaInfo but mark it as parsed
                        current_packet = json_obj
                        json_data.append(json_obj)
                    except ValueError:
                        pass
                    except Exception as e:
                        raise ValueError(f"Exception when parsing JSON log: {str(e)}")
                
                # Parse array metrics
                elif ":" in content and "[" in content and "]" in content:
                    try:
                        metric_name, values_str = content.split(':', 1)
                        metric_name = metric_name.strip()
                        
                        # Extract values from array format [x, y, z]
                        values_str = values_str.strip()
                        if values_str.startswith('[') and values_str.endswith(']'):
                            values_str = values_str[1:-1].strip()
                            # Convert values to appropriate type (float or int)
                            values = []
                            for val in values_str.split(','):
                                val = val.strip()
                                if not val:
                                    continue
                                try:
                                    # Try to convert to float first
                                    if '.' in val:
                                        values.append(float(val))
                                    else:
                                        values.append(int(val))
                                except ValueError:
                                    # Keep as string if conversion fails
                                    values.append(val)
                            
                            # Store in net_states
                            net_states[metric_name] = values

                            # Also attach to the current packet if it exists
                            if current_packet is not None:
                                if "state_metrics" not in current_packet:
                                    current_packet["state_metrics"] = {}
                                current_packet["state_metrics"][metric_name] = values
                    except Exception as e:
                        print(f"Warning: Failed to parse metric line: {content}, error: {str(e)}")
        self.net_data = json_data


def eval_network(dst_audio_info: NetInfo):
    net_data = dst_audio_info.net_data
    ssrc_info = {}

    delay_list = []
    loss_count = 0
    last_seqNo = {}
    for item in net_data:
        ssrc = item["packetInfo"]["header"]["ssrc"]
        sequence_number = item["packetInfo"]["header"]["sequenceNumber"]
        tmp_delay = item["packetInfo"]["arrivalTimeMs"] - \
            item["packetInfo"]["header"]["sendTimestamp"]
        if (ssrc not in ssrc_info):
            ssrc_info[ssrc] = {
                "time_delta": -tmp_delay,
                "delay_list": [],
                "received_nbytes": 0,
                "start_recv_time": item["packetInfo"]["arrivalTimeMs"],
                "avg_recv_rate": 0
            }
        if ssrc in last_seqNo:
            loss_count += max(0, sequence_number -
                              last_seqNo[ssrc] - 1)
        last_seqNo[ssrc] = sequence_number

        ssrc_info[ssrc]["delay_list"].append(
            ssrc_info[ssrc]["time_delta"] + tmp_delay)
        ssrc_info[ssrc]["received_nbytes"] += item["packetInfo"]["payloadSize"]
        if item["packetInfo"]["arrivalTimeMs"] != ssrc_info[ssrc]["start_recv_time"]:
            ssrc_info[ssrc]["avg_recv_rate"] = ssrc_info[ssrc]["received_nbytes"] / \
                (item["packetInfo"]["arrivalTimeMs"] -
                    ssrc_info[ssrc]["start_recv_time"])

    # filter short stream
    ssrc_info = {key: val for key,
                 val in ssrc_info.items() if len(val["delay_list"]) >= 10}

    # scale delay list
    # for ssrc in ssrc_info:
    #     min_delay = min(ssrc_info[ssrc]["delay_list"])
    #     # ssrc_info[ssrc]["scale_delay_list"] = [
    #     #     min(max_delay, delay) for delay in ssrc_info[ssrc]["delay_list"]]
    #     delay_pencentile_95 = np.percentile(
    #         ssrc_info[ssrc]["scale_delay_list"], 95)
    #     ssrc_info[ssrc]["delay_score"] = (
    #         max_delay - delay_pencentile_95) / (max_delay - min_delay)
    #     print("Queue Delay_min (ms): {}".format(min_delay))
    #     print("Queue Delay_95 (ms): {}".format(delay_pencentile_95))
    #     print("Queue Delay_max (ms): {}".format(max_delay))
    #     print("Receive Rate (KB/s): {}".format(
    #         ssrc_info[ssrc]["avg_recv_rate"]))
    #     print("Ground Truth Receive Rate (KB/s): {}".format(ground_recv_rate))
    #     print("Loss Count: {}".format(loss_count))

    # avg delay
    avg_delay = np.mean(
        [np.mean(ssrc_info[ssrc]["delay_list"]) for ssrc in ssrc_info])

    # receive rate score
    recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"]
                      for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]
    # avg recv rate
    avg_recv_rate = np.mean(recv_rate_list)

    print("Avg Delay (ms): {}".format(avg_delay))
    print("Avg Receive Rate (KB/s): {}".format(avg_recv_rate))
    print("Loss Count: {}".format(loss_count))
    return net_data


def get_network_data(args):
    print("----- Network Statistics -----")
    dst_network_info = NetInfo(args.receiver_log)
    dst_net_data = eval_network(dst_network_info)
    src_network_info = NetInfo(args.sender_log)
    src_net_data = eval_network(src_network_info)
    print("")
    return dst_net_data, src_net_data


def init_network_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="path to output artifacts.")
    parser.add_argument("--sender_log", type=str,
                        default=None, help="the path of sender log.")
    parser.add_argument("--receiver_log", type=str,
                        default=None, help="the path of receiver log.")
    return parser


if __name__ == "__main__":
    parser = init_network_argparse()
    args = parser.parse_args()
    if args.sender_log is None:
        args.sender_log = os.path.join(args.output_dir, "sender.log")
    if args.receiver_log is None:
        args.receiver_log = os.path.join(args.output_dir, "receiver.log")

    out_dict = {}
    receiver_packet_info, sender_packet_info = get_network_data(args)
    out_dict["receiver_packet_info"] = receiver_packet_info
    out_dict["sender_packet_info"] = sender_packet_info

    out_path = os.path.join(args.output_dir, "call_metrics.json")
    with open(out_path, 'w') as f:
        f.write(json.dumps(out_dict))
    

    print("Processed call logs.")
    print("Output written to", out_path)
    print("")
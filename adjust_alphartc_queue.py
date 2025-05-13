import os
import argparse

target_code_path = "/opt/home_dir/AlphaRTC/modules/pacing/paced_sender.cc"
local_copy_path = "/opt/home_dir/StarryNet/paced_sender.cc"

# restore original paced_sender.cc
os.system(f"cp {local_copy_path} {target_code_path}")

# modify AlphaRTC queue size in paced_sender.cc
parser = argparse.ArgumentParser()
parser.add_argument("--queue_size", type=int, default=2000)
args = parser.parse_args()
with open(local_copy_path, "r") as f:
    lines = f.readlines()
lines[191] = f"pacing_controller_.SetQueueTimeLimit(TimeDelta::Millis({args.queue_size}));\n"

# save modified paced_sender.cc
with open(target_code_path, "w") as f:
    f.writelines(lines)

# compile code
os.system(f"cd /opt/home_dir/AlphaRTC/scripts && ./compile.sh")

# write queue size to log file
with open(f"/opt/home_dir/outputs/queue_size.txt", "w") as f:
    f.write(f"{args.queue_size}")

print(f"Queue size set to {args.queue_size}")












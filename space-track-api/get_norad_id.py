def get_norad_ids(tle_file):
    norad_ids = {}
    
    with open(tle_file, 'r') as f:
        lines = f.readlines()
        
    # Process 3 lines at a time (name + two TLE lines)
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        sat_name = lines[i].strip()
        tle_line1 = lines[i + 1].strip()
        
        # NORAD ID is characters 3-7 in first TLE line
        tle_line2 = lines[i + 2].strip()
        norad_id = tle_line2.split(" ")[1]
        
        norad_ids[sat_name] = norad_id
        
    return norad_ids

# Example usage
tle_file = "Starlink.tle"
norad_map = get_norad_ids(tle_file)

# for sat_name, norad_id in norad_map.items():
#     print(f"Satellite: {sat_name}, NORAD ID: {norad_id}")

with open("norad_ids.txt", "w") as f:
    for sat_name, norad_id in norad_map.items():
        f.write(f"{sat_name}: {norad_id}\n")

import requests
import datetime
import numpy as np
import time
from tqdm import tqdm

# API credentials
username = "your_username"
password = "your_password"
siteCred = {'identity': username, 'password': password}

# Base URL for Space-Track API
base_url = "https://www.space-track.org"

# TLE API endpoint
tle_endpoint = "/basicspacedata/query/class/tle/EPOCH/>{start_date}%2C<{end_date}/NORAD_CAT_ID/{norad_cat_id}/orderby/EPOCH%20desc/limit/1/format/tle/emptyresult/show"

# Function to fetch TLE data
def fetch_tle_data(session, norad_cat_id, start_date, end_date):
    # Construct the API request URL
    url = base_url + tle_endpoint.replace("{start_date}", start_date).replace("{end_date}", end_date).replace("{norad_cat_id}", norad_cat_id)

    # Send GET request with authentication
    response = session.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.text
        # Process the data as needed
        return data
    else:
        print("Failed to fetch TLE data. Status code:", response.status_code)
        return None


def load_norad_ids(limit=None):
    norad_ids = {}
    with open("./norad_ids.txt", "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            sat_name, norad_id = line.strip().split(": ")
            norad_ids[sat_name] = norad_id
    return norad_ids

def generate_tle_dates(N_TLE):
    # generate N_TLE dates after 2022
    dates = []
    for i in range(N_TLE):
        # Get base date and add days
        month = i % 12 + 1
        base_date = datetime.datetime(2022, month, 1)
        # Add fraction of day to get time component
        start_date = f'{base_date.year}-{base_date.month}-01'
        end_date = f'{base_date.year}-{base_date.month}-28'
        dates.append((start_date, end_date))
    return dates

N_TLE = 1  # TLEs for a month
norad_ids = load_norad_ids(limit=100) # NOTE: limit the number of norad_ids to first 100
dates = generate_tle_dates(N_TLE)

# for testing
# norad_ids = {"STARLINK-24": "44238",
#              "STARLINK-1047": "44752"}

# rate limits 20 requests per minute and 200 requests per hour
with requests.Session() as session:
    login_url = base_url + "/ajaxauth/login"
    login_response = session.post(login_url, data=siteCred)
    if login_response.status_code != 200:
        print("Failed to login")
        exit(0)
    print('Login successful')
    requests_sent = 0
    for i in tqdm(range(1, N_TLE + 1)):
        with open(f"Starlink-{i}.tle", "w") as f:
            for sat_name, norad_id in norad_ids.items():
                tle_data = fetch_tle_data(session, norad_id, dates[i-1][0], dates[i-1][1])
                requests_sent += 1
                if tle_data is not None:
                    # write to file as 
                    f.write(sat_name + "\n")
                    f.write(tle_data)
                time.sleep(1)
                if requests_sent > 0 and requests_sent % 18 == 0:
                    print(f"Rate limit hit. Sleeping for 60 seconds")
                    time.sleep(60)
                if requests_sent > 0 and requests_sent % 198 == 0:
                    print(f"Rate limit hit. Sleeping for 1 hr")
                    time.sleep(3600)
        print(f"Starlink-{i}.txt written")

# 100 Satellites
# 12 TLEs per year
# 12 * 100 = 1200 requests ~ 6 hours

# 1000 Satellites
# 1 TLE
# 1 * 100 = 100 requests ~ 5 minutes
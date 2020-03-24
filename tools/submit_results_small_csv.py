import argparse
import csv
import json
from datetime import datetime

import urllib3
from influxdb import InfluxDBClient



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(description='Add password!')
parser.add_argument("--p")
args = parser.parse_args()
p = args.p
# Create a client
client = InfluxDBClient('dbod-coffea-bench.cern.ch', 8080,'admin' ,p ,'coffea_bench', ssl=True)

def read_data():
    with open('tools/coffea-bench.csv') as f:
        return [x.split(',') for x in f.readlines()[1:]]

a = read_data()

for metric in a:
    time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    influx_metric = [
        {
        "measurement": metric[1],
        "tags": {
            "test": metric[2],
        },
        "time": time,
        "fields": {
             "run_time": metric[10]
        }
    }
        ]
    print(json.dumps(influx_metric, indent=4, sort_keys=True))
    client.write_points(influx_metric)

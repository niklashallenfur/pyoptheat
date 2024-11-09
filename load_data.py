import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')


def load_data(ids, from_time: float, to_time: float):
    sensor_entity_ids = "','".join(ids)
    query = f"""
SELECT m.entity_id, s.state, s.last_updated_ts 
FROM states_meta m
JOIN states s ON m.metadata_id = s.metadata_id
WHERE m.entity_id IN ('{sensor_entity_ids}')
AND last_updated_ts >= '{from_time - 3600}'
AND last_updated_ts <= '{to_time + 3600}'
"""
    with pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name) as conn:
        df = pd.read_sql(query, conn)
    df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s')
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df.dropna(inplace=True)
    minute_timestamps = pd.date_range(start=datetime.fromtimestamp(from_time),
                                      end=datetime.fromtimestamp(to_time), freq='min')
    data = pd.DataFrame(index=minute_timestamps)
    for sensor in ids:
        sensor_df = df[df['entity_id'] == sensor].set_index('last_updated_ts')['state']
        data[sensor] = (sensor_df
                                 .reindex(sensor_df.index.union(minute_timestamps))
                                 .sort_index()
                                 .interpolate(method='time'))
    return data
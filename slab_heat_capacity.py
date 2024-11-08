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

start_time = datetime.fromisoformat("2024-10-21T17:00:00+02:00").timestamp()
end_time = datetime.fromisoformat("2024-10-24T23:00:00+02:00").timestamp()

sensor_entities = ['sensor.radiator_flow_temp',
                   'sensor.radiator_return_temp',
                   'sensor.house_hall_temp']

sensor_entity_ids = "','".join(sensor_entities)

query = f"""
SELECT m.entity_id, s.state, s.last_updated_ts 
FROM states_meta m
JOIN states s ON m.metadata_id = s.metadata_id
WHERE m.entity_id IN ('{sensor_entity_ids}')
AND last_updated_ts >= '{start_time - 3600}'
AND last_updated_ts <= '{end_time + 3600}'
"""
with pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name) as conn:
    df = pd.read_sql(query, conn)

df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s')
df['state'] = pd.to_numeric(df['state'], errors='coerce')
df.dropna(inplace=True)
minute_timestamps = pd.date_range(start=datetime.fromtimestamp(start_time),
                                  end=datetime.fromtimestamp(end_time), freq='min')

interpolated = pd.DataFrame(index=minute_timestamps)
for sensor in sensor_entities:
    sensor_df = df[df['entity_id'] == sensor].set_index('last_updated_ts')['state']
    interpolated[sensor] = (sensor_df
                            .reindex(sensor_df.index.union(minute_timestamps))
                            .sort_index()
                            .interpolate(method='time'))

radiator_flow_temp = interpolated['sensor.radiator_flow_temp']
radiator_return_temp = interpolated['sensor.radiator_return_temp']
house_hall_temp = interpolated['sensor.house_hall_temp']

radiator_temp_diff = (radiator_flow_temp - radiator_return_temp).apply(lambda x: x if 0 < x else 0)
radiator_power = radiator_temp_diff * 980


fig, ax1 = plt.subplots(1, 1, figsize=(20, 15), sharex=True)

ax1.plot(radiator_flow_temp.index, radiator_flow_temp, label="radiator_flow_temp", color="#FFAAAA")
ax1.plot(radiator_return_temp.index, radiator_return_temp, label="radiator_return_temp", color="#AAAAFF")
ax1.plot(house_hall_temp.index, house_hall_temp, label="house_hall_temp", color="#AAFFAA")


ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True)


ax2 = ax1.twinx()
ax2.plot(interpolated.index, radiator_power, label='radiator_power', color='#991111')
ax2.set_ylabel('Power (W)')
ax2.set_ylim(bottom=0)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()

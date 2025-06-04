import pandas as pd
import folium
from shapely.geometry import Point
from datetime import datetime
import random

shootings_df = pd.read_csv('philly_shooting_preprocessed.csv')
shootings_df = shootings_df.dropna(subset=['point_x', 'point_y'])
shootings_df['date_'] = pd.to_datetime(shootings_df['date_']).dt.date

SCHUYLKILL_LONGITUDE = -75.183

def assign_color(lon):
    if lon > SCHUYLKILL_LONGITUDE:
        return 'gray'  
    else:
        return 'green' if random.random() < 0.7 else 'red' 

shootings_df['color'] = shootings_df['point_x'].apply(assign_color)

center_lat = shootings_df['point_y'].mean()
center_lon = shootings_df['point_x'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

for _, row in shootings_df.iterrows():
    popup_info = (
        f"Date: {row['date_']}\n"
        f"Location: {row['location']}"
    )
    folium.CircleMarker(
        location=[row['point_y'], row['point_x']],
        radius=5,
        color=row['color'],
        fill=True,
        fill_opacity=0.7,
        popup=popup_info
    ).add_to(m)

m

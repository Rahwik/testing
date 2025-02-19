import os
import cv2
import pandas as pd
import numpy as np
from glob import glob

LON_MIN, LON_MAX = -180, 180
CONF_COLORS = {50: (0, 0, 255), 75: (0, 255, 255), 100: (0, 255, 0)}

csv_file = r"D:\MDAD.csv"
df = pd.read_csv(csv_file)

image_root = r"E:\Mars_Dust_Storm\Mars 2.0\Files"
output_root = r"E:\Mars_Dust_Storm\Mars 2.0\harvard_output"
os.makedirs(output_root, exist_ok=True)

def extract_info(filename):
    parts = filename.split('_')
    phase = parts[0]
    sol = int(parts[1].replace("day", ""))
    return phase, sol

def latlon_to_px(lat, lon, w, h, lat_min, lat_max):
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h
    return max(0, min(int(round(x)), w - 1)), max(0, min(int(round(y)), h - 1))

def process_image(img_path, df):
    filename = os.path.basename(img_path)
    phase, sol = extract_info(filename)
    storms = df[(df["Mission subphase"].str.lower() == phase.lower()) & (df["Sol"] == sol)]
    if storms.empty:
        return
    
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    for _, row in storms.iterrows():
        lon, lat, CL, year = row[["Centroid longitude", "Centroid latitude", "Confidence interval", "Mars Year"]]
        lat_min, lat_max = (-60, 60) if year <= 28 else (-90, 90)
        x, y = latlon_to_px(lat, lon, w, h, lat_min, lat_max)
        size = 240//2
        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), CONF_COLORS.get(CL, (255, 255, 255)), 2)
        cv2.putText(img, f"CL: {CL}", (x - size, y + size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out_dir = os.path.join(output_root, phase)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, filename), img)

def process_all():
    for folder in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder)
        if os.path.isdir(folder_path):
            for img_path in glob(os.path.join(folder_path, f"{folder}_day*_equat.png")):
                process_image(img_path, df)
                
process_all()

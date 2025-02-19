import json
import pandas as pd
import os

CSV_FILE = r"D:\MDAD.csv"
OUTPUT_FILE = r"E:\Mars_Dust_Storm\Mars 2.0\coco\coco_annotations.json"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def create_coco_structure():
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }

def process_row(row, coco, category_map, image_map, annotation_id):
    mission_subphase = str(row["Mission subphase"])
    sol = str(int(row["Sol"])).zfill(2)
    file_name = f"{mission_subphase}_day{sol}_equat.png"
    
    if file_name not in image_map:
        image_id = len(image_map) + 1
        image_map[file_name] = image_id
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT
        })
    else:
        image_id = image_map[file_name]
    
    category_name = str(row["Member ID"])
    if category_name not in category_map:
        category_id = len(category_map) + 1
        category_map[category_name] = category_id
        coco["categories"].append({
            "id": category_id,
            "name": category_name
        })
    else:
        category_id = category_map[category_name]
    
    xmin, ymin = row["Centroid longitude"], row["Centroid latitude"]
    xmax, ymax = row["Maximum latitude"], row["Minimum latitude"]
    
    bbox_width = abs(xmax - xmin)
    bbox_height = abs(ymax - ymin)
    bbox = [xmin, ymin, bbox_width, bbox_height]
    area = bbox_width * bbox_height
    
    coco["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "segmentation": []
    })
    
    return annotation_id + 1

def save_coco_annotations(coco, output_file):
    try:
        with open(output_file, "w") as f:
            json.dump(coco, f, indent=4)
        print(f"COCO annotations saved as {output_file}")
    except Exception as e:
        print(f"Error saving COCO annotations: {e}")

def main():
    df = load_csv(CSV_FILE)
    if df is None:
        return
    
    coco = create_coco_structure()
    category_map = {}
    image_map = {}
    annotation_id = 1
    
    for index, row in df.iterrows():
        annotation_id = process_row(row, coco, category_map, image_map, annotation_id)
    
    save_coco_annotations(coco, OUTPUT_FILE)

if __name__ == "__main__":
    main()
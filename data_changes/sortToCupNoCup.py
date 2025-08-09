import os
import cv2
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Base photos directory
photos_dir = r'C:\Users\barys\PycharmProjects\BeerDL\photos'

# Define mapping of classes
CUP_CLASSES = ['cup', 'wine glass']
BOTTLE_CLASSES = ['bottle']

# Confidence threshold
conf_thresh = 0.5

# Iterate through each beer type directory (ipa, cider, stout, etc.)
for beer_type in os.listdir(photos_dir):
    beer_type_path = os.path.join(photos_dir, beer_type)
    
    # Skip if not a directory
    if not os.path.isdir(beer_type_path):
        continue
    
    print(f"Processing {beer_type} directory...")
    
    # Create output directories for this beer type
    cups_dir = os.path.join(beer_type_path, 'cups')
    bottles_dir = os.path.join(beer_type_path, 'bottles')
    
    os.makedirs(cups_dir, exist_ok=True)
    os.makedirs(bottles_dir, exist_ok=True)
    
    # Iterate through each brand directory within the beer type
    for brand_dir in os.listdir(beer_type_path):
        brand_path = os.path.join(beer_type_path, brand_dir)
        
        # Skip if not a directory or if it's the cups/bottles directory
        if not os.path.isdir(brand_path) or brand_dir in ['cups', 'bottles']:
            continue
            
        print(f"  Processing brand: {brand_dir}")
        
        # Iterate over images in the brand directory
        for filename in os.listdir(brand_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(brand_path, filename)
                img = cv2.imread(image_path)
                
                if img is None:
                    print(f"    Could not read image: {image_path}")
                    continue
                
                results = model(image_path)
                
                for result in results:
                    detections = result.to_df()
                    print(f"    Found {len(detections)} detections in {filename}")
                    
                    for idx, row in detections.iterrows():
                        box = row['box']
                        x1 = int(box['x1'])
                        y1 = int(box['y1'])
                        x2 = int(box['x2'])
                        y2 = int(box['y2'])
                        label = row['name']
                        confidence = row['confidence']
                        
                        if confidence < conf_thresh:
                            continue
                        
                        # Crop the object
                        crop = img[y1:y2, x1:x2]
                        
                        # Build output path with brand prefix to avoid filename conflicts
                        if label in CUP_CLASSES:
                            out_path = os.path.join(cups_dir, f"{brand_dir}_{filename}_crop_{idx}.jpg")
                            cv2.imwrite(out_path, crop)
                            print(f"      Saved cup crop: {out_path}")
                        elif label in BOTTLE_CLASSES:
                            out_path = os.path.join(bottles_dir, f"{brand_dir}_{filename}_crop_{idx}.jpg")
                            cv2.imwrite(out_path, crop)
                            print(f"      Saved bottle crop: {out_path}")

print("Processing complete!")

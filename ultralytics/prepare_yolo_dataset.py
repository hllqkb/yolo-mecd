import json
import os
import shutil
# A baseline change coco dataset to yolo dataset converter
# --- Configuration --- #
ORIGINAL_DATA_BASE = '/home/hllqk/projects/yolo-mecd/ultralytics/datasets/dataverse_files/'

# Training set configuration
ORIGINAL_TRAIN_IMAGES_SRC = os.path.join(ORIGINAL_DATA_BASE, 'CitDet-train', 'train', 'images')
ORIGINAL_TRAIN_ANNOTATION_FILE = os.path.join(ORIGINAL_DATA_BASE, 'CitDet-train', 'train', 'train_annotations.json')

# Validation set configuration
ORIGINAL_VAL_IMAGES_SRC = os.path.join(ORIGINAL_DATA_BASE, 'CitDet-test', 'test', 'images')
ORIGINAL_VAL_ANNOTATION_FILE = os.path.join(ORIGINAL_DATA_BASE, 'CitDet-test', 'test', 'test_annotations.json')

NEW_DATASET_BASE = os.path.join(ORIGINAL_DATA_BASE, 'CitDet_YOLO_Split')
TRAIN_IMAGES_DIR = os.path.join(NEW_DATASET_BASE, 'train', 'images')
TRAIN_LABELS_DIR = os.path.join(NEW_DATASET_BASE, 'train', 'labels')
VAL_IMAGES_DIR = os.path.join(NEW_DATASET_BASE, 'val', 'images')
VAL_LABELS_DIR = os.path.join(NEW_DATASET_BASE, 'val', 'labels')

# --- Clean up previous run and Create Directories --- #
if os.path.exists(NEW_DATASET_BASE):
    shutil.rmtree(NEW_DATASET_BASE)
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

# --- Function to save classes.txt --- #
def save_classes_file(categories, output_dir):
    """Save the class names to classes.txt in YOLO format"""
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        # Sort categories by their YOLO ID (which is based on the order in the COCO file)
        sorted_categories = sorted(categories, key=lambda x: x['id'])
        for cat in sorted_categories:
            f.write(f"{cat['name']}\n")
    print(f"Saved classes.txt to {output_dir}")

# --- Processing Function --- #
def process_dataset_split(annotation_file, images_src_dir, dest_images_dir, dest_labels_dir, save_classes=False):
    print(f"\nProcessing annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Save classes.txt if requested (only for training set)
    if save_classes:
        save_classes_file(coco_data['categories'], dest_labels_dir)

    image_id_to_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    # Create mapping from COCO category ID to YOLO class ID
    category_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

    processed_count = 0
    for img_id, img_info in image_id_to_info.items():
        original_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # Determine source path (handle .jpg vs .jpeg)
        src_path_jpg = os.path.join(images_src_dir, original_filename)
        src_path_jpeg = os.path.join(images_src_dir, original_filename.replace('.jpg', '.jpeg'))

        src_path = None
        if os.path.exists(src_path_jpg):
            src_path = src_path_jpg
        elif os.path.exists(src_path_jpeg):
            src_path = src_path_jpeg
        else:
            print(f"Warning: Image file not found for {original_filename} in {images_src_dir}. Skipping.")
            continue

        # Determine destination paths
        dest_image_path = os.path.join(dest_images_dir, original_filename)
        dest_label_path = os.path.join(dest_labels_dir, os.path.splitext(original_filename)[0] + '.txt')

        # Copy image
        shutil.copy(src_path, dest_image_path)

        # Write YOLO label file
        with open(dest_label_path, 'w') as f_out:
            if img_id in annotations_by_image_id:
                for ann in annotations_by_image_id[img_id]:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x_min, y_min, width, height]

                    # Convert COCO bbox to YOLO bbox format
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width_norm = bbox[2] / img_width
                    height_norm = bbox[3] / img_height

                    yolo_class_id = category_id_to_yolo_id.get(category_id, -1)
                    if yolo_class_id == -1:
                        print(f"Warning: Category ID {category_id} not found in categories mapping. Skipping annotation.")
                        continue

                    f_out.write(f"{yolo_class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
        processed_count += 1
    print(f"Processed {processed_count} images and annotations.")

# --- Run Processing for Train and Val --- #
# Only save classes.txt for training set (it will be available for validation set too)
process_dataset_split(ORIGINAL_TRAIN_ANNOTATION_FILE, ORIGINAL_TRAIN_IMAGES_SRC, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, save_classes=True)
process_dataset_split(ORIGINAL_VAL_ANNOTATION_FILE, ORIGINAL_VAL_IMAGES_SRC, VAL_IMAGES_DIR, VAL_LABELS_DIR)

print("\nDataset preparation complete!")
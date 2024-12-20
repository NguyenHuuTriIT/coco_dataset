import os
import cv2
import csv
import json
import numpy as np
from pathlib import Path
import multiprocessing
import time
from tqdm import tqdm

# Đường dẫn tới thư mục chứa ảnh và file chú thích
data_dir = Path("D:/THUC_TAP/COCO DATASET/val2017")  # Thư mục ảnh gốc
annotation_file = "D:/THUC_TAP/COCO DATASET/annotations/instances_val2017.json"  # File annotation

# Thư mục để lưu ảnh sau khi xử lý
output_dir = Path("E:/PYTHON/project_COCO/output")  # Thư mục chứa ảnh đầu ra
os.makedirs(output_dir, exist_ok=True)

# File CSV để lưu thông tin
csv_file = output_dir / "object_info_parallel.csv"


# Tạo file CSV nếu chưa có
def create_csv(csv_file):
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["image_name", "object_id", "category_id", "object_area", "image_area", "object_area_percentage",
                 "background_removed"])


# Hàm đọc dữ liệu JSON từng phần với yield
def read_json_with_yield(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]

        # Nhóm annotations theo image_id
        annotation_map = {}
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in annotation_map:
                annotation_map[image_id] = []
            annotation_map[image_id].append(ann)

        # Yield từng ảnh và các annotations liên quan
        for img in images:
            yield {
                "id": img["id"],
                "file_name": img["file_name"],
                "annotations": annotation_map.get(img["id"], [])
            }


# Hàm tính diện tích từ mask
def calculate_mask_area(mask):
    return np.sum(mask > 0)


# Hàm tách và lưu ảnh vật thể
def extract_and_save_objects_bw(image_info):
    try:
        img_path = data_dir / image_info["file_name"]
        img = cv2.imread(str(img_path))

        # Kiểm tra xem ảnh có được tải thành công không
        if img is None:
            print(f"Không thể load ảnh: {image_info['file_name']}")
            return False

        # Tạo thư mục riêng cho từng ảnh gốc
        img_output_dir = output_dir / f"{image_info['file_name'].split('.')[0]}"
        os.makedirs(img_output_dir, exist_ok=True)

        # Lưu ảnh gốc (PNG)
        cv2.imwrite(str(img_output_dir / "original.png"), img)

        img_area = img.shape[0] * img.shape[1]  # Diện tích ảnh gốc

        for i, ann in enumerate(image_info["annotations"]):
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for seg in ann["segmentation"]:
                if isinstance(seg, list):  # Kiểm tra nếu segmentation là danh sách các điểm
                    points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [points], 255)

            object_area = calculate_mask_area(mask)

            # Bỏ qua vật thể có diện tích < 2% diện tích ảnh gốc
            if object_area / img_area < 0.02:
                continue

            # Tạo ảnh trắng đen từ mask
            bw_img = np.where(mask == 255, 255, 0).astype(np.uint8)  # Nền trắng, vật thể đen

            # Lưu ảnh vật thể (JPG)
            bw_output_path = img_output_dir / f"object_{i}_bw.jpg"
            cv2.imwrite(str(bw_output_path), bw_img)

            # Lưu thông tin vào CSV
            object_area_percentage = (object_area / img_area) * 100
            background_removed = np.all(mask == 0)

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([image_info["file_name"],
                                 i,
                                 ann["category_id"],
                                 object_area,
                                 img_area,
                                 object_area_percentage,
                                 background_removed
                                 ])
    except Exception as e:
        print(f"Lỗi xử lý ảnh {image_info['file_name']}: {e}")
        return False
    return True


if __name__ == "__main__":
    # Tạo file CSV để lưu thông tin
    create_csv(csv_file)

    # Đọc dữ liệu JSON từng phần với yield
    image_data_generator = read_json_with_yield(annotation_file)

    # Đo thời gian xử lý
    start_time = time.time()

    # Sử dụng multiprocessing để xử lý song song
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_and_save_objects_bw, image_data_generator),
                            desc="Đang xử lý ảnh"))

    # Kết quả xử lý
    successful_count = sum(results)
    end_time = time.time()

    print(f"\nĐã xử lý thành công {successful_count} ảnh.")
    print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây.")

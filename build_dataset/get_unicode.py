import json
import os
import os.path as osp
import random




# train_data_dir = "/home/dev/Project/VQ-Font/datasets/train_font_image"
# valid_data_dir = "/home/dev/Project/VQ-Font/datasets/valid_font_image"

# train_names = []
# valid_names = []
# for folderName in os.listdir(train_data_dir):
#     if folderName == "reference_images": continue
    
#     for fileName in os.listdir(osp.join(train_data_dir, folderName)):
#         train_names.append(fileName.split('.')[0])

# for folderName in os.listdir(valid_data_dir):
#     for fileName in os.listdir(osp.join(valid_data_dir, folderName)):
#         valid_names.append(fileName.split('.')[0])

# train_chars = list(set(train_names))
# valid_chars = list(set(valid_names))
# 글자 기준으로 명시적으로 split
all_chars = [chr(c) for c in range(0xAC00, 0xD7A4)]  # 11,172자
random.seed(42)
random.shuffle(all_chars)

split = int(len(all_chars) * 0.8)
train_chars = all_chars[:split]   # 8,937자
valid_chars = all_chars[split:]   # 2,235자

# ③ 유니코드 HEX 변환
train_unis = [hex(ord(ch))[2:].upper() for ch in train_chars]
valid_unis = [hex(ord(ch))[2:].upper() for ch in valid_chars]

# ④ JSON 저장
with open("./build_dataset/train_unis_v3.json", "w", encoding="utf-8") as f:
    json.dump(train_unis, f, ensure_ascii=False, indent=2)

with open("./build_dataset/val_unis_v3.json", "w", encoding="utf-8") as f:
    json.dump(valid_unis, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_unis)} 글자, Valid: {len(valid_unis)} 글자 저장 완료")

import pandas as pd
import os

# Đọc các file
train_csv = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\archive\Csv\train.csv"
test_csv = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\archive\Csv\test.csv"
wiki_csv = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\wiki_descriptions.csv"

# Đọc dữ liệu
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
wiki_df = pd.read_csv(wiki_csv)

print("Original train shape:", train_df.shape)
print("Original test shape:", test_df.shape)
print("Wiki shape:", wiki_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("Wiki columns:", wiki_df.columns.tolist())

# Merge train với wiki_descriptions
train_merged = train_df.merge(wiki_df, on='binomial', how='left')
print("\nTrain after merge shape:", train_merged.shape)

# Merge test với wiki_descriptions
test_merged = test_df.merge(wiki_df, on='binomial', how='left')
print("Test after merge shape:", test_merged.shape)

# Tạo đường dẫn hình ảnh động
def create_image_path(row):
    uuid = row['UUID']
    class_id = row['class_id']
    train_dir = f"archive/train/{class_id}/{uuid}.jpg"
    return train_dir

train_merged['image_path'] = train_merged.apply(create_image_path, axis=1)

# Tương tự cho test (nếu UUID có)
if 'UUID' in test_merged.columns:
    test_merged['image_path'] = test_merged.apply(create_image_path, axis=1)

# Sắp xếp cột để dễ đọc
train_cols = ['binomial', 'class_id', 'UUID', 'poisonous', 'country', 'continent', 
              'description', 'image_path', 'X', 'Y', 'height', 'width']
train_cols = [col for col in train_cols if col in train_merged.columns]
train_final = train_merged[train_cols]

# Lưu
train_output = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\train_with_descriptions.csv"
train_final.to_csv(train_output, index=False)
print(f"\n✅ Train dataset saved: {train_output}")

test_cols = ['binomial', 'class_id', 'UUID', 'poisonous', 'country', 'continent', 'description']
test_cols = [col for col in test_cols if col in test_merged.columns]
if 'image_path' in test_merged.columns:
    test_cols.append('image_path')
test_final = test_merged[test_cols]

test_output = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\test_with_descriptions.csv"
test_final.to_csv(test_output, index=False)
print(f"✅ Test dataset saved: {test_output}")

# Thống kê
print("\n" + "="*60)
print("THỐNG KÊ DATASET")
print("="*60)
print(f"\nTrain set:")
print(f"  - Total records: {len(train_final)}")
print(f"  - Unique species: {train_final['binomial'].nunique()}")
print(f"  - With description: {train_final['description'].notna().sum()}")
print(f"  - Venomous: {(train_final['poisonous'] == 1).sum()}")
print(f"  - Non-venomous: {(train_final['poisonous'] == 0).sum()}")

print(f"\nTest set:")
print(f"  - Total records: {len(test_final)}")
print(f"  - Unique species: {test_final['binomial'].nunique()}")
print(f"  - With description: {test_final['description'].notna().sum()}")

print(f"\nSample rows from train:")
print(train_final[['binomial', 'poisonous', 'description', 'image_path']].head(3))
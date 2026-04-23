# 🐍 Hệ Thống Tìm Kiếm Loài Rắn - Snake Species Search System

## Mô Tả
Hệ thống tìm kiếm thông tin loài rắn theo mô tả ngôn ngữ tự nhiên sử dụng:
- **CLIP Model**: Tạo embedding từ text và hình ảnh
- **FAISS Index**: Tìm kiếm nhanh chóng
- **Flask**: Website giao diện người dùng

## Cấu Trúc Thư Mục
```
TimKiemRan/
├── create_embeddings.py       # Script tạo embeddings
├── app.py                     # Flask server
├── requirements.txt           # Dependencies
├── templates/
│   └── index.html            # Website
├── train_with_descriptions.csv # Dữ liệu train
├── embeddings/               # Thư mục output (tự động tạo)
│   ├── faiss_index.bin       # FAISS index
│   ├── metadata.pkl          # Metadata
│   └── embeddings.npy        # Embeddings
└── archive/
    └── train/                # Hình ảnh (hoặc test/)
```
## Dataset

Dự án sử dụng bộ dữ liệu Snake Species Dataset từ Kaggle:

🔗 Dataset: https://www.kaggle.com/datasets/saidmassinissafazez/snakes-species-details-dataset

Dataset bao gồm:
- Hình ảnh các loài rắn
- Tên khoa học (Binomial name)
- Quốc gia phân bố
- Lục địa
- Thông tin độc / không độc
- Mô tả loài

## Hướng Dẫn Cài Đặt & Chạy

### 1. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 2. Tạo CLIP Embeddings & FAISS Index
Chạy lần đầu tiên (mất 5-30 phút tùy số lượng dữ liệu và GPU):
```bash
python create_embeddings.py
```

Điều này sẽ tạo thư mục `embeddings/` chứa:
- `faiss_index.bin` - FAISS index
- `metadata.pkl` - Thông tin về mỗi hình ảnh
- `embeddings.npy` - Vector embeddings

### 3. Chạy Flask Server
```bash
python app.py
```

Output:
```
Running on http://localhost:5000
```

### 4. Mở Website
Mở trình duyệt và truy cập: **http://localhost:5000**

## Cách Sử Dụng Website

### 3 Cách Tìm Kiếm

#### 1️⃣ **Tìm Theo Mô Tả (Text Search)**
- Nhập mô tả bằng ngôn ngữ tự nhiên
- Ví dụ: "rắn xanh", "rắn độc vàng", "rắn có vân ngang"
- Dùng CLIP text encoder để so sánh với image embeddings trong FAISS
- ✨ **Ưu điểm**: Nhanh, không cần hình ảnh

#### 2️⃣ **Tìm Theo Hình Ảnh (Image Search)**
- Upload một hình ảnh rắn
- Dùng CLIP image encoder để so sánh trực tiếp với các hình trong database
- ✨ **Ưu điểm**: Chính xác, visual matching

#### 3️⃣ **Tìm Kết Hợp (Fusion Search)** ⭐
- Nhập **cả mô tả VÀ upload hình ảnh**
- Kết hợp: **70% hình ảnh + 30% mô tả** (có thể điều chỉnh)
- ✨ **Ưu điểm**: Kết quả tốt nhất, kết hợp text + image information

### Quy Trình Tìm Kiếm

```
1. Chọn tab (Text / Image / Fusion)
2. Nhập mô tả hoặc upload hình (tùy theo tab)
3. Điều chỉnh số kết quả (1-10, mặc định 5)
4. Nhấn "Tìm Kiếm"
5. Xem kết quả:
   - Hình ảnh loài rắn
   - Tên khoa học (Binomial name)
   - Điểm tương đồng (0-1, cao hơn = giống nhau hơn)
   - Thông tin: Quốc gia, Lục địa, Độc/Không độc
```

## Công Nghệ

### CLIP Model
- **OpenAI CLIP ViT-Base-Patch32**
- Kích thước: 512-dimensional embeddings
- Ưu điểm: Kết hợp hiểu text + image trong cùng không gian vector

### FAISS Index
- **IndexFlatIP** (Inner Product)
- Lưu trữ: **Image embeddings normalized**
- Tìm kiếm: O(n) brute force (nhanh với ~1000-10000 embeddings)
- Scoring: Cosine similarity (0-1, cao hơn = giống nhau hơn)

### Flask + JavaScript
- Backend: Python Flask
- Frontend: HTML5 + CSS3 + Vanilla JavaScript
- API: 3 endpoints (/api/search/text, /api/search/image, /api/search/fusion)

## Ghi Chú

### GPU vs CPU
- **GPU**: Nhanh hơn, tương tác liền mạch
- **CPU**: Chậm hơn nhưng vẫn chạy được

Script sẽ tự động detect GPU nếu có.

### Tối Ưu Hóa
Nếu muốn dùng GPU NVIDIA nhanh hơn:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Troubleshooting

**Lỗi: CUDA out of memory**
- Giảm số lượng items xử lý trong `create_embeddings.py`
- Hoặc dùng CPU

**Lỗi: Module not found**
- Chạy: `pip install -r requirements.txt` lại

**Lỗi: Hình ảnh không hiển thị**
- Kiểm tra đường dẫn trong `train_with_descriptions.csv`
- Đảm bảo `archive/` folder tồn tại

## Tính Năng

✅ Tìm kiếm theo mô tả text tự nhiên
✅ Hiển thị hình ảnh loài rắn
✅ Điểm tương đồng (similarity score)
✅ Thông tin chi tiết: độc/không độc, quốc gia, lục địa
✅ Giao diện thân thiện, responsive
✅ Hỗ trợ 1-10 kết quả

## Author
Snake Species Search System - 2024

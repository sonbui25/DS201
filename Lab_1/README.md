# Hướng dẫn chạy

Mô tả ngắn: script sẽ huấn luyện lần lượt hai mô hình (One-layer MLP và Three-layer MLP) để phân loại chữ viết tay trong bộ dữ liệu MNIST và hiển thị cửa sổ kết quả sau mỗi lần huấn luyện.

## Yêu cầu
- Python 3.12, PyTorch, torchvision, scikit-learn, matplotlib, tabulate
- (Nếu dùng conda) kích hoạt môi trường:  
  conda activate ds201_env

## Các bước chạy
1. Mở terminal (hoặc PowerShell) và điều hướng đến thư mục project:
   ```
   cd \DS201\Lab_1
   ```
2. Chạy:
   ```
   python main.py
   ```
3. Chương trình sẽ huấn luyện model One-layer MLP trước. Khi hoàn tất, một cửa sổ đồ họa sẽ hiện các biểu đồ/điểm số.
4. Quan sát kết quả. Đóng cửa sổ đồ họa để chương trình tiếp tục.
5. Tiếp theo chương trình sẽ huấn luyện model Three-layer MLP và lặp lại bước 3–4.
6. Khi toàn bộ hoàn tất, kết quả và mô hình được lưu trong thư mục `results`.

## Ghi chú
- Kết quả huấn luyện (checkpoint/biểu đồ) được lưu tự động — kiểm tra thư mục `result` / `models`.

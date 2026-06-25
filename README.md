# Phân loại kỹ thuật tấn công MITRE ATT&CK từ văn bản Cyber Threat Intelligence

## 1. Giới thiệu dự án
Trong hoạt động an toàn thông tin hiện đại, việc phân tích lượng lớn báo cáo tình báo mối đe dọa (CTI) thủ công mất rất nhiều thời gian và dễ thiếu nhất quán. Dự án này nhằm xây dựng một hệ thống tự động phân loại văn bản CTI. Ở giai đoạn này, hệ thống giải quyết bài toán phân loại đơn nhãn (Single-label classification): nhận đầu vào là một đoạn văn bản mô tả hành vi tấn công và đầu ra là một kỹ thuật MITRE ATT&CK duy nhất phù hợp nhất.

## 2. Tính năng cốt lõi
* **Tiền xử lý thông minh:** Làm sạch văn bản CTI, chuẩn hóa cấu trúc nhưng vẫn giữ lại các từ khóa chuyên môn (ví dụ: PowerShell, registry, C2) để đảm bảo mô hình không mất đi các dấu hiệu nhận diện quan trọng.
* **Mô hình cơ sở (Baseline Models):** Xây dựng và đánh giá các mô hình phân loại dựa trên kỹ thuật máy học truyền thống như Logistic Regression và Linear SVM kết hợp cùng TF-IDF n-gram.
* **Đánh giá chuyên sâu:** Đánh giá hiệu năng mô hình không chỉ bằng Accuracy mà tập trung vào các độ đo như Macro-F1, Precision và Recall để xử lý tình trạng lệch lớp (imbalanced data) đặc thù của dữ liệu CTI.
* **Giao diện tương tác:** Tích hợp ứng dụng web demo bằng Streamlit, hỗ trợ chuyên gia nhập nhanh văn bản CTI và nhận về kỹ thuật tấn công được dự đoán.

## 3. Cấu trúc dự án

```text
attack-classification/
├── data/               # Chứa dữ liệu gốc (raw) và dữ liệu sau khi làm sạch (processed)
├── notebooks/          # Không gian nghiên cứu (EDA, thử nghiệm Baseline & Error Analysis)
├── src/                # Chứa mã nguồn chính (xử lý dữ liệu, huấn luyện, đánh giá)
├── app/                # Chứa mã nguồn giao diện web (Streamlit)
├── results/            # Chứa các file kết quả (metrics, biểu đồ, ma trận nhầm lẫn)
├── models/             # Thư mục lưu trữ các mô hình đã huấn luyện (.pkl)
├── requirements.txt    # Danh sách các thư viện môi trường cần thiết
└── README.md           # Tài liệu hướng dẫn dự án

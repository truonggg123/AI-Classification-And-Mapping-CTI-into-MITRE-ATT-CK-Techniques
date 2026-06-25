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




#### Nhiệm vụ:
1. Data Engineer (Tiền xử lý & Khám phá dữ liệu) (Trường + Khoa)
•	Nhiệm vụ:
o	Đọc dữ liệu CTI, thực hiện EDA (vẽ biểu đồ phân bố nhãn).
o	Viết hàm clean_text để chuẩn hóa văn bản, đặc biệt lưu ý giữ lại các keyword kỹ thuật.
o	Chia tập dữ liệu thành Train/Test split.
•	File phụ trách: notebooks/01_eda.ipynb, src/data_loader.py, src/preprocess.py.
2. Feature Engineer (Trích xuất đặc trưng) (Quy)
•	Nhiệm vụ:
o	Tokenize
o	Cài đặt và tinh chỉnh các bộ vectorizer: CountVectorizer và TfidfVectorizer (thử nghiệm với các n-gram khác nhau).
o	Nghiên cứu tích hợp Word2Vec hoặc FastText (nếu có thời gian) để nắm bắt ngữ nghĩa của các từ vựng OOV (Out-Of-Vocabulary) trong mảng an toàn thông tin.
•	File phụ trách: Hỗ trợ hoàn thiện phần biểu diễn dữ liệu trong src/preprocess.py hoặc tạo một file riêng biệt để định nghĩa các bộ vectorizer.
3. ML Pipeline & Core Modeling (Xây dựng luồng huấn luyện) (Hiếu)
Trường sẽ đóng vai trò thiết lập bộ khung tự động hóa việc huấn luyện mô hình.
•	Nhiệm vụ:
o	Thiết lập Pipeline của scikit-learn để nối trực tiếp các Vectorizer với các mô hình phân loại.
o	Cài đặt GridSearchCV để tự động chạy và tìm siêu tham số tốt nhất.
o	Triển khai tích hợp trước 3 mô hình cơ bản và chạy nhanh nhất: Logistic Regression, LinearSVC, và Multinomial Naive Bayes.
•	File phụ trách: notebooks/02_baseline_tfidf_svm.ipynb, src/train_baseline.py (phần luồng chính).
4. Advanced Baselines & Evaluation (Mô hình nâng cao & Đánh giá) (P.Anh)
Hiếu sẽ bổ sung các mô hình phức tạp hơn vào Pipeline của Trường và chịu trách nhiệm "chấm điểm" hệ thống một cách khắt khe nhất.
•	Nhiệm vụ:
o	Bổ sung Random Forest và LightGBM vào hệ thống Pipeline để so sánh.
o	Viết các hàm tính toán độ đo thực tế: Macro-F1, Precision, Recall.
o	Trích xuất ma trận nhầm lẫn (Confusion Matrix) và phân tích lỗi (Error Analysis) xem mô hình đang nhầm lẫn ở những technique nào.
•	File phụ trách: Hỗ trợ ở src/train_baseline.py, phụ trách chính src/evaluate.py và notebooks/04_error_analysis.ipynb.



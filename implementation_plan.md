# Kế hoạch Nghiên cứu & Thực nghiệm Deep Learning: Ánh xạ CTI sang MITRE ATT&CK (CTI-to-MITRE & TRAM)

Tài liệu này xác định kế hoạch thực nghiệm chuyên sâu giai đoạn 2 (Deep Learning) cho hệ thống tự động ánh xạ CTI Narrative sang **188 Nhãn Active MITRE ATT&CK Parent Techniques**. 

Kế hoạch thiết lập khung đánh giá 4 kịch bản (**Scenario A, B, C, D**) độc lập và toàn diện trên 2 tập dữ liệu benchmark chính (**CTI-to-MITRE** và **TRAM**), đồng thời tích hợp 4 kiến trúc mô hình Deep Learning tiên tiến: **TextCNN**, **ModernBERT**, **SecureBERT 2.0 + Asymmetric Loss (ASL)**, và **Bi-Encoder Dense Retrieval**.

---

## 🎯 Khung 4 Kịch Bản Thực Nghiệm (4-Scenario Evaluation Protocol)

Mọi mô hình (từ Baseline đến Deep Learning) và mọi phương pháp Augmentation đều được đánh giá qua 4 kịch bản độc lập:

1. **Scenario A (In-Domain CTI-to-MITRE Benchmark):**
   - **Train Set:** `dataset/processed/cti_to_mitre/train.csv` (10,345 mẫu - 80%)
   - **Test Set:** `dataset/processed/cti_to_mitre/test.csv` (2,599 mẫu - 20%)
   - **Không gian nhãn:** 188 Active Parent Techniques.
   - **Mốc Augmentation (MEAN):** `--target_count 55` mẫu/nhãn.

2. **Scenario B (In-Domain TRAM Benchmark):**
   - **Train Set:** `dataset/processed/tram/train.csv` (6,803 mẫu - 80%)
   - **Test Set:** `dataset/processed/tram/test.csv` (1,705 mẫu - 20%)
   - **Không gian nhãn:** 50 Active Parent Techniques.
   - **Mốc Augmentation (MEAN):** `--target_count 151` mẫu/nhãn.

3. **Scenario C (Cross-Dataset Generalization - Domain Shift Evaluation):**
   - **Kịch bản C1:** Train trên `CTI-to-MITRE Train` $\rightarrow$ Test trên `TRAM Test`.
   - **Kịch bản C2:** Train trên `TRAM Train` $\rightarrow$ Test trên `CTI-to-MITRE Test`.
   - **Không gian nhãn:** Đánh giá trên tập nhãn giao (Intersecting Active Labels).
   - **Mục tiêu:** Kiểm chứng tính bền vững của mô hình và phương pháp Augmentation trước sự thay đổi phân phối văn bản báo cáo CTI.

4. **Scenario D (Joint Dataset Benchmark):**
   - **Train Set:** `dataset/processed/joint/train.csv` (17,161 mẫu - 80%)
   - **Test Set:** Đánh giá độc lập trên `cti_to_mitre/test.csv`, `tram/test.csv`, và `joint/test.csv` (4,291 mẫu - 20%).
   - **Không gian nhãn:** Không gian nhãn hợp (Union Label Space - 188 Active Techniques).
   - **Mốc Augmentation (MEAN):** `--target_count 95` mẫu/nhãn.

---

## 🔄 Ma Trận So Sánh Các Phương Pháp Augmentation (Data Augmentation Benchmark Matrix)

Trên từng Kịch bản (Scenario A, B, C, D), thực hiện so sánh đối chứng 6 phương pháp Augmentation:

| STT | Phương Pháp Augmentation | Mã Chạy CLI (`--mode`) | Mô Tả Kỹ Thuật |
| :---: | :--- | :---: | :--- |
| 1 | **No Augmentation** | `no_aug` | Dữ liệu gốc sau tiền xử lý (Baseline đối chứng). |
| 2 | **Synonym Replacement (SR)** | `sr` | Thay thế $n = \lceil \alpha \times L \rceil$ từ bằng từ đồng nghĩa An ninh mạng. |
| 3 | **Random Insertion (RI)** | `ri` | Chèn ngẫu nhiên từ đồng nghĩa của một từ hợp lệ vào vị trí bất kỳ trong câu. |
| 4 | **Random Swap (RS)** | `rs` | Hoán đổi vị trí 2 từ ngẫu nhiên trong câu (Bảo vệ dấu câu cuối). |
| 5 | **Random Deletion (RD)** | `rd` | Xóa ngẫu nhiên các từ không thuộc Tap Bảo vệ với xác suất $p = \alpha$. |
| 6 | **Cyber EDA (Proposed Method)** | `eda` | Kết hợp 4 biến đổi + Tap Bảo vệ Từ vựng TF-IDF (3,636 từ) + Greedy Resampling. |

---

## 👥 Phân Công Nhiệm Vụ Trong Nhóm (Task Allocation)

| Thành viên | Vai trò & Trách nhiệm chính | Mô hình / Module Phụ trách | Output Bàn giao |
| :--- | :--- | :--- | :--- |
| **Trường** *(Team Lead)* | • Thiết kế Data Pipeline & Split độc lập 4 Scenario<br>• Phát triển module Cyber EDA & Benchmark Operations<br>• Fine-tune & Benchmark ModernBERT | **ModernBERT**<br>+ Data Split & Augmentation Module | • `src/dataset_splitter.py`<br>• `src/augmentation.py`<br>• `src/models/modernbert_trainer.py`<br>• Notebook `05_modernbert_experiment.ipynb` |
| **Khoa** | • Xây dựng Baseline (Logistic Regression, Linear SVC)<br>• Xây dựng và thực nghiệm kiến trúc TextCNN<br>• Đánh giá 4 Scenario trên Baseline & TextCNN | **TextCNN & Baseline**<br>+ Baseline Pipeline | • Notebook `03_kaggle_baseline_pipeline.ipynb`<br>• `src/models/text_cnn.py`<br>• Notebook `04_textcnn_experiment.ipynb` |
| **P.Anh** | • Cài đặt hàm Asymmetric Loss (ASL)<br>• Fine-tune SecureBERT 2.0 trên ngữ cảnh an ninh mạng<br>• Tinh chỉnh ngưỡng động (Dynamic Thresholding) cho Deep Learning | **SecureBERT 2.0 + ASL**<br>+ ASL Loss Module | • `src/losses/asl_loss.py`<br>• `src/models/securebert_asl_trainer.py`<br>• Notebook `06_securebert_asl_experiment.ipynb` |
| **Quy** | • Xây dựng kiến trúc Dual-Tower Bi-Encoder<br>• Trích xuất Embeddings mô tả chuẩn MITRE 188 nhãn<br>• Huấn luyện Supervised Contrastive / Cosine Similarity Retrieval | **Bi-Encoder Dense Retrieval**<br>+ Dual-Tower Architecture | • `src/models/bi_encoder.py`<br>• `src/mitre_description_extractor.py`<br>• Notebook `07_biencoder_retrieval_experiment.ipynb` |

---

## 🏗️ Chiến Lược Kiến Trúc Mô Hình & Loss Functions

### 1. TextCNN (Baseline Deep Learning)
- Sử dụng mô hình Convolutional Neural Network cho phân loại văn bản đa nhãn với các kích thước filter $k \in \{3, 4, 5\}$.
- Kết hợp Global Max Pooling và Dense Classifier với Sigmoid Activation.

### 2. ModernBERT (General SOTA Transformer)
- Kiến trúc Transformer thế hệ mới (ModernBERT-base) hỗ trợ ngữ cảnh dài, Rotary Positional Embeddings (RoPE), và FlashAttention-2 giúp tăng tốc huấn luyện.
- Fine-tuning với hàm mất mát BCE + Class Weighting.

### 3. SecureBERT 2.0 + Asymmetric Loss (ASL)
- Sử dụng **SecureBERT 2.0** (bộ mã hóa Transformer pre-trained chuyên biệt trên dữ liệu An toàn thông tin / CTI).
- Kết hợp hàm mất mát bất đối xứng **Asymmetric Loss (ASL)** nhằm giải quyết triệt để bài toán imbalance cực đoan ở 101 nhãn Tail ($<30$ mẫu):
  $$\mathcal{L}_{ASL} = - \sum_{k=1}^{188} y_k (1 - p_k)^{\gamma_+} \log(p_k) + (1 - y_k) (p_m)^{\gamma_-} \log(1 - p_m)$$
  với $p_m = \max(p_k - \text{margin}, 0)$, $\gamma_- = 4, \gamma_+ = 1, \text{margin} = 0.05$.

### 4. Bi-Encoder Dense Retrieval (Zero-Shot / Few-Shot Semantic Alignment)
- Kiến trúc Dual-Tower: 
  - Tower 1: $E_{CTI}(x) = \text{SecureBERT}(x_{\text{CTI narrative}})$
  - Tower 2: $E_{MITRE}(t_i) = \text{SecureBERT}(\text{Description of Technique } t_i)$
- Dự đoán bằng độ tương đồng Cosine: $s(x, t_i) = \cos(E_{CTI}(x), E_{MITRE}(t_i))$.
- Đánh giá khả năng nhận diện zero-shot / few-shot cho các kỹ thuật hiếm thông qua chỉ số **Recall@3** và **Recall@5**.

---

## 🧪 Hệ Thống Bảng & Biểu Đồ Thực Nghiệm (Required Outputs)

Mọi kết quả thực nghiệm sẽ được xuất và quản lý động theo từng thư mục mô hình: `results/[model_name]_results/` 
(Ví dụ: `results/baseline_results/`, `results/textcnn_results/`, `results/modernbert_results/`, `results/securebert_asl_results/`, `results/biencoder_results/`).

### 🎯 Các Chỉ Số Đánh Giá Chuẩn Quốc Tế (Evaluation Metrics)
Mỗi mô hình đều được đánh giá qua bộ 9 chỉ số chuẩn mực NCKH:
1. **Macro F1** (Đánh giá mức độ công bằng trên cả nhãn hiếm)
2. **Micro F1** (Đánh giá tổng thể số lượng mẫu)
3. **Precision (Macro)**
4. **Recall (Macro)**
5. **Hamming Loss** (Độ hao hụt dự đoán đa nhãn)
6. **Exact Match Accuracy** (Tỷ lệ khớp 100% tập nhãn)
7. **Recall@1, Recall@3, Recall@5, Recall@10** (Tỷ lệ nhãn thực tế nằm trong Top-k kỹ thuật được mô hình xếp hạng cao nhất)

### 📊 Danh Sách Bảng Kết Quả Chi Tiết Theo Kịch Bản (Tables)
1. **Scenario A Output**: `results/[model_name]_results/scenario_A_cti_to_mitre.csv` / `.png` - Bảng & biểu đồ thực nghiệm In-Domain CTI-to-MITRE (Bao gồm Recall@1, Recall@3, Recall@5, Recall@10).
2. **Scenario B Output**: `results/[model_name]_results/scenario_B_tram.csv` / `.png` - Bảng & biểu đồ thực nghiệm In-Domain TRAM (Bao gồm Recall@k).
3. **Scenario C Output**: `results/[model_name]_results/scenario_C_cross_dataset.csv` / `.png` - Bảng & biểu đồ thực nghiệm Cross-Dataset Generalization.
4. **Scenario D Output**: `results/[model_name]_results/scenario_D_joint_dataset.csv` / `.png` - Bảng & biểu đồ thực nghiệm Joint Dataset.
5. **Master Summary Table**: `results/[model_name]_results/master_table_all_scenarios_comparison.csv` - Bảng tổng hợp toàn bộ 4 Scenario.
6. **Table 4 (4-Tier Analysis)**: `results/[model_name]_results/4_tier_error_analysis.json` & `table5_per_label_metrics.csv` - Phân tích hiệu năng theo 4 tầng (Head $>500$, Major $100-499$, Medium $30-99$, Tail $<30$).

### 📈 Biểu Đồ So Sánh Quỹ Đạo Học Theo Epoch & Ranking Metrics
7. **Figure: Comparison of Macro F1 across Epochs for Deep Learning Models**:
   - **Tên file xuất:** `results/deeplearning_results/epoch_learning_trajectory_comparison.png`
   - **Cấu trúc biểu đồ:** Biểu đồ sóng đôi 2 Subplots song song:
     - **Subplot trái (CTI-to-MITRE):** Trục tung `F1_MACRO`, Trục hoành `Epochs` ($1 \rightarrow 10$).
     - **Subplot phải (TRAM):** Trục tung `F1_MACRO`, Trục hoành `Epochs` ($1 \rightarrow 10$).
   - **So sánh các mô hình Deep Learning trong Plan hiện tại:**
     - **Đường màu xanh lam:** `TextCNN`
     - **Đường màu cam/xanh lá:** `ModernBERT-base`
     - **Đường màu đỏ (SOTA):** `SecureBERT 2.0 + ASL` (Mô hình đề xuất)
   - **Phân biệt phương pháp Augmentation qua Kiểu đường (Line Styles):**
     - **Nét liền (Solid Line `-`):** Dữ liệu gốc (`No Augmentation`).
     - **Nét đứt (Dashed Line `--`):** Dữ liệu tăng cường (`Cyber EDA`).
   - **Ý nghĩa khoa học:** Đánh giá tốc độ hội tụ, khả năng chống over-fitting và sự bùng nổ chỉ số Macro F1 ở các Epoch muộn khi áp dụng Cyber EDA so với dữ liệu gốc.

8. **Figure: Top-k Ranking Performance (Recall@k Benchmark with Cyber EDA)**:
   - **Tên file xuất:** `results/[model_name]_results/recall_at_k_ranking.png`
   - **Tiêu đề hình:** `Chart 8: Top-k Ranking Performance (Recall@k) Across Baseline Models & Datasets (Cyber EDA Augmentation)`
   - **Cấu trúc biểu đồ:** Biểu đồ cột sóng đôi 2 subplots thể hiện tỷ lệ tìm thấy nhãn đúng tại các ngưỡng Top-k ($k \in \{1, 3, 5, 10\}$) cho cả 2 mô hình (Logistic Regression vs Linear SVC) trên 2 tập dataset (CTI-to-MITRE Scenario A và TRAM Scenario B) áp dụng tăng cường dữ liệu Cyber EDA.
   - **Ý nghĩa khoa học:** Minh chứng khả năng xếp hạng ứng viên của mô hình, giúp chuyên gia SOC nhanh chóng định vị kỹ thuật MITRE ATT&CK đúng trong Top-k gợi ý.

---

## 🔍 Khung Phân Tích Lỗi Chuyên Sâu (Comprehensive Error Analysis Framework)

Để tài liệu NCKH đạt chất lượng cao nhất, hệ thống thực nghiệm tích hợp khung phân tích lỗi toàn diện trên 3 khía cạnh tại `results/[model_name]_results/`:

### 1. Phân Tích Hiệu Năng Theo 4 Tầng Tần Suất (4-Tier Frequency Breakdown)
Chia 188 nhãn MITRE ATT&CK Active Parent Techniques thành 4 tầng dựa trên tần suất mẫu huấn luyện:
- **Head Tier ($>500$ mẫu):** Các kỹ thuật xuất hiện phổ biến nhất (T1059, T1071,...).
- **Major Tier ($100 - 499$ mẫu):** Các kỹ thuật tần suất cao.
- **Medium Tier ($30 - 99$ mẫu):** Các kỹ thuật tần suất trung bình.
- **Tail Tier ($<30$ mẫu - 101 nhãn hiếm):** Nhóm kỹ thuật hiếm bị mất cân bằng nặng nề nhất.
- **Output:** `results/[model_name]_results/4_tier_error_analysis.json` & `table4_frequency_tier_breakdown.png`

### 2. Phân Tích Cặp Nhãn Hay Nhầm Lẫn Nhất (Confused Label Matrix)
- Trích xuất Top 10 cặp nhãn MITRE ATT&CK có tỷ lệ nhầm lẫn cao nhất (co-occurrence of False Positives & False Negatives).
- Giúp phát hiện các kỹ thuật có mô tả ngữ nghĩa quá gần nhau (ví dụ: nhầm lẫn giữa T1059.001 PowerShell và T1059.003 Windows Command Shell, hoặc T1003 OS Credential Dumping và T1555 Credentials from Password Stores).
- **Output:** `results/[model_name]_results/confused_label_pairs.csv` & `confused_label_pairs.png`

### 3. Phân Tích Định Tính Theo 3 Dạng Lỗi (Categorical Case Studies)
- **Zero-Hit / Under-Prediction (Mô hình bỏ sót):** Nhãn thực tế có trong Ground Truth nhưng mô hình không dự đoán được (Recall thấp ở nhãn Tail).
- **Spurious / Over-Prediction (Mô hình gán thừa):** Mô hình gán nhãn không có trong Ground Truth do bắt nhầm từ khóa nhiễu (Precision thấp).
- **Partial Match / Exact Match Failure:** Phân tích các văn bản CTI dự đoán đúng một phần tập nhãn nhưng không đạt được Exact Match Accuracy.
- **Output:** `results/[model_name]_results/text_case_studies.json` (Trích xuất 9 mẫu định tính tiêu biểu cho bài báo).

### 4. Danh Sách Các Nhãn Không Dự Đoán Được (Zero-F1 / Unpredicted Techniques Tracker)
- Thống kê toàn bộ các kỹ thuật MITRE ATT&CK có chỉ số **F1-Score = 0.0** (mô hình không thể nhận diện thành công mẫu nào).
- Thống kê số lượng nhãn bị F1 = 0.0 theo từng tầng tần suất (Head, Major, Medium, Tail) để theo dõi sự giảm thiểu nhãn chết khi áp dụng Cyber EDA và các mô hình Deep Learning.
- **Output:** `results/[model_name]_results/zero_f1_unpredicted_labels.csv` & tích hợp trong `4_tier_error_analysis.json`

---

## 📂 Cấu Trúc Thư Mục Dự Án Hoàn Chỉnh

```text
AI-Classification-And-Mapping-CTI-into-MITRE-ATT-CK-Techniques/
├── DATASET_GUIDE.txt                    # Hướng dẫn quy trình 3 bước xử lý dữ liệu cho đồng đội
├── PROJECT_EXECUTION_SUMMARY.txt        # Báo cáo tổng hợp tiến độ & cơ chế Cyber EDA
├── implementation_plan.md               # Tài liệu Kế hoạch Thực nghiệm 4 Scenario
│
├── dataset/
│   ├── raw/                             # Dữ liệu thô gốc (dataset.csv, single_label.json, multi_label.json)
│   └── processed/                       # Dữ liệu tiền xử lý theo từng tập
│       ├── cti_to_mitre/                # (12,944 samples, Train 10,345 / Test 2,599)
│       ├── tram/                        # (8,508 samples, Train 6,803 / Test 1,705)
│       └── joint/                       # (21,452 samples, Train 17,161 / Test 4,291)
│
├── src/                                 # Mã nguồn Python mô đun hóa
│   ├── __init__.py
│   ├── merge_datasets.py                # Chuẩn hóa & Khử trùng lặp raw dataset cho cti_to_mitre, tram, joint
│   ├── preprocessing.py                 # Clean text, ẩn ma Txxxx, binarizer động, 80/20 stratified split
│   ├── augmentation.py                  # Cyber EDA, SR, RI, RS, RD, Back-Translation
│   ├── losses/
│   │   ├── __init__.py
│   │   └── asl_loss.py                  # Asymmetric Loss (ASL)
│   └── models/
│       ├── __init__.py
│       ├── text_cnn.py                  # Mô hình TextCNN
│       ├── modernbert_trainer.py        # Pipeline fine-tune ModernBERT
│       ├── securebert_asl_trainer.py    # Pipeline fine-tune SecureBERT 2.0 + ASL
│       └── bi_encoder.py                # Pipeline Bi-Encoder Dual-Tower
│
├── notebooks/                           # Jupyter Notebooks thực thi (100% English)
│   ├── 01a_explore_cti_to_mitre.ipynb   # EDA độc lập tập CTI-to-MITRE
│   ├── 01b_explore_tram.ipynb           # EDA độc lập tập TRAM
│   ├── 01c_explore_joint.ipynb          # EDA tập Joint (Hợp nhất)
│   ├── 01_merge_and_explore_dataset.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_kaggle_baseline_pipeline.ipynb # Notebook thực nghiệm 4 Scenario Baseline & Augmentation
│   ├── 04_textcnn_experiment.ipynb      # Notebook thực nghiệm TextCNN (Khoa)
│   ├── 05_modernbert_experiment.ipynb   # Notebook thực nghiệm ModernBERT (Trường)
│   ├── 06_securebert_asl_experiment.ipynb # Notebook thực nghiệm SecureBERT + ASL (P.Anh)
│   └── 07_biencoder_retrieval_experiment.ipynb # Notebook thực nghiệm Bi-Encoder (Quy)
│
└── results/                             # Thư mục lưu kết quả thực nghiệm
    ├── EDA/                             # Biểu đồ PNG & CSV báo cáo EDA của 3 tập
    │   ├── cti_to_mitre/
    │   ├── tram/
    │   └── joint/
    └── baseline_results/                # Bảng CSV, biểu đồ PNG & báo cáo JSON của 4 Scenario
```

---

## 🎯 Kế Hoạch Xác Nhận & Kiểm Thử (Verification Plan)

### Automated Verification
- Kiểm tra tính liên thông dữ liệu qua lệnh:
  ```bash
  python src/merge_datasets.py --target_dataset cti_to_mitre
  python src/preprocessing.py --target_dataset cti_to_mitre
  python src/augmentation.py --train_file dataset/processed/cti_to_mitre/train.csv --mode eda --save_csv
  ```
- Kiểm tra notebook `03_kaggle_baseline_pipeline.ipynb` tạo đủ các bảng CSV và biểu đồ PNG trong `results/baseline_results/` cho 4 Kịch bản Scenario A, B, C, D.

### Manual Review
- Đánh giá chỉ số Macro F1 & Micro F1 giữa các phương pháp Augmentation trên từng Scenario.
- Kiểm tra số lượng nhãn bị F1 = 0.0 ở tầng Tail (<30 mẫu).

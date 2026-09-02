# 🛡️ AI-Classification-And-Mapping-CTI-into-MITRE-ATT-CK-Techniques
> **Nghiên cứu Khoa học (NCKH): Hệ thống Phân loại Đa nhãn & Ánh xạ Báo cáo Mối đe dọa An ninh mạng (CTI Narrative) sang Kỹ thuật MITRE ATT&CK**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE%20ATT%26CK-v2.1%20Enterprise-red.svg)](https://attack.mitre.org/)
[![Dataset Split](https://img.shields.io/badge/Split-70%20Train%20%7C%2010%20Val%20%7C%2020%20Test-green.svg)](#-cấu-trúc-bộ-dữ-liệu--3-benchmarks-chuẩn)
[![Evaluation Seed](https://img.shields.io/badge/Controlled%20Seed-42-purple.svg)](#-kế-hoạch-thực-nghiệm-đối-chứng-team-ablation-plan-v2-single-seed)

---

## 📌 Mục Tiêu Nghiên Cứu (Research Objectives)

Dự án phát triển giải pháp toàn diện cho bài toán **Extreme Multi-Label Classification & Imbalance Learning** trong lĩnh vực An toàn thông tin:
1. **Tự động ánh xạ CTI Narrative $\rightarrow$ 188 Kỹ thuật MITRE ATT&CK Parent Techniques** (`Txxxx`).
2. **Khắc phục mất cân bằng lớp cực đoan (Long-tail Imbalance)**: Hơn 50% nhãn có dưới 30 mẫu huấn luyện.
3. **Phát triển cơ chế Tăng cường Dữ liệu An ninh mạng (Cyber EDA)**: Bảo vệ ngữ cảnh miền bằng tri thức STIX v2.1 kết hợp lọc động TF-IDF IDF (3,636 từ vựng được bảo vệ), khắc phục triệt để hiện tượng sai lệch ngữ nghĩa (Domain Distortion) của các phương pháp EDA truyền thống.
4. **Khung Thực nghiệm Đối chứng Chuẩn hóa (Team Plan V2 - Single Seed 42)**: Kiểm chứng khoa học đa kịch bản (A0, G0, B1, G1, B2_E1) trên 3 benchmark chuẩn.

---

## 🗂️ Cấu Trúc Bộ Dữ Liệu & 3 Benchmarks Chuẩn

Dữ liệu được làm sạch, ẩn danh 5 thực thể kỹ thuật (`[CVE]`, `[IPV4]`, `[URL]`, `[FILE_PATH]`, `[HASH]`), và phân tầng đa nhãn theo tỷ lệ **70% Train - 10% Validation - 20% Test** (`MultilabelStratifiedShuffleSplit`, `random_state=42`):

```plaintext
dataset/processed/
├── enterprise-attack.json                     # Tri thức STIX MITRE ATT&CK Enterprise
│
├── cti_to_mitre/                              # Benchmark 1: CTI-to-MITRE (188 Active Classes)
│   ├── 02_processed_cti_dataset.csv           # Toàn bộ 12,944 mẫu sạch
│   ├── train.csv                              # Tập train gốc (70% - 9,060 mẫu)
│   ├── val.csv                                # Tập validation (10% - 1,295 mẫu)
│   ├── test.csv                               # Tập test độc lập (20% - 2,599 mẫu)
│   ├── multilabel_binarizer.pkl               # Binarizer 188 classes
│   └── train_augmented_eda.csv                # Tập train sau Cyber EDA (~13,600+ mẫu)
│
├── tram/                                      # Benchmark 2: TRAM (50 Active Classes)
│   ├── 02_processed_cti_dataset.csv           # Toàn bộ 8,508 mẫu sạch (sau lọc outlier <= 3 nhãn)
│   ├── train.csv                              # Tập train gốc (70% - 5,955 mẫu)
│   ├── val.csv                                # Tập validation (10% - 851 mẫu)
│   ├── test.csv                               # Tập test độc lập (20% - 1,705 mẫu)
│   ├── multilabel_binarizer.pkl               # Binarizer 50 classes
│   └── train_augmented_eda.csv                # Tập train sau Cyber EDA (~8,200+ mẫu)
│
└── joint/                                     # Benchmark Chính: JOINT (188 Active Classes)
    ├── 02_processed_cti_dataset.csv           # Toàn bộ 21,452 mẫu hợp nhất
    ├── train.csv                              # Tập train gốc (70% - 15,016 mẫu)
    ├── val.csv                                # Tập validation (10% - 2,145 mẫu)
    ├── test.csv                               # Tập test độc lập (20% - 4,291 mẫu)
    ├── multilabel_binarizer.pkl               # Binarizer 188 classes
    └── train_augmented_eda.csv                # Tập train sau Cyber EDA (~20,500+ mẫu)
```

### 📋 Cấu Trúc Các Cột Dữ Liệu (Schema)
* **`Cleaned_Text`**: Văn bản CTI đã chuẩn hóa, giữ cấu trúc câu ngữ pháp, đã ẩn danh thực thể $\rightarrow$ *Dành cho Transformer/SecBERT/RoBERTa/ModernBERT*.
* **`Tokenized_Text`**: Chuỗi token viết thường bảo toàn ký tự kỹ thuật (`.`, `/`, `:`, `-`, `_`) $\rightarrow$ *Dành cho TF-IDF Baselines*.
* **`Labels`**: Danh sách mã Parent Technique MITRE ATT&CK (`Txxxx`), ví dụ: `T1003,T1557`.
* **`Label_Count`**: Số lượng nhãn của mẫu ($\le 3$).
* **`source_sample_id`**: ID định danh duy nhất của mẫu để truy vết phân tích lỗi.
* **`is_augmented`**: `0` (mẫu gốc ban đầu), `1` (mẫu sinh mới qua Augmentation).

---

## ⚡ Cấu Hình Tăng Cường Dữ Liệu Tối Ưu (Winning Cyber EDA Config_4)

Qua quá trình Hyperparameter Search & Validation Tuning, hệ thống sử dụng cấu hình **`Config_4 (High Random Swap)`** làm chuẩn mực cố định (**Frozen Cyber EDA**):

| Phép Biến Đổi | Tham Số | Tỷ Lệ Tối Ưu | Cơ Chế Hoạt Động |
| :--- | :--- | :---: | :--- |
| **Synonym Replacement (SR)** | `alpha_sr` | **0.10** | Thay thế từ bằng từ đồng nghĩa An ninh mạng STIX. |
| **Random Insertion (RI)** | `alpha_ri` | **0.05** | Chèn từ đồng nghĩa hợp lệ vào vị trí ngẫu nhiên. |
| **Random Swap (RS)** | `alpha_rs` | **0.15** | Hoán đổi vị trí từ có **bảo toàn dấu câu cuối câu** (`.`, `,`, `;`, `!`, `?`). |
| **Random Deletion (RD)** | `p_rd` | **0.05** | Xóa từ ngẫu nhiên (không xóa các từ trong Protected Set). |

* **Mốc cân bằng nhãn hiếm (Target Count):** Tự động tính toán theo giá trị trung bình thực nghiệm (**Empirical MEAN**) của từng tập:
  * `cti_to_mitre`: Bù đắp các nhãn có $< 48$ mẫu $\rightarrow$ tối thiểu **48 mẫu/nhãn**.
  * `tram`: Bù đắp các nhãn $< 119$ mẫu $\rightarrow$ tối thiểu **119 mẫu/nhãn**.
  * `joint`: Bù đắp các nhãn $< 80$ mẫu $\rightarrow$ tối thiểu **80 mẫu/nhãn**.

---

## 🎯 Kế Hoạch Thực Nghiệm Đối Chứng: Team Ablation Plan V2 (Single Seed)

### 1. Nguyên Tắc Kiểm Soát Biến Thiên (Controlled Protocol)
Để đảm bảo tính nghiêm ngặt trong bài báo khoa học, toàn bộ các kịch bản đối chứng sử dụng **cố định `Seed 42`**:
* **Cùng bộ phân chia**: `train.csv` (70%), `val.csv` (10%), `test.csv` (20%).
* **Cùng file Cyber EDA đóng băng**: `train_augmented_eda.csv`.
* **Cùng siêu tham số mô hình, hàm mất mát (ASL), learning rate, epochs**.
* **Cùng giao thức tối ưu ngưỡng xác suất trên tập Validation**.

### 2. Ma Trận 5 Kịch Bản Đối Chứng & Phân Công Nhiệm Vụ

| ID | Notebook | Người Phụ Trách | Stage 1 (Huấn luyện) | Stage 2 (Fine-tuning) | Decision Rule (Ra quyết định) | Mục Tiêu Đối Chứng Chính |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **A0** | [`NCKH_A0_seed42.ipynb`](notebooks/NCKH_A0_seed42.ipynb) | **Khoa** | Original Train Only | Không | Global Threshold | **Baseline đối chứng gốc** |
| **G0** | [`NCKH_G0_seed42.ipynb`](notebooks/NCKH_G0_seed42.ipynb) | **Quy** | Original + Generic WordNet EDA | Không | Global Threshold | **So sánh: G0 vs A0 / B1** (Đo hiệu quả Cyber EDA vs Generic EDA) |
| **B1** | [`NCKH_B1_seed42.ipynb`](notebooks/NCKH_B1_seed42.ipynb) | **Trường** *(Lead)* | Original + Cyber EDA (`Config_4`) | Không | Global Threshold | **So sánh: B1 vs A0 / G0** (Chứng minh sức mạnh Cyber EDA 1 giai đoạn) |
| **G1** | [`NCKH_G1_seed42.ipynb`](notebooks/NCKH_G1_seed42.ipynb) | **Hiếu** | Original + Generic WordNet EDA | Original-only Fine-tune | Global + Minority-F2 Dynamic Threshold | **So sánh: G1 vs E1** (Generic 2-stage vs Proposed 2-stage) |
| **B2_E1** | [`NCKH_B2_E1_seed42.ipynb`](notebooks/NCKH_B2_E1_seed42.ipynb) | **P.Anh** | Original + Cyber EDA (`Config_4`) | Original-only Fine-tune | B2 Global + E1 Minority-F2 Dynamic Threshold | **Đề xuất SOTA: B1 $\rightarrow$ B2 $\rightarrow$ E1** (Đo bước nhảy hiệu năng toàn diện) |

### 3. Nhiệm Vụ Bổ Sung: Sinh Generic EDA (Quy + Khoa)
* **Người thực hiện:** **Quy + Khoa** phối hợp.
* **Mục tiêu:** Sinh file **Generic WordNet EDA** (`train_augmented_generic_eda_sr.csv`) từ `train.csv` cho 3 benchmark (`joint`, `cti_to_mitre`, `tram`).
* **Yêu cầu kỹ thuật:** Sử dụng WordNet/NLTK thông thường (không dùng STIX Cyber Knowledge Base), giữ nguyên cùng số lượng mẫu và quy tắc minority schedule như Cyber EDA để đảm bảo tính công bằng khi so sánh (`G0 vs B1`, `G1 vs E1`).

---

## 🚀 Hướng Dẫn Chạy Toàn Bộ Data Pipeline

### Bước 1: Gộp & Khử trùng lặp dữ liệu thô
```powershell
# Gộp cả 3 tập dữ liệu:
python src/merge_datasets.py --target_dataset all
```

### Bước 2: Tiền xử lý & Phân tầng 70/10/20
```powershell
# Tiền xử lý, ẩn danh thực thể & chia Stratified 70/10/20:
python src/preprocessing.py --target_dataset all --train_size 0.70 --val_size 0.10 --test_size 0.20
```

### Bước 3: Tăng cường dữ liệu Cyber EDA (Config_4)
```powershell
# Chạy Cyber EDA cho từng tập:
python src/augmentation.py --train_file dataset/processed/cti_to_mitre/train.csv --mode eda --save_csv
python src/augmentation.py --train_file dataset/processed/tram/train.csv --mode eda --save_csv
python src/augmentation.py --train_file dataset/processed/joint/train.csv --mode eda --save_csv
```

### Bước 4: Chạy Thực Nghiệm trên Notebooks
1. Mở notebook được phân công trong thư mục `notebooks/` (`NCKH_A0_seed42.ipynb`, `NCKH_B1_seed42.ipynb`, v.v.).
2. Chọn benchmark cần chạy (mặc định: `DATASET_SUBSET = "joint"`).
3. Chạy `Run All` để huấn luyện và tự động xuất kết quả đánh giá ra thư mục `results/`.

---

## 📊 Bộ Chỉ Số Đánh Giá Chuẩn Quốc Tế (Evaluation Metrics)

Mọi kịch bản đều được đánh giá trên tập **`test.csv` (20%)** độc lập thông qua bộ chỉ số chuẩn:
1. **Macro F1**: Chỉ số chính đánh giá mức độ công bằng và chất lượng phân loại trên các nhãn hiếm (Tail Classes).
2. **Micro F1 & Weighted F1**: Đánh giá tổng thể hiệu năng trên toàn bộ tập dữ liệu.
3. **Precision (Macro) & Recall (Macro)**.
4. **Hamming Loss & Exact Match Accuracy (Subset Accuracy)**.
5. **Recall@1, Recall@3, Recall@5, Recall@10**: Tỷ lệ nhãn thực tế nằm trong Top-k kỹ thuật được mô hình xếp hạng cao nhất.
6. **4-Tier Frequency Breakdown**: Phân tích lỗi theo 4 tầng tần suất:
   * **Head Tier** ($>500$ mẫu)
   * **Major Tier** ($100 - 499$ mẫu)
   * **Medium Tier** ($30 - 99$ mẫu)
   * **Tail Tier** ($<30$ mẫu - Nhóm 101 nhãn hiếm).

---

## 👥 Phân Công Trách Nhiệm Thành Viên (Team Members)

* **Trường (Team Lead):** Kiến trúc hệ thống, Module Cyber EDA & Data Pipeline, Kịch bản **B1** (`NCKH_B1_seed42.ipynb`).
* **Khoa:** Xây dựng Baseline, Module TextCNN, Kịch bản **A0** (`NCKH_A0_seed42.ipynb`), Phối hợp sinh Generic EDA.
* **Quy:** Kiến trúc Bi-Encoder Retrieval, Kịch bản **G0** (`NCKH_G0_seed42.ipynb`), Phối hợp sinh Generic EDA.
* **Hiếu:** Thí nghiệm mô hình Transformer 2 giai đoạn, Kịch bản **G1** (`NCKH_G1_seed42.ipynb`).
* **P.Anh:** Hàm mất mát Asymmetric Loss (ASL), Fine-tuning SecureBERT 2.0, Kịch bản **B2_E1** (`NCKH_B2_E1_seed42.ipynb`).

---

## 📜 Quy Tắc Đồng Thuận Trong Nhóm (Team Governance)

> [!IMPORTANT]
> **Tuyệt đối không tự ý thay đổi:**
> 1. Random seed huấn luyện (`seed = 42`).
> 2. Phân chia tập dữ liệu (`train.csv`, `val.csv`, `test.csv` tỷ lệ 70/10/20).
> 3. File tăng cường dữ liệu Cyber EDA (`train_augmented_eda.csv`).
> 4. Tham số hàm mất mát ASL và protocol tối ưu threshold trên tập Validation.
> Mọi thay đổi về siêu tham số cần được thảo luận và thống nhất chung trong nhóm để đảm bảo tính nhất quán của ma trận đối chứng trong bài báo.

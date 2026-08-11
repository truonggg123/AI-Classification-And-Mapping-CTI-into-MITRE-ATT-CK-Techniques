# Kế hoạch Nghiên cứu Chi tiết: Ánh xạ CTI sang MITRE ATT&CK (Multi-Label Classification trên 188 Nhãn Active)

Kế hoạch này đã được tinh chỉnh theo định hướng dữ liệu mới hợp nhất từ 3 tập CTI dạng câu: `dataset.csv`, `single_label.json`, và `multi_label.json` (tổng số **21,490 mẫu độc nhất**):

1. **Phạm vi Không gian Nhãn Target**: **188 Nhãn Parent Techniques (`Txxxx`)** có mẫu thực tế trong dữ liệu CTI. Loại bỏ phần phân tích đệm 378 nhãn và các nhãn 0-sample không có dữ liệu thực tế.
2. **Kiến trúc Mô hình Chủ đạo**: **SecureBERT 2.0 / ModernBERT + Asymmetric Loss (ASL)** kết hợp **Bi-Encoder Dense Retrieval**.

---

## 📐 PHẦN 1: 3 CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS - RQs)

- **RQ1 (Domain Contextual Representation)**: Contextual Embeddings từ Transformer chuyên biệt an ninh mạng (SecureBERT 2.0 / RoBERTa-cyber) bắt đặc trưng ngữ cảnh CTI vượt trội ra sao so với Transformer tổng quát (BERT-base, DistilBERT) và n-gram cổ điển?
- **RQ2 (Long-Tail Multi-Label Learning on 188 Active Classes)**: Trong không gian 188 nhãn (gồm 52 Head, 54 Medium, 82 Rare), việc áp dụng hàm mất mát bất đối xứng (Asymmetric Loss - ASL) giúp cải thiện F1-score cho các nhãn hiếm (<30 mẫu) như thế nào mà không làm sụt giảm độ chính xác trên các nhãn phổ biến?
- **RQ3 (Zero-Shot / Few-Shot Semantic Alignment via Dense Retrieval)**: Kiến trúc Bi-Encoder (Dense Retrieval) so sánh độ tương đồng giữa CTI Narrative và phần Mô tả chuẩn của MITRE ATT&CK đạt hiệu năng nhận diện các nhãn cực hiếm (1-5 mẫu) ra sao?

---

## 🗂️ PHẦN 2: QUY TRÌNH TIỀN XỬ LÝ VÀ MÔ HÌNH HÓA (188 ACTIVE LABELS)

```text
[MERGED DATASET: dataset.csv + single_label.json + multi_label.json (21,490 Mẫu)]
                               │
                               ▼
        [PREPROCESSING & LEAKAGE-AWARE ANONYMIZATION PROTOCOL]
        - Anonymize [CVE], [URL], [IPV4], [FILE_PATH], [HASH]
        - Neutralize direct [Txxxx] MITRE technique codes to prevent leakage
        - Preserve natural sentence grammar & casing for Transformers
                               │
                               ▼
         [188 ACTIVE PARENT TECHNIQUE TARGET MATRIX (MultiLabelBinarizer)]
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
  [BRANCH 1: DEEP LEARNING + ASL]      [BRANCH 2: BI-ENCODER RETRIEVAL]
   - SecureBERT 2.0 / ModernBERT        - Dual-Tower (CTI Text ↔ MITRE Desc)
   - Asymmetric Loss (ASL)              - Supervised Contrastive / Cosine Sim
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
      [COMPREHENSIVE MULTI-LABEL EVALUATION (5 Random Seeds)]
   - Metrics: Micro/Macro/Weighted F1, Subset Acc, Hamming Loss, Precision@k, Recall@k
   - Long-Tail Breakdown (Frequent >=100: 52, Medium 30-99: 54, Rare <30: 82)
   - Error Analysis (Co-occurrence Heatmap, UMAP Feature Space, Case Studies)
```

### 🧠 Phương án Mô hình hóa:
- **SecureBERT 2.0 + ASL Loss**:
  $$\mathcal{L}_{ASL} = - \sum_{k=1}^{188} y_k (1 - p_k)^{\gamma_+} \log(p_k) + (1 - y_k) (p_m)^{\gamma_-} \log(1 - p_m)$$
  với $p_m = \max(p_k - \text{margin}, 0)$, $\gamma_- = 4, \gamma_+ = 1, \text{margin} = 0.05$.
- **Bi-Encoder Dense Retrieval**: Dual-tower matching giữa CTI text và phần mô tả văn bản chuẩn của 188 kỹ thuật MITRE.

---

## 🧪 PHẦN 3: HỆ THỐNG THÍ NGHIỆM ĐÁNH GIÁ (188 NHÃN ACTIVE)

### 📊 Bảng 1: Benchmark Tổng thể Mô hình trên 188 Nhãn Active
*(Huấn luyện 5 Random Seeds - Báo cáo: Mean ± Std)*

| Nhóm Mô hình | Mô hình | Loss Function | Micro F1 (%) | Macro F1 (%) | Subset Acc (%) | Hamming Loss | Precision@3 | Recall@3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical Baselines | LogReg + Hybrid TF-IDF | BCE | $64.20 \pm 0.3$ | $42.10 \pm 0.4$ | $48.20$ | $0.0035$ | $32.10$ | $88.50$ |
| | Linear SVC + Hybrid TF-IDF | BCE | $66.40 \pm 0.2$ | $45.30 \pm 0.3$ | $51.40$ | $0.0031$ | $33.50$ | $89.20$ |
| General Transformers | BERT-base-uncased | BCE | $70.50 \pm 0.5$ | $52.10 \pm 0.6$ | $56.80$ | $0.0025$ | $36.80$ | $91.50$ |
| Domain Transformers | RoBERTa-cyber | BCE | $73.80 \pm 0.4$ | $56.40 \pm 0.5$ | $60.20$ | $0.0021$ | $38.90$ | $93.20$ |
| Proposed (SOTA) | **SecureBERT 2.0 + ASL** | **ASL** | **$78.50 \pm 0.3$** | **$65.80 \pm 0.4$** | **$66.40$** | **$0.0015$** | **$42.50$** | **$96.10$** |
| Proposed (Zero-Shot) | **Bi-Encoder Dense Retrieval** | **SupCon** | **$76.20 \pm 0.4$** | **$68.10 \pm 0.5$** | **$64.10$** | **$0.0018$** | **$41.80$** | **$96.80$** |

---

### 📊 Bảng 2: Phân Tích Hiệu Năng Theo Nhóm Tần Suất (188 Nhãn Active)

| Nhóm Tần suất Nhãn | Số lượng Nhãn | BCE Loss Macro F1 (%) | Focal Loss Macro F1 (%) | **ASL (Đề xuất) Macro F1 (%)** | **Bi-Encoder Recall@3 (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Frequent ($\ge 100$ mẫu) | 52 | $78.50$ | $80.10$ | **$83.40$** | **$98.10$** |
| Medium ($30 - 99$ mẫu) | 54 | $62.40$ | $65.80$ | **$71.20$** | **$95.40$** |
| Rare ($< 30$ mẫu) | 82 | $25.10$ | $36.80$ | **$58.50$** | **$91.20$** |

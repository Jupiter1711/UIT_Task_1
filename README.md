# UIT DSC 2026 — Task 1 · LegalIR

Truy hồi văn bản pháp luật tiếng Việt. Đầu vào một câu hỏi, đầu ra **tối đa 5 `doc_id`**.
Độ đo chính là **Recall**, Precision chỉ dùng khi hoà Recall.

| | warmup (500 câu) | Codabench public (1.000 câu) |
|---|---:|---:|
| **Kết quả hiện tại** | **0.9275** | **0.9308** |
| Trần rổ ứng viên | 0.9593 | — |

Kiến trúc: BM25 + dense retrieval → RRF fusion (k=60) → cross-encoder reranker đã fine-tune.

Repo này chỉ có **2 notebook** — vừa đủ để chạy lại kết quả trên. Toàn bộ logic nằm trong
notebook, không import module ngoài nào.

```
notebooks/
├── baseline.ipynb            PIPELINE CHÍNH — 11 cell, Kaggle GPU T4 x2
└── finetune_reranker.ipynb   fine-tune cross-encoder, 8 cell, Kaggle GPU T4 x2
```

---

## 0. Đọc trước khi chạm vào bất cứ thứ gì

⛔ **RÀNG BUỘC CỨNG của BTC.** Vi phạm là mất điểm hoặc bị loại:

1. **Tổng tham số CẢ hệ thống < 4 tỉ** — cộng gộp embedding + reranker + mọi model khác,
   **không phải** giới hạn cho từng model. Hiện dùng 2 × 0,57B = **1,14B / 4B**.
2. **Quantization / LoRA / GPTQ / AWQ / GGUF KHÔNG làm giảm** số tham số tính theo luật.
3. **Cấm mọi API**, thương mại lẫn phi thương mại. Chỉ model mã nguồn mở chạy local.
4. **Chỉ dùng model đã đăng ký và được BTC duyệt.** Thêm hay đổi model phải đăng ký lại.
5. **Chỉ dùng data BTC cấp.** Cấm thu thập data ngoài, gán nhãn thủ công, augment nguồn ngoài.
   ✅ Fine-tune trên `train.json` thì **được phép** — đó là hướng cải tiến chính của bài này.
6. Nộp qua Organization đã đăng ký trên Codabench, **tối đa 10 lượt/ngày**.

Model đang dùng, cả hai đều nằm trong danh sách BTC duyệt:
`AITeamVN/Vietnamese_Embedding_v2` (~0,57B) + `AITeamVN/Vietnamese_Reranker` (~0,57B, đã fine-tune).

⚠️ **Bẫy của bộ chấm** (`data/Scoring-Program-Task-LegalIR/scoring.py`):
`len(pred) > 5` **hoặc** `pred` rỗng → **Recall = Precision = 0** cho câu đó. Số câu trong
submission phải khớp đúng số câu reference, lệch là scorer raise Exception.
⇒ Chiến lược đã chốt: **luôn trả đúng 5 `doc_id`, dạng string.**

---

## 1. Những thứ KHÔNG có trong repo

Repo chỉ chứa code. Bốn nhóm file phải lấy riêng:

| Cần | Lấy ở đâu | Dung lượng | Bắt buộc? |
|---|---|---|---|
| `data/` — corpus + train/warmup/public + scoring program | BTC cấp | ~1 GB | ✅ luôn cần |
| `reranker_ft` — reranker đã fine-tune | xin trực tiếp | ~2,2 GB | Cách A |
| `train_pairs_pool_n10.jsonl.gz` — dữ liệu huấn luyện | xin trực tiếp | 22 MB | Cách B |
| `chunk_emb_L512_w220o40.npy` — cache embedding | xin trực tiếp, hoặc để notebook tự sinh | ~2,2 GB | không |

Cache embedding chỉ để tiết kiệm thời gian: có thì bỏ qua được **1,5–2 giờ GPU** mỗi lần chạy.

### Cấu trúc `data/` phải đúng như sau
```
data/
├── selected-contexts/           8.532 file context_*.json  {link, name, passage, id}
├── train.json                   7.000 câu   {qid: {question, answer: [doc_id...]}}
├── warmup_IR.json                 500 câu   CÓ gold, dùng làm held-out
├── public-official.json         1.000 câu   answer = null, đây là tập phải nộp
└── Scoring-Program-Task-LegalIR/scoring.py
```

---

## 2. Chuẩn bị Kaggle (làm một lần)

Tạo **Dataset** tên tuỳ ý, ví dụ `uit-dsc`, upload nguyên `data/` vào. Notebook không
hardcode đường dẫn — nó `glob("/kaggle/input/**/<tên file>")`, nên đặt thư mục con thế nào
cũng được, miễn **giữ nguyên tên file**:

```
uit-dsc/
├── selected-contexts/context_*.json
├── train.json
├── warmup_IR.json
└── public-official.json
```

Trong notebook: **Settings → Accelerator → GPU T4 x2**, **Internet ON**
(cell 0 cài package, cell 5 và 7 tải model từ HuggingFace).

---

## 3. Chạy

### Cách A — đã có reranker fine-tune (~2 giờ, khuyến nghị)

1. Tạo Dataset thứ hai chứa model, ví dụ `reranker-pool-n10`: các file `config.json`,
   `model.safetensors`, `tokenizer.json`, `sentencepiece.bpe.model`… Đặt thẳng ở gốc
   dataset cũng được, cell 1 dò theo `model.safetensors`.
2. *(Tuỳ chọn)* Dataset thứ ba chứa `chunk_emb_L512_w220o40.npy`.
   **Tên file phải khớp chính xác** — cell 5 dò theo tên gắn với cấu hình.
3. Notebook mới → File → Import Notebook → `notebooks/baseline.ipynb`.
4. Add Input: `uit-dsc` + `reranker-pool-n10` (+ dataset cache nếu có).
5. **Run All.**

Kết quả trong `/kaggle/working`: **`submission.zip`** để nộp Codabench, kèm
`warmup_eval_*.json` và `warmup_pred_*.json`.

> **Nhanh hơn ~20 phút: chạy cell 0–8 rồi bỏ qua cell 9, chạy thẳng cell 10.**
> Cell 9 là chẩn đoán, chỉ dump điểm reranker của từng chunk để phân tích lỗi về sau,
> **không ảnh hưởng gì tới `submission.zip`**. "Run All" thì nó vẫn chạy.

> ⛔ **Chỉ được có ĐÚNG MỘT model trong Input.** Cell 1 `raise` ngay nếu thấy nhiều hơn.
> Chốt chặn này có vì trước đây `sorted(...)[0]` bốc nhầm model **mà không báo lỗi gì** —
> chạy hết 3 tiếng xong mới phát hiện đã đo nhầm bản.

### Cách B — tự fine-tune reranker (~3,5 giờ, 2 lần chạy)

Dùng khi có `train_pairs_pool_n10.jsonl.gz` nhưng chưa có model.

**Lần 1 — fine-tune (~1,5 giờ).**
- Tạo Dataset chứa `train_pairs_pool_n10.jsonl.gz`.
  Kaggle tự giải nén file `.gz` khi upload; notebook chấp nhận cả `.jsonl` lẫn `.jsonl.gz`.
- Import `notebooks/finetune_reranker.ipynb`, Add Input: `uit-dsc` + dataset vừa tạo.
- **Sửa `EPOCHS = 2` thành `EPOCHS = 1`.** Cross-encoder này overfit ở epoch 2 trong
  **cả hai** lần train đã thực hiện (dev acc@1 tụt 0.8855→0.8825 và 0.8541→0.8421).
  Vòng lặp chỉ lưu checkpoint tốt nhất nên epoch 2 không làm hỏng gì, chỉ tốn ~95 phút.
- Run All → `/kaggle/working/reranker_ft`.
- **Quick Save** (KHÔNG dùng "Save & Run All" — nó chạy lại từ đầu).

**Lần 2 — đo và nộp (~2 giờ).** Y như Cách A, nhưng Add Input là **output của notebook
lần 1** (Add Input → Your Work → Notebook Output) thay vì dataset model.

---

## 4. Kiểm tra xem có chạy đúng không

Đối chiếu đúng 5 mốc này, theo thứ tự xuất hiện trong log:

| Cell | Dòng in ra | Giá trị đúng |
|---|---|---|
| 1 | `RERANK_MODEL:` | phải trỏ vào `reranker_ft`, **không** phải `AITeamVN/Vietnamese_Reranker` |
| 2 | `8532 văn bản` | **8.532** |
| 3 | `... chunk từ 8532 docs` | **529.877** chunk |
| 8 | `Recall@5 ...` | **0.9275** |
| 8 | `tran ro ...` | **0.9593**, và `nhat duoc` **96.7%** |

Nếu `RERANK_MODEL` in ra tên HuggingFace gốc thì đang chạy reranker **chưa** fine-tune —
kết quả sẽ ra ~0.84 chứ không phải 0.9275. Cell 1 có in cảnh báo ở trường hợp này.

Cell 1 đặt `EVAL_SET = "both"` nên cell 8 chạy trên cả `warmup` lẫn `dev665`.
**Không có `dev665.json` thì cell 1 in một dòng ghi chú rồi chỉ chạy `warmup` — vô hại**,
`dev665` là tập đo nội bộ, không liên quan tới submission.

**Trước khi nộp**, kiểm tra `submission.json`: đủ **1.000 câu**, đúng bộ qid của
`public-official.json`, mỗi câu **đúng 5** `doc_id` **dạng string**.

---

## 5. Những cái bẫy đã trả giá rồi — đừng giẫm lại

| Bẫy | Chuyện gì đã xảy ra | Cách tránh |
|---|---|---|
| `/kaggle/working` **xoá sạch khi session chết** | suýt mất 3 giờ train | Quick Save ngay sau khi train xong |
| "Save & Run All" | chạy lại từ đầu, không phải lưu nhanh | dùng **Quick Save** |
| Nhiều model trong Input | `sorted(...)[0]` bốc nhầm, **không báo lỗi gì** | cell 1 giờ `raise`; luôn đọc dòng `RERANK_MODEL:` |
| Cache `.npy` tên cố định | đổi hyperparameter xong vẫn nạp embedding CŨ, không dấu hiệu gì | tên cache gắn theo cấu hình |
| SentenceTransformer chạy fp32 | embed 530k chunk ước tính **8 giờ** | `torch_dtype=float16` → còn 1,5–2 giờ |
| OOM ở `reduce_add_coalesced`, đúng 978 MiB | bảng embedding XLM-R = 250.002×1024, gradient fp32 = 977 MiB, DataParallel gom hết về GPU 0 | `FREEZE_EMB=True` |
| OOM lần 2 ở `from_pretrained` | **kernel chưa restart**, 10,6 GB đã bị chiếm sẵn | Restart Session trước khi train |
| `!pip install pkg>=3.0.0` không có nháy | shell hiểu `>` là chuyển hướng stdout vào file tên `=3.0.0` → cài bản bất kỳ, nuốt luôn output | luôn `!pip install -q "pkg>=3.0.0"` |
| **69,2% câu warmup nằm trong `train.json`** | train thẳng lên `train.json` là đã train lên 69% warmup → điểm warmup lạc quan giả | dữ liệu huấn luyện đã **loại sẵn 348 câu đó**; đừng tự sinh lại mà bỏ bước này |

---

## 6. Ghi chú về dữ liệu (đã kiểm, đừng mất công đo lại)

- `passage` rất dài: median **4.813 từ**, p90 18.406, max **1.242.409 từ**. Doc gold còn dài
  hơn: median 14.831 từ, median **72 "Điều"**/doc ⇒ phải tìm 1 chunk đúng trong ~72 chunk.
  Đó là lý do phải chunk theo "Điều" — riêng bước này đã **nhân đôi Recall** (0.37 → 0.74).
- **20 doc rỗng cả `passage` lẫn `name`** — gold của 11 câu train + 1 câu warmup,
  **không thể truy hồi được**. Đó là trần cứng, không phải lỗi pipeline.
- **40 cụm doc gần trùng** (Jaccard ≥ 0,8). ⚠️ **KHÔNG được dedup xoá bớt** — 8 cụm có
  nhiều doc đều từng là gold cho các câu hỏi khác nhau, xoá là mất gold vĩnh viễn.
- `name` chứa số hiệu văn bản, nhưng **chỉ 1,1% câu hỏi** nhắc số hiệu dạng `xx/yyyy`
  ⇒ đừng đầu tư vào matching số hiệu, đã thử và loại.
- `public-official.json` ∩ `warmup_IR.json` = **53 câu**. Là data BTC cấp nên tra gold không
  vi phạm §5 của luật, nhưng đây là **quyết định của cả nhóm**, hiện chưa dùng.

# MiniLLM-2B

MiniLLM adalah proyek pembelajaran untuk membangun decoder-only Transformer dari ukuran **tiny** sampai kelas **2B parameter** memakai Python dan PyTorch. Repository ini menyediakan fondasi yang benar-benar dapat dijalankan: Byte-level BPE tokenizer, pembersihan data, GQA + RoPE + RMSNorm + SwiGLU, causal-language-model training, checkpoint/resume, sampling, chat template, CLI, REST API, Docker, dan unit test.

> Mulai dari `model_tiny.json`. Jangan langsung menjalankan konfigurasi 2B sebelum seluruh test, overfit test, data pipeline, dan checkpoint/resume terbukti benar.

## Status implementasi

| Area | Status v0.1 | Catatan |
|---|---:|---|
| Byte-level BPE tokenizer | ✅ | Token khusus system/user/assistant |
| Data cleaning + exact dedup | ✅ | Input TXT, JSON, JSONL |
| Transformer decoder | ✅ | RoPE, RMSNorm, GQA, SwiGLU, SDPA causal |
| Training dasar | ✅ | AdamW, warmup-cosine, accumulation, clipping |
| Atomic checkpoint + resume | ✅ | Model, optimizer, scheduler, RNG, step, token |
| Sampling dan chat CLI | ✅ | Greedy, temperature, top-k, top-p |
| REST API | ✅ | Health + OpenAI-style chat completions |
| Docker | ✅ | Bobot di-mount dari luar image |
| Distributed/FSDP | Direncanakan | Phase scaling setelah single-device stabil |
| SFT/LoRA | Direncanakan | Assistant-only loss masking diperlukan |
| KV cache native | Direncanakan | Implementasi awal menghitung ulang konteks |
| INT8/INT4/GGUF | Direncanakan | Harus disertai uji regresi kualitas |

## Arsitektur

Aliran utama:

```text
text -> tokenizer -> token IDs -> embedding -> N x Transformer block
     -> final RMSNorm -> LM head -> logits -> sampling -> next token
```

Setiap block memakai pre-normalization:

```text
h = x + GQA(RMSNorm(x))
y = h + SwiGLU(RMSNorm(h))
```

Konfigurasi `configs/model_2b.json` mengikuti blueprint awal (hidden 2560, 30 layer, 20 query head, 5 KV head, FFN 6912). Karena input embedding dan LM head **tidak diikat**, hitungan aktualnya sekitar **2,25B parameter**, bukan tepat 2,00B. Gunakan skrip penghitung parameter sebelum memulai training besar.

## Instalasi

Python 3.10+ diperlukan.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .
```

Untuk GPU NVIDIA, pasang build PyTorch yang sesuai dengan CUDA dari situs resmi PyTorch, lalu jalankan `python -c "import torch; print(torch.cuda.is_available())"`.

## Quick start aman

### 1. Siapkan teks legal

Buat `data/raw/sample.txt` dari teks milik sendiri, public domain, atau dataset yang lisensinya mengizinkan training. Jangan masukkan credential, nomor identitas, data pribadi, atau benchmark evaluasi.

```bash
python scripts/prepare_data.py \
  --input data/raw \
  --output data/clean/documents.jsonl
```

### 2. Latih tokenizer tiny

Untuk smoke test, samakan vocab tokenizer dengan `configs/model_tiny.json`, yaitu 512:

```bash
python scripts/train_tokenizer.py \
  --input data/raw \
  --vocab-size 512 \
  --output tokenizer/artifacts/tokenizer.json
```

Byte-level BPE mempertahankan kemampuan merepresentasikan byte sehingga teks Unicode tidak bergantung pada satu token `<|unk|>` saja. Kualitas pemotongan tetap perlu diuji pada Bahasa Indonesia, English, code, angka, URL, JSON, emoji, newline, dan tab.

### 3. Jalankan test

```bash
pytest -q
```

### 4. Overfit dataset kecil

`scripts/pretrain.py` membaca teks biasa. Untuk eksperimen pertama, gunakan corpus berulang yang cukup panjang agar menghasilkan minimal satu batch penuh.

```bash
python scripts/pretrain.py \
  --model configs/model_tiny.json \
  --train configs/pretrain_debug.json \
  --tokenizer tokenizer/artifacts/tokenizer.json \
  --data data/raw/sample.txt \
  --output checkpoints/tiny
```

Loss harus turun pada corpus kecil. Bila tidak, periksa label shift, causal mask, tokenizer, learning rate, sequence length, dan ukuran dataset sebelum menaikkan model.

### 5. Resume training

```bash
python scripts/pretrain.py \
  --model configs/model_tiny.json \
  --train configs/pretrain_debug.json \
  --tokenizer tokenizer/artifacts/tokenizer.json \
  --data data/raw/sample.txt \
  --output checkpoints/tiny \
  --resume checkpoints/tiny/step_00000250.pt
```

### 6. Chat

Model base kecil belum otomatis pandai mengikuti instruksi. CLI ini baru memberi antarmuka chat; kualitas percakapan yang baik memerlukan pretraining memadai dan SFT.

```bash
python scripts/chat.py \
  --model-config configs/model_tiny.json \
  --checkpoint checkpoints/tiny/step_00002000.pt \
  --tokenizer tokenizer/artifacts/tokenizer.json
```

Perintah: `/clear` untuk menghapus history dan `/exit` untuk keluar.

### 7. Jalankan API

```bash
export MINILLM_CONFIG=configs/model_tiny.json
export MINILLM_CHECKPOINT=checkpoints/tiny/step_00002000.pt
export MINILLM_TOKENIZER=tokenizer/artifacts/tokenizer.json
uvicorn deployment.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Chat completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Apa itu Python?"}],"max_tokens":64}'
```

Jangan expose server langsung ke internet sebelum menambahkan autentikasi, rate limit, batas request, timeout, dan logging yang aman.

## Menghitung parameter

```bash
python scripts/count_parameters.py configs/model_tiny.json
python scripts/count_parameters.py configs/model_debug.json
python scripts/count_parameters.py configs/model_2b.json
```

Skrip menghitung parameter langsung dari dimensi konfigurasi dan tidak mengalokasikan model, sehingga aman dipakai untuk memeriksa konfigurasi 2B di laptop.

## Mendapatkan data training

Gunakan hanya sumber dengan provenance dan lisensi yang jelas. Kandidat awal:

- Wikipedia/Wikibooks/Wikisource dumps;
- buku public domain;
- dokumentasi dan materi pendidikan open-license;
- corpus Bahasa Indonesia dengan dataset card dan lisensi;
- repository code yang mempertahankan metadata lisensi;
- web corpus terbuka yang telah dikurasi dan difilter.

Semua sumber sebaiknya dinormalisasi ke JSONL:

```json
{"id":"...","source":"wikipedia_id","language":"id","license":"CC BY-SA","text":"...","metadata":{}}
```

Manifest dataset harus menyimpan sumber, versi/tanggal, lisensi, jumlah dokumen, jumlah token, tokenizer hash, konfigurasi cleaning, shard hash, serta language mixture. Pisahkan dokumen ke train/validation/test **sebelum** packing agar dokumen yang sama tidak bocor antar-split.

Pipeline produksi yang perlu ditambahkan pada fase berikutnya:

1. downloader/streaming adapter;
2. HTML dan boilerplate removal;
3. language identification dengan confidence;
4. quality score per sumber;
5. exact, paragraph, dan near-duplicate removal;
6. secret/PII filtering;
7. benchmark contamination scan;
8. tokenize, EOS boundary, packing;
9. binary shards + mmap;
10. statistik dan license manifest.

Komposisi awal yang dapat diuji: 35% English, 30% Indonesia, 15% code, 10% matematika/sains, 5% dokumentasi, dan 5% percakapan. Ini baseline eksperimen, bukan aturan permanen.

## Roadmap fase 1–10

1. **Tokenizer:** validasi round trip, Unicode, special token, dan efisiensi Indonesia/English/code.
2. **Transformer core:** shape, gradient, causal mask, RoPE, GQA, RMSNorm, SwiGLU.
3. **Mini model:** overfit 100–1000 sequence, accumulation, checkpoint/resume, generation.
4. **Data pipeline:** provenance, lisensi, cleaning, quality, dedup, secret filter, split, packing, shard.
5. **Pretraining engine:** validation, BF16/FP16, JSONL metrics, atomic retention, NaN/OOM recovery.
6. **Scaling:** DDP lalu FSDP, activation checkpointing, per-rank sampler, multi-GPU resume.
7. **Base evaluation:** perplexity, fixed prompts, Bahasa Indonesia, English, code, math, repetition.
8. **SFT/chat:** schema messages, assistant-only labels, full fine-tune dan LoRA, multi-turn/system tests.
9. **Inference/quantization:** KV cache, streaming stabil, INT8/INT4, benchmark, safe export.
10. **Deployment:** Python API, streaming REST, batching, auth/rate limit, Docker, observability.

Urutan scaling yang disarankan: tiny → 5–20M → 50M → 150M → 500M → 1B → kelas 2B. Setiap kenaikan hanya dilakukan setelah correctness, validation, dan resume lulus.

## Batasan v0.1

- Training saat ini single-process dan belum memakai BF16 autocast, DDP, atau FSDP.
- Loader awal membaca satu corpus teks ke memori; dataset skala besar memerlukan streaming binary shards.
- Generation belum memakai KV cache sehingga makin lambat ketika konteks bertambah.
- Belum ada SFT, LoRA, quantization, Hugging Face export, GGUF, atau dynamic batching.
- Checkpoint menyimpan RNG, tetapi posisi/shuffle iterator DataLoader belum dipulihkan secara bit-identical.
- Pembersihan awal hanya normalisasi dan exact dedup; jangan menganggapnya cukup untuk web-scale data.

Daftar ini sengaja eksplisit agar fitur yang belum dibuat tidak terlihat seolah sudah production-ready.

## Aturan proyek

- Jangan scale model yang belum lolos overfit test.
- Jangan menjalankan training panjang tanpa checkpoint dan resume test.
- Jangan memakai dataset tanpa provenance dan lisensi.
- Jangan mencampurkan benchmark evaluasi ke training.
- Jangan mengganti tokenizer setelah pretraining dimulai.
- Selalu ukur validation loss, bukan hanya training loss.
- Pertahankan milestone checkpoint dan bobot presisi asli setelah quantization.
- Utamakan correctness, lalu profile dan optimisasi.

## Lisensi

Kode repository menggunakan MIT License. Lisensi kode **tidak otomatis berlaku** pada dataset maupun bobot model; keduanya wajib memiliki dokumentasi lisensi sendiri.

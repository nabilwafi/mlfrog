# XAUUSD MT5 Bar Fetcher

Ambil data OHLCV **XAUUSD** dari MetaTrader 5 dengan rentang tanggal configurable, chunk per tahun (atau bulan), dan logger rotating file.

## Prasyarat

- Windows + MetaTrader 5 terinstall
- Terminal MT5 sudah login ke akun broker (atau isi kredensial di `config.yaml`)
- Python 3.10+

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml`:

```yaml
symbol: XAUUSD
timeframe: H1          # M1, M5, M15, M30, H1, H4, D1, W1, MN1

date_range:
  start_date: "2016-01-01"
  end_date: "2026-01-01"   # 10 tahun → otomatis 10 chunk per tahun

chunk:
  unit: year
  step: 1
```

## Jalankan

```bash
python fetch_bars.py
python fetch_bars.py --config config.yaml
```

Output: `./data/raw/XAUUSD_H1.csv` (atau `.parquet` jika dikonfigurasi).

Log: `./logs/xauusd_fetcher.log` (rotating, max 10 MB × 5 backup).

## Catatan

- `combine_chunks: true` → satu file gabungan (dedupe by Date).
- `combine_chunks: false` → file terpisah per chunk, mis. `XAUUSD_H1_20160101_20170101.csv`.
- Pastikan simbol di Market Watch broker kamu benar (`XAUUSD`, `XAUUSD.a`, dll.) — sesuaikan `symbol` di config.

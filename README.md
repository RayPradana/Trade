# Indodax Automated Trading Bot (Python)

Bot trading otomatis untuk Indodax menggunakan Python. Bot melakukan:

- Analisis tren koin dengan moving average cepat/lambat
- Analisis order book (spread, ketidakseimbangan volume), volume, support/resistance, dan candlestick dari data trades real-time
- Pemilihan gaya trading otomatis (scalping, day trading, swing, position) sesuai kondisi pasar
- Eksekusi limit order beli/jual otomatis dengan batasan slippage, stop-loss, take-profit (stop-limit secara logis melalui stop-loss guard)
- Risk management: target profit, batas rugi, pembatasan ukuran order berdasarkan modal/cash, dan proteksi over-sell posisi
- Mode **dry-run** bawaan agar aman untuk simulasi tanpa mengeksekusi order sungguhan
- **Auto-resume**: menyimpan state portofolio sehingga bot melanjutkan posisi terakhir jika restart
- Analisis lanjutan: RSI, MACD, Bollinger Bands untuk konfirmasi tren dan momentum
- **Staged entry**: alokasikan modal bertahap (bukan all-in) sesuai volatilitas & kepercayaan sinyal

> Gunakan dokumentasi resmi Indodax: https://github.com/btcid/indodax-official-api-docs

## Prasyarat

- Python 3.10+
- `pip install -r requirements.txt`

## Konfigurasi

Semua konfigurasi diambil dari variabel lingkungan (bisa diset di `.env`):

| Variabel | Deskripsi | Default |
| --- | --- | --- |
| `INDODAX_KEY` | API key Indodax (hanya diperlukan untuk mode live) | - |
| `TRADE_PAIR` | Pasangan dagang fallback jika pemilihan otomatis tidak menemukan kandidat | `btc_idr` |
| `BASE_ORDER_SIZE` | Ukuran order dasar (dalam aset dasar) | `0.0001` |
| `RISK_PER_TRADE` | Risiko per transaksi (0.01 = 1%) | `0.01` |
| `DRY_RUN` | `true/false` untuk simulasi (set `false` untuk mode live) | `true` |
| `RUN_ONCE` | Jalankan satu siklus lalu berhenti | `false` |
| `MIN_CONFIDENCE` | Ambang kepercayaan minimum untuk eksekusi | `0.52` |
| `INTERVAL_SECONDS` | Interval candlestick hasil agregasi trades | `300` |
| `FAST_WINDOW` | Periode MA cepat | `12` |
| `SLOW_WINDOW` | Periode MA lambat | `48` |
| `MAX_SLIPPAGE_PCT` | Batas slippage relatif | `0.001` |
| `INITIAL_CAPITAL` | Modal awal (quote currency, mis. IDR) | `1000000` |
| `TARGET_PROFIT_PCT` | Target profit relatif (0.2 = 20%) | `0.2` |
| `MAX_LOSS_PCT` | Batas kerugian relatif (0.1 = 10%) | `0.1` |
| `TRADE_PAIRS` | Daftar pasangan dipisah koma untuk discan otomatis (jika kosong, bot tarik seluruh pairs dari API) | `btc_idr` |
| `AUTO_RESUME` | `true/false` untuk mengaktifkan pemulihan state otomatis | `true` |
| `STATE_FILE` | Lokasi file state JSON untuk auto-resume | `bot_state.json` |
| `STAGED_ENTRY_STEPS` | Jumlah maksimum langkah entry bertahap (mis. 3 langkah 50/30/20%) | `3` |

## Menjalankan

### Konfigurasi via `.env`

Salin `.env.example` menjadi `.env` di root repo, lalu isi:

```
INDODAX_KEY=your_api_key
TRADE_PAIRS=btc_idr,eth_idr
DRY_RUN=true
```

### Dry-run (simulasi, aman)

```bash
RUN_ONCE=true DRY_RUN=true python main.py
```

### Mode live (eksekusi order)

```bash
DRY_RUN=false INDODAX_KEY=your_api_key python main.py
```

Opsi penting:

- `RUN_ONCE=true` menjalankan satu siklus analisa + keputusan.

## Cara Kerja Singkat

1. **Pengambilan data**: ticker, order book (depth), dan riwayat trades via API publik.
2. **Analisa**:
   - Candlestick dibangun dari trades dengan interval yang dipilih.
   - Tren dihitung memakai MA cepat/lambat.
   - Order book dihitung spread, volume bid/ask, dan imbalance.
   - Volatilitas dihitung dari imbal hasil candle.
   - Momentum & konfirmasi: **RSI**, **MACD**, **Bollinger Bands** untuk filter overbought/oversold dan kekuatan tren.
   - Support dan resistance dihitung dari harga penutupan terbaru untuk menghindari entry dekat level kritis.
   - Portfolio tracker menghitung ekuitas (cash + posisi mark-to-market) terhadap target profit / batas rugi.
3. **Pemilihan strategi**:
   - **Scalping**: spread tipis, likuiditas tinggi, volatilitas rendah.
   - **Day Trading**: tren aktif dengan volatilitas moderat.
   - **Swing Trading**: tren kuat dengan volatilitas lebih tinggi.
   - **Position Trading**: kondisi lain (lebih tenang / tren panjang).
4. **Keputusan trading**: aksi beli/jual/hold dengan stop-loss & take-profit dinamis.
5. **Eksekusi & guard**: 
   - **Dry-run**: hanya log simulasi.
   - **Live**: mengirim **limit order** `trade` ke `tapi` Indodax (butuh API key/secret). Stop-limit dijaga secara logis lewat stop-loss/take-profit dan pengecekan slippage.
   - Risk guard: ukuran order dibatasi modal/posisi yang tersedia, tidak akan oversell; portfolio guard berhenti otomatis bila target profit tercapai atau batas rugi terlampaui.
   - **Auto-resume**: state portofolio dan keputusan terakhir disimpan di `STATE_FILE` agar bot melanjutkan sesi jika terhenti mendadak.

## Keamanan & Catatan

- Pastikan `DRY_RUN=true` saat mencoba pertama kali.
- Gunakan kunci API dengan izin minimal.
- Perdagangan aset kripto berisiko tinggi; lakukan uji sendiri sebelum live.

## Testing

```bash
python -m unittest discover
```

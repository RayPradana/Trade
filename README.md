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
| `RISK_PER_TRADE` | Risiko per transaksi (0.01 = 1%); ukuran order = `RISK_PER_TRADE × INITIAL_CAPITAL / harga_koin` | `0.01` |
| `DRY_RUN` | `true/false` untuk simulasi (set `false` untuk mode live) | `true` |
| `RUN_ONCE` | Jalankan satu siklus lalu berhenti | `false` |
| `REALTIME_MODE` | Aktifkan polling cepat untuk data hampir real-time (default 1s jika aktif) | `false` |
| `MIN_CONFIDENCE` | Ambang kepercayaan minimum untuk eksekusi | `0.52` |
| `INTERVAL_SECONDS` | Interval candlestick hasil agregasi trades (default 1 detik bila `REALTIME_MODE=true`) | `300` |
| `GRID_ENABLED` | Aktifkan mode grid trading (menempatkan buy/sell bertingkat) | `false` |
| `GRID_LEVELS_PER_SIDE` | Jumlah level grid di tiap sisi harga anchor | `3` |
| `GRID_SPACING_PCT` | Jarak antar level grid (0.004 = 0.4%) | `0.004` |
| `GRID_ORDER_SIZE` | (Opsional) Ukuran order per level; default memakai `RISK_PER_TRADE × INITIAL_CAPITAL / harga` | - |
| `ORDER_QUEUE_ENABLED` | Aktifkan antrean permintaan order dengan rate limit | `true` |
| `ORDER_MIN_INTERVAL` | Jeda minimal antar permintaan order (detik) — rekomendasi 2.0 s untuk menghindari HTTP 429 | `2.0` |
| `SCAN_REQUEST_DELAY` | Jeda antar request saat scan pasangan (mencegah HTTP 429) | `0.2` |
| `ACCOUNT_INFO_CACHE_TTL` | Cache saldo akun selama N detik (0 = nonaktif) | `30` |
| `OPEN_ORDERS_CACHE_TTL` | Cache open-orders per pair selama N detik (0 = nonaktif) | `15` |
| `MAX_OPEN_POSITIONS` | Batas maksimum posisi aktif sekaligus (0 = tanpa batas) | `0` |
| `MAX_HOLD_SECONDS` | Paksa jual posisi yang sudah terbuka lebih dari N detik tanpa profit (0 = nonaktif) | `0` |
| `VOLUME_HIGH_THRESHOLD_IDR` | Volume 24-jam (IDR) ambang batas hold time adaptif (0 = nonaktif) | `0` |
| `MAX_HOLD_SECONDS_VOLUME_HIGH` | Hold time maksimum untuk pair volume tinggi (detik) | `5400` |
| `MAX_HOLD_SECONDS_VOLUME_LOW` | Hold time maksimum untuk pair volume rendah (detik) | `1800` |
| `WEBSOCKET_ENABLED` | Coba gunakan WebSocket untuk data real-time (fallback ke REST) | `true` |
| `WEBSOCKET_URL` | URL WebSocket jika ingin override | - |
| `FAST_WINDOW` | Periode MA cepat | `12` |
| `SLOW_WINDOW` | Periode MA lambat | `48` |
| `MAX_SLIPPAGE_PCT` | Batas slippage relatif | `0.001` |
| `INITIAL_CAPITAL` | Modal awal (quote currency, mis. IDR) | `1000000` |
| `TARGET_PROFIT_PCT` | Target profit relatif (0.2 = 20%) | `0.2` |
| `MAX_LOSS_PCT` | Batas kerugian relatif (0.1 = 10%) | `0.1` |
| `STAGED_ENTRY_STEPS` | Jumlah maksimum langkah entry bertahap (mis. 3 langkah 50/30/20%) | `3` |
| `POSITION_CHECK_INTERVAL` | Interval (detik) polling saat sedang memegang posisi | `60` |
| `CYCLE_SUMMARY_INTERVAL` | Cetak ringkasan performa setiap N siklus scan penuh | `10` |
| `TRADE_MODE` | `continuous` = 24/7; `single` = satu siklus beli→jual lalu berhenti | `continuous` |
| `WEBSOCKET_SUBSCRIBE_MESSAGE` | Pesan JSON yang dikirim ke server WebSocket setelah koneksi (jika diperlukan) | - |

## Menjalankan

### Konfigurasi via `.env`

Salin `.env.example` menjadi `.env` di root repo, lalu isi:

```
INDODAX_KEY=your_api_key
DRY_RUN=true
```

### Mode 24/7 Otomatis (default)

Bot berjalan terus-menerus, memindai pair terbaik, masuk posisi, memantau, keluar, lalu mencari peluang berikutnya.

```bash
TRADE_MODE=continuous DRY_RUN=true python main.py
```

### Mode Single-Trade (satu siklus beli → jual lalu berhenti)

Berguna untuk otomasi / cron job — bot melakukan tepat satu transaksi lengkap (beli + jual) lalu keluar.

```bash
TRADE_MODE=single DRY_RUN=true python main.py
```

### Dry-run (simulasi, aman)

```bash
RUN_ONCE=true DRY_RUN=true python main.py
```

### Mode live (eksekusi order)

```bash
DRY_RUN=false INDODAX_KEY=your_api_key python main.py
```

### Mode hampir real-time (polling 1 detik)

- Set `REALTIME_MODE=true` agar `INTERVAL_SECONDS` default menjadi 1 detik (bisa diubah jika dibutuhkan).
- Contoh:

```bash
REALTIME_MODE=true DRY_RUN=true python main.py
```

### Mode Grid Trading

- Set `GRID_ENABLED=true` untuk menyalakan penempatan order beli/jual bertingkat simetris di sekitar harga saat ini.
- Atur kepadatan grid dengan `GRID_LEVELS_PER_SIDE` (jumlah level per sisi) dan `GRID_SPACING_PCT` (jarak persen antar level).
- (Opsional) Set `GRID_ORDER_SIZE` jika ingin ukuran order per level berbeda dari ukuran dinamis (`RISK_PER_TRADE × INITIAL_CAPITAL / harga`).
- Contoh:

```bash
GRID_ENABLED=true GRID_LEVELS_PER_SIDE=4 GRID_SPACING_PCT=0.003 DRY_RUN=true python main.py
```

### Streaming WebSocket & fallback REST

- Dengan `WEBSOCKET_ENABLED=true`, bot akan mencoba memakai WebSocket (jika tersedia) untuk data real-time. Bila gagal, otomatis jatuh ke polling REST dengan interval `INTERVAL_SECONDS`.
- Anda dapat mengatur endpoint dengan `WEBSOCKET_URL` bila diperlukan.

### Antrean order dengan rate limit

- `ORDER_QUEUE_ENABLED=true` mengaktifkan antrean order yang menegakkan jeda minimal antar request (`ORDER_MIN_INTERVAL`, default **2.0 detik**).
- Berlaku untuk `create_order` dan `cancel_order` agar tidak melampaui batas rate limit API.
- REST hanya dipakai untuk: `place order`, `cancel order`, `cek balance`, `cek open orders` — semua serialisasi melalui antrean ini.
- Data market (harga, orderbook, trade stream) diperoleh via WebSocket sehingga tidak membebani REST.

### Hold time adaptif berdasarkan volume

Fitur ini memungkinkan bot menahan posisi lebih lama di pair liquid dan lebih cepat keluar dari pair iliquid:

```
VOLUME_HIGH_THRESHOLD_IDR=5000000000   # 5 miliar IDR — batas volume "tinggi"
MAX_HOLD_SECONDS_VOLUME_HIGH=5400      # 90 menit untuk pair volume tinggi
MAX_HOLD_SECONDS_VOLUME_LOW=1800       # 30 menit untuk pair volume rendah
```

Aktifkan dengan menyetel `VOLUME_HIGH_THRESHOLD_IDR > 0`. Jika 0, fitur nonaktif dan `MAX_HOLD_SECONDS` dipakai sebagai batas tetap.

### Caching REST (mencegah spam API)

```
ACCOUNT_INFO_CACHE_TTL=30   # cache saldo 30 detik
OPEN_ORDERS_CACHE_TTL=15    # cache open-orders 15 detik
```

### Batasi posisi aktif

```
MAX_OPEN_POSITIONS=3   # maksimal 3 pair aktif sekaligus
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

## Operasi Mandiri 24/7

Bot dirancang untuk berjalan tanpa campur tangan:

- **Monitoring posisi**: Setiap siklus, jika bot sedang memegang posisi (hasil sesi sebelumnya maupun sesi saat ini), bot langsung menganalisa pair tersebut terlebih dahulu. Jika sinyal berubah ke *sell* atau kondisi stop terpenuhi, posisi ditutup otomatis sebelum mencari peluang baru.
- **Rotasi pair**: Saat kondisi stop (target profit / batas rugi / trailing stop) tercapai, bot **tidak berhenti** — ia melikuidasi sisa posisi lalu langsung memindai pair berikutnya.
- **Interval adaptif**: Saat memegang posisi, bot menggunakan `POSITION_CHECK_INTERVAL` (default 60 detik) yang lebih cepat daripada `INTERVAL_SECONDS` standar agar exit tidak terlambat.
- **Ringkasan berkala**: Setiap `CYCLE_SUMMARY_INTERVAL` siklus scan penuh, log ringkasan performa (PnL, ekuitas, jumlah trade, win-rate) dicetak.
- **Graceful shutdown**: Sinyal `SIGTERM` (mis. `docker stop`) diserap dengan aman — bot menyelesaikan siklus yang sedang berjalan baru berhenti.
- **Mode `single`**: Set `TRADE_MODE=single` untuk menjalankan tepat satu siklus beli→jual lalu berhenti otomatis (berguna untuk otomasi/skrip).
- **WebSocket yang benar**: Koneksi WebSocket kini dilengkapi **auto-reconnect** dengan exponential backoff (2 s → 4 s → … → 60 s), **partial-update merging** (pembaruan ticker tidak menghapus data order-book yang sudah ada), dan pengiriman pesan **subscription** otomatis via `WEBSOCKET_SUBSCRIBE_MESSAGE` setelah koneksi tersambung.

## Keamanan & Catatan

- Pastikan `DRY_RUN=true` saat mencoba pertama kali.
- Gunakan kunci API dengan izin minimal.
- Perdagangan aset kripto berisiko tinggi; lakukan uji sendiri sebelum live.

## Testing

```bash
python -m unittest discover
```

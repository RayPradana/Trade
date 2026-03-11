# Indodax Automated Trading Bot (Python)

Bot trading otomatis untuk Indodax menggunakan Python. Bot melakukan:

- Analisis tren koin dengan moving average cepat/lambat
- Analisa order book (spread, ketidakseimbangan volume), volume, dan candlestick dari data trades
- Pemilihan gaya trading otomatis (scalping, day trading, swing, position) sesuai kondisi pasar
- Eksekusi order beli/jual otomatis dengan batasan slippage, stop-loss, take-profit
- Mode **dry-run** bawaan agar aman untuk simulasi tanpa mengeksekusi order sungguhan

> Gunakan dokumentasi resmi Indodax: https://github.com/btcid/indodax-official-api-docs

## Prasyarat

- Python 3.10+
- `pip install -r requirements.txt`

## Konfigurasi

Konfigurasi dapat diatur lewat variabel lingkungan:

| Variabel | Deskripsi | Default |
| --- | --- | --- |
| `INDODAX_KEY` | API key Indodax (hanya diperlukan untuk mode live) | - |
| `INDODAX_SECRET` | API secret Indodax | - |
| `TRADE_PAIR` | Pasangan dagang, contoh `btc_idr` | `btc_idr` |
| `BASE_ORDER_SIZE` | Ukuran order dasar (dalam aset dasar) | `0.0001` |
| `RISK_PER_TRADE` | Risiko per transaksi (0.01 = 1%) | `0.01` |
| `DRY_RUN` | `true/false` untuk simulasi | `true` |
| `MIN_CONFIDENCE` | Ambang kepercayaan minimum untuk eksekusi | `0.52` |
| `INTERVAL_SECONDS` | Interval candlestick hasil agregasi trades | `300` |
| `FAST_WINDOW` | Periode MA cepat | `12` |
| `SLOW_WINDOW` | Periode MA lambat | `48` |
| `MAX_SLIPPAGE_PCT` | Batas slippage relatif | `0.001` |

## Menjalankan

### Dry-run (simulasi, aman)

```bash
python main.py --pair btc_idr --once
```

### Mode live (eksekusi order)

```bash
export INDODAX_KEY=your_key
export INDODAX_SECRET=your_secret
python main.py --pair btc_idr --live
```

Opsi penting:

- `--once` menjalankan satu siklus analisa + keputusan.
- `--interval` menimpa interval candlestick.
- `--min-confidence` mengubah ambang eksekusi.

## Cara Kerja Singkat

1. **Pengambilan data**: ticker, order book (depth), dan riwayat trades via API publik.
2. **Analisa**:
   - Candlestick dibangun dari trades dengan interval yang dipilih.
   - Tren dihitung memakai MA cepat/lambat.
   - Order book dihitung spread, volume bid/ask, dan imbalance.
   - Volatilitas dihitung dari imbal hasil candle.
3. **Pemilihan strategi**:
   - **Scalping**: spread tipis, likuiditas tinggi, volatilitas rendah.
   - **Day Trading**: tren aktif dengan volatilitas moderat.
   - **Swing Trading**: tren kuat dengan volatilitas lebih tinggi.
   - **Position Trading**: kondisi lain (lebih tenang / tren panjang).
4. **Keputusan trading**: aksi beli/jual/hold dengan stop-loss & take-profit dinamis.
5. **Eksekusi**: 
   - **Dry-run**: hanya log simulasi.
   - **Live**: mengirim order `trade` ke `tapi` Indodax (butuh API key/secret).

## Keamanan & Catatan

- Pastikan `DRY_RUN=true` saat mencoba pertama kali.
- Gunakan kunci API dengan izin minimal.
- Perdagangan aset kripto berisiko tinggi; lakukan uji sendiri sebelum live.

## Testing

```bash
python -m unittest discover
```

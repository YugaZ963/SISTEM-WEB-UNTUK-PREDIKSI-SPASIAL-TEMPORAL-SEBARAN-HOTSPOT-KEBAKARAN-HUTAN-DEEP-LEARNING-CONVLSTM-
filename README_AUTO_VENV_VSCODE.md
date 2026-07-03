# Auto Aktivasi `.venv` di VSCode

Folder ini sudah disiapkan agar ketika dibuka melalui VSCode, terminal PowerShell otomatis memakai virtual environment dari project utama:

```text
C:\laragon\www\sistem-web-skripsi-ta\.venv
```

Konfigurasi berada di:

```text
.vscode/settings.json
```

## Cara Pakai

1. Buka VSCode.
2. Pilih `File > Open Folder`.
3. Pilih folder:

```text
C:\laragon\www\sistem-web-skripsi-ta - 2
```

4. Buka terminal baru di VSCode.
5. Terminal akan otomatis:

```text
- mengaktifkan .venv dari project utama;
- masuk ke folder 02_Project_Web_Demo.
```

Jika berhasil, prompt terminal biasanya menampilkan:

```text
(.venv)
```

## Jalankan Sistem Web

Setelah terminal terbuka, jalankan:

```powershell
uvicorn backend.main:app --reload
```

Lalu buka browser:

```text
http://127.0.0.1:8000
```

## Catatan Penting

Konfigurasi ini tidak menyalin folder `.venv` ke paket sidang karena ukurannya besar. VSCode diarahkan untuk menggunakan `.venv` asli dari:

```text
C:\laragon\www\sistem-web-skripsi-ta\.venv
```

Artinya, folder paket sidang ini akan berjalan otomatis selama folder project utama dan `.venv` aslinya masih ada di lokasi tersebut.

Jika paket sidang dipindahkan ke laptop lain, maka `.venv` perlu dibuat ulang atau path interpreter di `.vscode/settings.json` perlu disesuaikan.

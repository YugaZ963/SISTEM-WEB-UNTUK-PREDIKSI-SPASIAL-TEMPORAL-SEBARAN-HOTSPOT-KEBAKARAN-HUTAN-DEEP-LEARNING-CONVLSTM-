param(
    [Parameter(Mandatory = $true)]
    [string]$Judul,

    [Parameter(Mandatory = $true)]
    [string]$Ringkasan,

    [string]$Kategori = "Perubahan Umum",
    [string]$Pelaksana = "Codex",
    [string]$LatarBelakang = "Tidak ada latar belakang tambahan.",
    [string[]]$FileTerdampak,
    [string[]]$DetailTeknis,
    [string[]]$Dampak,
    [string[]]$Verifikasi,
    [string[]]$Risiko,
    [string[]]$RencanaLanjut
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $projectRoot "LOG_PERUBAHAN_PROJECT.md"

function Convert-ToBulletList {
    param(
        [string[]]$Items,
        [string]$FallbackText
    )

    $normalizedItems = @()
    if ($null -ne $Items) {
        foreach ($item in $Items) {
            if ([string]::IsNullOrWhiteSpace($item)) {
                continue
            }

            # Split only if whole item is compact comma-separated tokens (e.g. "a,b,c")
            # and keep normal sentences as a single bullet.
            if ($item -match '^\S+(,\S+)+$') {
                $parts = $item -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
            } else {
                $parts = @($item.Trim())
            }
            if ($parts.Count -gt 0) {
                $normalizedItems += $parts
            }
        }
    }

    if ($normalizedItems.Count -eq 0) {
        return "- $FallbackText"
    }

    return ($normalizedItems | ForEach-Object { "- $_" }) -join "`r`n"
}

if (-not (Test-Path $logPath)) {
    $defaultHeader = @"
# LOG PERUBAHAN PROJECT

Dokumen ini dipakai untuk mencatat setiap perubahan penting pada project secara berurutan.
Semua entri baru harus ditambahkan (append), bukan mengganti isi lama.

## Riwayat Perubahan

"@
    Set-Content -Path $logPath -Value $defaultHeader -Encoding UTF8
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

$entry = @"
## [$timestamp] $Judul

- Kategori: $Kategori
- Pelaksana: $Pelaksana

### Ringkasan
$Ringkasan

### Latar Belakang
$LatarBelakang

### File Terdampak
$(Convert-ToBulletList -Items $FileTerdampak -FallbackText "Belum diisi.")

### Detail Teknis
$(Convert-ToBulletList -Items $DetailTeknis -FallbackText "Belum diisi.")

### Dampak
$(Convert-ToBulletList -Items $Dampak -FallbackText "Belum diisi.")

### Verifikasi
$(Convert-ToBulletList -Items $Verifikasi -FallbackText "Belum diisi.")

### Risiko / Catatan
$(Convert-ToBulletList -Items $Risiko -FallbackText "Tidak ada risiko tambahan.")

### Rencana Lanjut
$(Convert-ToBulletList -Items $RencanaLanjut -FallbackText "Tidak ada rencana lanjut tambahan.")

---
"@

Add-Content -Path $logPath -Value "`r`n$entry" -Encoding UTF8
Write-Output "Log berhasil ditambahkan ke: $logPath"

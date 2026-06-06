# FinSight AI — API smoke test (PowerShell)
# Kullanım: .\scripts\test_api.ps1
# Önkoşul: uvicorn backend.api.main:app --port 8000

$Base = "http://127.0.0.1:8000"
$Email = "demo@finsight.test"
$Password = "Demo1234!"

Write-Host "`n[1/5] Health..." -ForegroundColor Cyan
$h = Invoke-RestMethod "$Base/health"
Write-Host "  OK — $($h.status) v$($h.version)"

Write-Host "`n[2/5] Register..." -ForegroundColor Cyan
$reg = @{ email = $Email; password = $Password; full_name = "Demo User" } | ConvertTo-Json
try {
    Invoke-RestMethod "$Base/api/v1/auth/register" -Method POST -Body $reg -ContentType "application/json" | Out-Null
    Write-Host "  OK — kullanıcı oluşturuldu"
} catch {
    Write-Host "  SKIP — zaten kayıtlı veya hata"
}

Write-Host "`n[3/5] Login..." -ForegroundColor Cyan
$token = Invoke-RestMethod "$Base/api/v1/auth/login" -Method POST `
    -Body "username=$Email&password=$Password" `
    -ContentType "application/x-www-form-urlencoded"
$auth = @{ Authorization = "Bearer $($token.access_token)" }
Write-Host "  OK — token alındı"

Write-Host "`n[4/5] THYAO haberler..." -ForegroundColor Cyan
$news = Invoke-RestMethod "$Base/api/v1/stocks/THYAO/news?limit=3" -Headers $auth
Write-Host "  OK — $($news.count) haber"

Write-Host "`n[5/5] THYAO + ASELS karşılaştırma..." -ForegroundColor Cyan
$cmp = Invoke-RestMethod "$Base/api/v1/stocks/compare/multi?tickers=THYAO,ASELS" -Headers $auth
foreach ($t in $cmp.tickers) {
    $d = $cmp.comparison.$t
    if ($d.fiyat) { Write-Host "  $t — Fiyat: $($d.fiyat) TL, RSI: $($d.rsi)" }
    else { Write-Host "  $t — Hata: $($d.hata)" }
}

Write-Host "`nTüm smoke testler tamamlandı. Swagger: $Base/docs`n" -ForegroundColor Green

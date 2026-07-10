$key = "kiro-gateway-local"
$headers = @{ Authorization = "Bearer $key"; 'Content-Type' = 'application/json' }

Write-Host "=== 1) /v1/models : looking for 4.8 / sonnet-5 ==="
try {
    $r = Invoke-RestMethod -Uri 'http://localhost:8000/v1/models' -Headers $headers -TimeoutSec 10
    $ids = $r.data | ForEach-Object { $_.id }
    $ids | ForEach-Object { Write-Host "  $_" }
    if ($ids -contains 'claude-opus-4.8') { Write-Host "  --> claude-opus-4.8 PRESENT" }
    if ($ids -contains 'claude-sonnet-5') { Write-Host "  --> claude-sonnet-5 PRESENT" }
} catch {
    Write-Host ("  models FAIL: " + $_.Exception.Message)
}

Write-Host ""
Write-Host "=== 2) /v1/messages with an embedded system message (the 422 case) ==="
$body = @{
    model = "claude-opus-4-8"
    max_tokens = 20
    messages = @(
        @{ role = "user";   content = "say OK" },
        @{ role = "system"; content = "you are terse" }
    )
} | ConvertTo-Json -Depth 6
try {
    $resp = Invoke-RestMethod -Uri 'http://localhost:8000/v1/messages' -Method Post -Headers $headers -Body $body -TimeoutSec 60
    $text = ($resp.content | Where-Object { $_.type -eq 'text' } | Select-Object -First 1).text
    Write-Host "  RESULT: 200 OK (no 422). model=$($resp.model) reply='$text'"
} catch {
    $status = $_.Exception.Response.StatusCode.value__
    Write-Host "  RESULT: HTTP $status -> $($_.Exception.Message)"
}

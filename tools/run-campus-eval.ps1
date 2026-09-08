param(
    [string]$BaseUrl = "http://localhost:8000"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$evalPath = Join-Path $projectRoot "data\eval\campus_golden.json"
$payload = Get-Content -LiteralPath $evalPath -Raw -Encoding UTF8
$body = [System.Text.Encoding]::UTF8.GetBytes($payload)

Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/eval/run" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

param(
    [string]$BaseUrl = "http://localhost:8000"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$knowledgePath = Join-Path $projectRoot "data\knowledge\campus_knowledge.json"
$documents = Get-Content -LiteralPath $knowledgePath -Raw -Encoding UTF8 | ConvertFrom-Json
$payload = @{ documents = $documents } | ConvertTo-Json -Depth 8
$body = [System.Text.Encoding]::UTF8.GetBytes($payload)

Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/knowledge/add" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @("-q")
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$image = "echomind-dev-test"

docker build --target development -t $image $repoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dockerArgs = @(
    "run", "--rm",
    "--entrypoint", "python",
    "-v", "${repoRoot}:/app",
    "-w", "/app",
    $image,
    "-m", "pytest"
) + $PytestArgs

docker @dockerArgs
exit $LASTEXITCODE

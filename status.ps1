# Check how training is going on the pod.  Usage:  .\status.ps1
# Auto-refresh every 30s:  .\status.ps1 -Watch
param([switch]$Watch)

# Display remote Chinese output correctly
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$POD  = "root@103.196.86.68"
$PORT = "42099"
$KEY  = "$HOME\.ssh\id_ed25519"
$SH   = Join-Path $PSScriptRoot "status_remote.sh"

# Send the bash script via base64 (read as raw bytes -> no codepage / BOM issues)
$B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($SH))

function Show-Status {
    ssh -o BatchMode=yes -o ConnectTimeout=10 -p $PORT -i $KEY $POD "echo $B64 | base64 -d | bash"
}

if ($Watch) {
    while ($true) {
        Clear-Host
        Write-Host ("refreshed " + (Get-Date -Format "HH:mm:ss") + "  (Ctrl+C to quit)") -ForegroundColor Cyan
        Show-Status
        Start-Sleep -Seconds 30
    }
} else {
    Show-Status
}

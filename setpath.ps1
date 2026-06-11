$ProjectRoot = $PSScriptRoot
$ProjectBin = Join-Path -Path $ProjectRoot -ChildPath "bin"
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$PathEntries = $UserPath -split ";" | Where-Object { $_ }

if ($PathEntries -notcontains $ProjectBin) {
  [Environment]::SetEnvironmentVariable(
    "PATH",
    "$ProjectBin;$UserPath",
    "User"
  )
  Write-Host "Added to user PATH: $ProjectBin"
} else {
  Write-Host "Already in user PATH: $ProjectBin"
}

Write-Host "Restart PowerShell or VS Code for the updated PATH to be visible in new terminals."

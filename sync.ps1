<#
.SYNOPSIS
  Sync the Obsidian LLM Wiki into the Jekyll site, commit, and push.

.EXAMPLE
  .\sync.ps1              # convert wiki -> _posts, commit, and push to origin
  .\sync.ps1 -NoPush      # convert and commit locally, but do not push
  .\sync.ps1 -DryRun      # preview what would change, write nothing
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# First run: create the machine-specific config from the template.
if (-not (Test-Path "sync_config.yml")) {
    Write-Host "sync_config.yml not found - copying from sync_config.example.yml." -ForegroundColor Yellow
    Copy-Item "sync_config.example.yml" "sync_config.yml"
    Write-Host "Edit sync_config.yml (set vault_wiki_path) and run again." -ForegroundColor Yellow
    exit 1
}

$syncArgs = @()
if ($DryRun) { $syncArgs += "--dry-run" }
if ($NoPush) { $syncArgs += "--no-push" }

Write-Host "Running: python sync.py $syncArgs" -ForegroundColor Cyan
python sync.py @syncArgs
exit $LASTEXITCODE

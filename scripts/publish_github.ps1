# Publish adversarial-nids-cicids2017 to GitHub (run once after: gh auth login)
$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI;" + $env:Path

$RepoRoot = Split-Path $PSScriptRoot -Parent
$RepoName = "adversarial-nids-cicids2017"
$Owner = "hitvika"

Set-Location $RepoRoot

gh auth status | Out-Null

if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

git add .
git commit -m "Publish IITK B.Cyber adversarial NIDS project" 2>$null

$remote = "https://github.com/$Owner/$RepoName.git"
if (-not (git remote | Select-String -Quiet "origin")) {
    git remote add origin $remote
}

gh repo view "$Owner/$RepoName" 2>$null
if ($LASTEXITCODE -ne 0) {
    gh repo create $RepoName --public --source . --remote origin --description "Adversarial robustness evaluation for ML-based NIDS on CIC-IDS2017 (IITK B.Cyber project)" --push
} else {
    git push -u origin main
}

Write-Host "Published: https://github.com/$Owner/$RepoName"
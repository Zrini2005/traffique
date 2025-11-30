# Cleanup script for git repository
# Run this before pushing to remove unnecessary files

Write-Host "=" 60
Write-Host "🧹 Repository Cleanup Script" -ForegroundColor Cyan
Write-Host "=" 60

$removed = 0
$skipped = 0

# Function to safely remove file
function Remove-SafeFile {
    param($path)
    if (Test-Path $path) {
        Remove-Item $path -Force
        Write-Host "✓ Removed: $path" -ForegroundColor Green
        $script:removed++
    } else {
        Write-Host "⊘ Not found: $path" -ForegroundColor Yellow
        $script:skipped++
    }
}

# Function to safely remove directory
function Remove-SafeDir {
    param($path)
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "✓ Removed directory: $path" -ForegroundColor Green
        $script:removed++
    } else {
        Write-Host "⊘ Not found: $path" -ForegroundColor Yellow
        $script:skipped++
    }
}

Write-Host "`n📝 Removing broken/experimental test files..." -ForegroundColor Cyan
Write-Host "Keeping: test_confidence_levels.py, test_quick_production.py, test_sahi.py, test_visdrone_comparison.py" -ForegroundColor Gray
Remove-SafeFile "test_fast_mode.py"
Remove-SafeFile "test_minimal.py"
Remove-SafeFile "test_production_visual.py"
Remove-SafeFile "test_realtime_visual.py"
Remove-SafeFile "test_super_fast.py"
Remove-SafeFile "view_test_results.py"

Write-Host "`n📝 Removing empty/obsolete Python files..." -ForegroundColor Cyan
Remove-SafeFile "analytics.py"
Remove-SafeFile "process_multicam_sahi_visdrone.py"

Write-Host "`n📝 Removing old/redundant files..." -ForegroundColor Cyan
Write-Host "Keeping: estimate_processing_time.py, calculate_optimal_interval.py (utility scripts)" -ForegroundColor Gray
Remove-SafeFile "process_multicam_dedup.py"
Remove-SafeFile "process_multicam_fast_dedup.py"
Remove-SafeFile "process_production_system.py"
Remove-SafeFile "cleanup_repo.py"
Remove-SafeFile "temp_frame.jpg"

Write-Host "`n📝 Removing frontend backup files..." -ForegroundColor Cyan
Remove-SafeFile "frontend\src\pages\AnalysisPage_old.jsx"
Remove-SafeFile "frontend\src\pages\AnalysisPage_gradient.jsx"
Remove-SafeFile "frontend\src\pages\AnalysisPage_professional.jsx"

Write-Host "`n📝 Removing Python cache..." -ForegroundColor Cyan
Remove-SafeDir "__pycache__"
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-SafeDir $_.FullName
}

Write-Host "`n📝 Removing virtual environments..." -ForegroundColor Cyan
Remove-SafeDir ".venv"
Remove-SafeDir "venv"

Write-Host "`n📝 Removing runtime data..." -ForegroundColor Cyan
Remove-SafeDir "uploads"
Remove-SafeDir "output"
Remove-SafeDir "runs"

Write-Host "`n📝 Removing node_modules (will be reinstalled)..." -ForegroundColor Cyan
Remove-SafeDir "frontend\node_modules"
Remove-SafeDir "frontend\.vite"

Write-Host "`n" ("=" * 60)
Write-Host "✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "Files removed: $removed" -ForegroundColor Green
Write-Host "Files skipped (not found): $skipped" -ForegroundColor Yellow
Write-Host ("=" * 60)

Write-Host "`n⚠️  OPTIONAL: Remove large model files" -ForegroundColor Yellow
Write-Host "These will be re-downloaded automatically when needed:" -ForegroundColor Gray
Write-Host "  - yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt" -ForegroundColor Gray
Write-Host "  - yolo11n.pt" -ForegroundColor Gray
Write-Host "`nRun this to remove them:" -ForegroundColor Gray
Write-Host "  Remove-Item yolov8*.pt, yolo11n.pt" -ForegroundColor Gray

Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Update .gitignore to include:" -ForegroundColor White
Write-Host "   - __pycache__/" -ForegroundColor Gray
Write-Host "   - .venv/" -ForegroundColor Gray
Write-Host "   - venv/" -ForegroundColor Gray
Write-Host "   - uploads/" -ForegroundColor Gray
Write-Host "   - output/" -ForegroundColor Gray
Write-Host "   - runs/" -ForegroundColor Gray
Write-Host "   - *.pt (model files)" -ForegroundColor Gray
Write-Host "   - node_modules/" -ForegroundColor Gray
Write-Host "   - .vite/" -ForegroundColor Gray
Write-Host "   - temp_*.jpg" -ForegroundColor Gray
Write-Host "`n2. Review changes: git status" -ForegroundColor White
Write-Host "3. Stage changes: git add ." -ForegroundColor White
Write-Host "4. Commit: git commit -m 'Clean up repository'" -ForegroundColor White
Write-Host "5. Push: git push" -ForegroundColor White

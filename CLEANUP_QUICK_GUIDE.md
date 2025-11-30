# 🧹 Quick Cleanup Guide

## Run This First
```powershell
.\cleanup_for_git.ps1
```

## What Gets Removed?

### ❌ Remove (Safe)
- ✓ Broken test files: `test_fast_mode.py`, `test_minimal.py`, `test_production_visual.py`, `test_realtime_visual.py`, `test_super_fast.py` (5 files)
- ✓ Empty files: `analytics.py`, `process_multicam_sahi_visdrone.py`
- ✓ Old versions: `process_multicam_dedup.py`, `process_multicam_fast_dedup.py`, `process_production_system.py`
- ✓ Frontend backups: `AnalysisPage_*.jsx` (3 files)
- ✓ Python cache: `__pycache__/`
- ✓ Virtual envs: `.venv/`, `venv/`
- ✓ Runtime data: `uploads/`, `output/`, `runs/`
- ✓ Node modules: `frontend/node_modules/`, `frontend/.vite/`

### ✅ Keep (Important)
- ✓ `api_server.py` - Backend API
- ✓ `interactive_analytics.py` - Analytics engine
- ✓ `frontend/` - React app
- ✓ `*.md` files - Documentation
- ✓ `requirements.txt` - Dependencies
- ✓ All config files
- ✓ Utility scripts: `estimate_processing_time.py`, `calculate_optimal_interval.py`, `compare_models.py`, `analyze_results.py`
- ✓ Working tests: `test_confidence_levels.py`, `test_quick_production.py`, `test_sahi.py`, `test_visdrone_comparison.py`

## After Cleanup
```bash
# 1. Check status
git status

# 2. Add all files
git add .

# 3. Commit
git commit -m "Clean up repository for production"

# 4. Push
git push origin main
```

## Optional: Remove Models
```powershell
# Models will auto-download when needed
Remove-Item yolov8*.pt, yolo11n.pt
```

## Need Help?
See `CLEANUP_SUMMARY.md` for detailed information.

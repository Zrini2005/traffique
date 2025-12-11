import requests
import sys
import os
import argparse
import zipfile
import base64
import io

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 🛑 UPDATE THIS EVERY TIME YOU RESTART COLAB 🛑
COLAB_URL = "https://sandi-imprescriptible-daine.ngrok-free.dev" 
AUTH_TOKEN = "traffique-secure-project"
# ==========================================

def install_remote_packages(packages):
    if not packages: return True
    print(f"📦 Installing dependencies: {', '.join(packages)}...")
    try:
        response = requests.post(f"{COLAB_URL}/install", json={'token': AUTH_TOKEN, 'packages': packages}, timeout=300)
        if response.status_code == 200:
            print("   ✅ Install success.")
            return True
        print(f"   ❌ Install failed: {response.text}")
        return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

def compress_project(output_filename="project_bundle.zip"):
    """Zips the current directory, excluding junk folders"""
    print("🗜️  Zipping project (excluding .venv, .git)...")
    
    zip_buffer = io.BytesIO()
    
    # Folders/Files to IGNORE
    IGNORE_DIRS = {'.venv', 'venv', 'env', '.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'output'}
    IGNORE_EXTS = {'.pyc', '.zip'} 
    # Note: I removed .mp4 from ignore so your video gets uploaded!
    # If your video is huge, you might want to exclude it and upload manually.

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if any(file.endswith(ext) for ext in IGNORE_EXTS): continue
                if file == output_filename or file == "colabTest.py": continue
                
                file_path = os.path.join(root, file)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, "."))
    
    return zip_buffer.getvalue()

def upload_project():
    """Uploads the zipped project to Colab"""
    zip_bytes = compress_project()
    encoded_content = base64.b64encode(zip_bytes).decode('utf-8')
    
    print(f"⬆️  Uploading project bundle ({len(zip_bytes)/1024:.1f} KB)...")
    try:
        response = requests.post(
            f"{COLAB_URL}/upload_zip",
            json={'token': AUTH_TOKEN, 'content': encoded_content},
            timeout=120
        )
        if response.status_code == 200:
            print("   ✅ Project uploaded & extracted.")
            return True
        else:
            print(f"   ❌ Upload failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Upload connection error: {e}")
        return False

def run_main_script(filename, script_args):
    """Executes the script on the remote server with arguments"""
    print(f"🚀 Running {filename} on GPU with args: {script_args}...")
    try:
        response = requests.post(
            f"{COLAB_URL}/run", 
            # ✅ FIX: Sending 'args' in the JSON payload
            json={'token': AUTH_TOKEN, 'filename': filename, 'args': script_args},
            timeout=600
        )
        if response.status_code == 200:
            data = response.json()
            print("\n" + "─" * 40)
            print("📜 REMOTE OUTPUT:")
            print("─" * 40)
            print(data.get('output', 'No output'))
            print("─" * 40)
            if data.get('status') == 'error': print("❌ Execution Failed")
            else: print("✅ Execution Complete")
        else:
            print(f"❌ Server Error: {response.text}")
    except Exception as e:
        print(f"❌ Run Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="The main python script to run")
    parser.add_argument("--install", "-i", nargs="+", help="Pip packages", default=[])
    # This captures all remaining arguments (like your video file)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the script")
    args = parser.parse_args()

    # 1. Install
    if args.install:
        if not install_remote_packages(args.install): return

    # 2. Upload Whole Project
    if not upload_project(): return

    # 3. Run
    # ✅ FIX: Passing the captured 'args.args' to the function
    run_main_script(args.script, args.args)

if __name__ == "__main__":
    main()
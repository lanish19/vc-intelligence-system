#!/usr/bin/env python3
"""
VC Intelligence System - App Launcher

Simple script to launch the Streamlit web app.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'plotly']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_api_key():
    """Check if Gemini API key is configured"""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY is configured")
        return True
    else:
        print("⚠️  GEMINI_API_KEY not found")
        print("   Set it with: export GEMINI_API_KEY=your_key_here")
        print("   Or create a .env file with: GEMINI_API_KEY=your_key_here")
        return False

def main():
    """Main launcher function"""
    print("🧠 VC Intelligence System - Starting Web App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API key (warning only)
    check_api_key()
    
    # Launch Streamlit app
    app_path = Path(__file__).parent / "vc_app.py"
    
    if not app_path.exists():
        print("❌ App file not found: vc_app.py")
        sys.exit(1)
    
    print("\n🚀 Launching Streamlit app...")
    print("   App will open in your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
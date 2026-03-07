import subprocess
import webbrowser
import time
import os
import sys

# =========================================
# Face Recognition System Launcher
# =========================================

def start_backend():
    """
    Start Flask backend server
    """
    print("Starting backend server (Flask API)...")

    subprocess.Popen(
        [sys.executable, "server/app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def start_web_server():
    """
    Start static web server
    """
    print("Starting web interface server...")

    subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def open_browser():
    """
    Open browser automatically
    """
    url = "http://localhost:8000/web/index.html"

    print("Opening web interface...")
    webbrowser.open(url)


def main():

    print("=======================================")
    print("  Real-Time Face Recognition System")
    print("=======================================")

    # Ensure working directory is project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    start_backend()

    time.sleep(2)

    start_web_server()

    time.sleep(2)

    open_browser()

    print("System started successfully.")
    print("Press CTRL + C to stop.")


if __name__ == "__main__":
    main()
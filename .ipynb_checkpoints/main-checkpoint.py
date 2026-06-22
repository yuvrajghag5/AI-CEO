
import subprocess
import time
import sys
from pyngrok import ngrok
 
DASHBOARD_PATH = "dashboard/dashboard.py"
PORT = 8501
 
streamlit_process = None
 
 
def start_streamlit():
    global streamlit_process
    print("Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", DASHBOARD_PATH,
            "--server.port", str(PORT),
            "--server.headless", "true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)  # give Streamlit a moment to actually bind the port
 
 
def start_tunnel():
    print("Opening ngrok tunnel...")
    public_url = ngrok.connect(PORT)
    print(f"\nDashboard is live at: {public_url}\n")
    print("Press Ctrl+C to stop the dashboard and tunnel.\n")
    return public_url
 
 
def shutdown(public_url):
    print("\nShutting down...")
    if public_url:
        ngrok.disconnect(public_url)
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
    print("Stopped.")
 
 
def main():
    public_url = None
    try:
        start_streamlit()
        public_url = start_tunnel()
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(public_url)
 
 
if __name__ == "__main__":
    main()
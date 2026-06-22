# run once before the final file:- python3 -c "from pyngrok import ngrok; ngrok.set_auth_token('3FUuBCmRw6lJJEdI84wwVNohJwZ_7CyVLrEjYbg2bYyqYxsxX')"



import time
from pyngrok import ngrok
 
PORT = 8501
 
public_url = ngrok.connect(PORT)
print(f"\nDashboard is live at: {public_url}\n")
print("Keep this script running. Press Ctrl+C to stop the tunnel.\n")
 
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    ngrok.disconnect(public_url)
    print("Tunnel closed.")
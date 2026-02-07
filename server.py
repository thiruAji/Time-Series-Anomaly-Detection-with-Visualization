from waitress import serve
from app import app
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5001
    local_ip = get_ip()
    
    print(f"\n==================================================================")
    print(f"🚀 PRODUCTION SERVER STARTED")
    print(f"------------------------------------------------------------------")
    print(f"Server is running on:")
    print(f"  > Local:   http://localhost:{port}")
    print(f"  > Network: http://{local_ip}:{port}")
    print(f"------------------------------------------------------------------")
    print(f"Press Ctrl+C to stop the server")
    print(f"==================================================================\n")
    
    serve(app, host=host, port=port)

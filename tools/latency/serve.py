"""
Serve the latency test page over HTTPS on the local network, with the headers
browsers require for multi-threaded WebAssembly (cross-origin isolation).

    uv run python tools/latency/serve.py          # then open the printed URL on your phone

HTTPS is needed because phones only enable WebGPU and threads on secure
pages. The certificate is self-signed (made on first run), so the phone will
warn once: choose "Advanced -> Proceed" (Chrome) or "Show details -> visit
this website" (Safari). Phone and computer must be on the same network; if
the page doesn't load, the firewall may be blocking the port
(e.g. `sudo ufw allow 8443/tcp`).
"""

import argparse
import http.server
import socket
import ssl
import subprocess
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT, KEY = HERE / '.cert' / 'cert.pem', HERE / '.cert' / 'key.pem'


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.mjs': 'text/javascript', '.js': 'text/javascript',
                      '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
                      '.json': 'application/json'}

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def lan_addresses():
    out = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split()
    return [a for a in out if ':' not in a] or [socket.gethostbyname(socket.gethostname())]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8443)
    a = p.parse_args()
    if not CERT.exists():
        CERT.parent.mkdir(exist_ok=True)
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '365',
                        '-keyout', str(KEY), '-out', str(CERT), '-subj', '/CN=duke-latency-test'],
                       check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)
    server = http.server.ThreadingHTTPServer(('0.0.0.0', a.port), partial(Handler, directory=str(HERE)))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    for addr in lan_addresses():
        print(f'open https://{addr}:{a.port}/ on your phone')
    print('Ctrl-C to stop')
    server.serve_forever()


if __name__ == '__main__':
    main()

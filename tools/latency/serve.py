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

The page posts its results, failures and log to /report as it goes; each page
visit is saved as results/<time>-<device>.json, so nothing needs pasting.
"""

import argparse
import http.server
import json
import re
import socket
import ssl
import subprocess
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT, KEY = HERE / '.cert' / 'cert.pem', HERE / '.cert' / 'key.pem'
RESULTS = HERE / 'results'
MAX_REPORT = 4_000_000


def device_slug(user_agent):
    """'... (Linux; Android 10; K) ... Chrome/154.0 ...' -> 'android10-chrome154'."""
    # Most specific first: Android user agents also say "Linux".
    os_match = next(filter(None, (re.search(pat, user_agent) for pat in (
        r'Android [\d.]+', r'iPhone OS \d+', r'iPad', r'Mac OS X', r'Windows NT [\d.]+', r'Linux'))), None)
    browser = re.search(r'(Firefox|Edg|CriOS|Chrome|Version)/(\d+)', user_agent)
    parts = [os_match.group(0) if os_match else 'unknown']
    if browser:
        name = {'Edg': 'edge', 'CriOS': 'chrome', 'Version': 'safari'}.get(browser.group(1), browser.group(1))
        parts.append(name + browser.group(2))
    return re.sub(r'[^a-z0-9]+', '', parts[0].lower()) + ''.join('-' + p.lower() for p in parts[1:])


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

    def do_POST(self):
        if self.path != '/report':
            return self.send_error(404)
        n = int(self.headers.get('Content-Length', 0))
        if not 0 < n <= MAX_REPORT:
            return self.send_error(413)
        try:
            report = json.loads(self.rfile.read(n))
            session = re.sub(r'[^0-9A-Za-z-]', '', str(report['session']))[:40]
        except (ValueError, KeyError, TypeError):
            return self.send_error(400)
        report['client'] = self.client_address[0]
        name = f"{session}-{device_slug(str(report.get('env', {}).get('browser', '')))}.json"
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / name).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"report {name}: {len(report.get('results', []))} rows, "
              f"{len(report.get('failures', []))} failures{' (done)' if report.get('done') else ''}", flush=True)
        self.send_response(204)
        self.end_headers()


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

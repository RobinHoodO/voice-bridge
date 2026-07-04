import http.server
import socketserver
import subprocess
import os
from urllib.parse import urlsplit, parse_qs

PORT = 8888
TOKEN = os.environ.get("PHONE_SERVER_TOKEN")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        request_token = parse_qs(parsed.query).get("token", [""])[0]
        if not TOKEN or request_token != TOKEN:
            self.send_response(401)
            self.end_headers()
            return

        if parsed.path == '/ask':
            # Run the voice script in a detached background process
            subprocess.Popen(['bash', os.path.expanduser('~/voice.sh')], start_new_session=True)
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = """
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background: #121212;
                        color: #ffffff;
                    }
                    .mic {
                        font-size: 48px;
                        margin-bottom: 20px;
                        animation: pulse 1.5s infinite;
                    }
                    @keyframes pulse {
                        0% { transform: scale(1); opacity: 0.8; }
                        50% { transform: scale(1.1); opacity: 1; }
                        100% { transform: scale(1); opacity: 0.8; }
                    }
                </style>
            </head>
            <body>
                <div class="mic">🎙️</div>
                <h2>Listening...</h2>
                <p>Speak now. Window will close shortly.</p>
                <script>
                    setTimeout(function() {
                        if (window.history.length > 1) {
                            window.history.back();
                        } else {
                            window.close();
                        }
                    }, 6000);
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

# Allow port reuse
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
    print(f"Phone server running on 127.0.0.1:{PORT}")
    httpd.serve_forever()

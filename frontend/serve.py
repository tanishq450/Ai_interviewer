#!/usr/bin/env python3
"""Tiny static server — serves frontend/ at http://localhost:3000"""
import http.server, socketserver, os, webbrowser

PORT = 3000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map['.js'] = 'application/javascript'

print(f"\n🚀  Frontend running at  http://localhost:{PORT}")
print("   Make sure the backend is running: python main.py\n")

webbrowser.open(f"http://localhost:{PORT}")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()

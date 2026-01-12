from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class COOPCOEPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), COOPCOEPHandler)
    print("Serving on http://127.0.0.1:8000 (COOP/COEP enabled)")
    server.serve_forever()


if __name__ == "__main__":
    main()

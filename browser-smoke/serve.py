from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
JFK_SAMPLE = ROOT.parent / "mel-spec" / "testdata" / "jfk_f32le.wav"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def translate_path(self, path):
        request_path = unquote(urlparse(path).path)
        if request_path == "/mel-spec/jfk_f32le.wav":
            return str(JFK_SAMPLE)
        return super().translate_path(path)

    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".mjs"):
            return "text/javascript"
        if path.endswith(".wav"):
            return "audio/wav"
        return super().guess_type(path)


if __name__ == "__main__":
    port = 8787
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving {ROOT} at http://127.0.0.1:{port}/browser-smoke/")
    server.serve_forever()

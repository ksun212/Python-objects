import contextlib
import http.server
import pathlib
import threading
from ssl import PROTOCOL_TLS_SERVER, SSLContext

# Generated with:
# $ openssl req -new -x509 -days 3650 -nodes -out cert.pem \
#     -keyout cert.pem -addext "subjectAltName = DNS:localhost"
CERT_FILE = str(pathlib.Path(__file__).parent / "certs" / "cert.pem")


class HttpServerThread(threading.Thread):
    def __init__(self, handler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = http.server.HTTPServer(("localhost", 7777), handler)

    def run(self):
        self.server.serve_forever(poll_interval=0.001)

    def terminate(self):
        self.server.shutdown()
        self.server.server_close()
        self.join()


class HttpsServerThread(HttpServerThread):
    def __init__(self, handler, *args, **kwargs):
        super().__init__(handler, *args, **kwargs)
        sslcontext = SSLContext(PROTOCOL_TLS_SERVER)
        sslcontext.load_cert_chain(CERT_FILE)
        self.server.socket = sslcontext.wrap_socket(self.server.socket, server_side=True)


def create_server(thread_class):
    def server(handler):
        server_thread = thread_class(handler, daemon=True)
        server_thread.start()
        try:
            yield server_thread
        finally:
            server_thread.terminate()
    return contextlib.contextmanager(server)


http_server = create_server(HttpServerThread)
https_server = create_server(HttpsServerThread)

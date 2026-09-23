"""GitHub-OAuth reverse proxy in front of the J-lens dashboard.

Runs on the always-on EC2 box, NOT on the GPU. It listens on loopback, Caddy
terminates TLS in front of it, and it forwards authenticated requests to the
dashboard's uvicorn - which still binds 127.0.0.1, now on the GPU host, reached
over an SSH tunnel this box holds open.

    Internet --443--> Caddy --127.0.0.1:7870--> authproxy --127.0.0.1:7860-->
                                                          ssh -L --> vast:7860

Two separate steps, and conflating them is the whole point of this file:

  * Authentication proves who the visitor is. Any GitHub user can do that, so on
    its own it proves nothing worth having.
  * Authorization asks whether that login is a collaborator on the repo. It uses
    a SERVER-SIDE token (JLENS_GITHUB_TOKEN), never the visitor's token, so the
    visitor's OAuth grant needs no scopes at all and a hostile visitor cannot
    influence the answer.

The collaborator answer is cached for the life of the session by being carried
inside the signed session cookie, so a logged-in user costs zero GitHub API
calls per request. Revoking someone takes effect within JLENS_SESSION_TTL_S.

Stdlib only, on purpose: this is the code that decides who gets in, it runs on a
box with no venv and no build tools, and a dependency-free auth path is one less
thing to keep patched.

    python3 authproxy.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import http.client
import json
import os
import secrets
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------- config

SESSION_COOKIE = "jlens_session"
STATE_COOKIE = "jlens_oauth_state"
PREFIX = "/_jlens"  # our own routes; everything else is proxied

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
GITHUB_API = "https://api.github.com"

# The dashboard's flood run streams for minutes over SSE, so the read timeout on
# the upstream leg has to be generous. GitHub's own calls get a short one.
UPSTREAM_TIMEOUT_S = 900
GITHUB_TIMEOUT_S = 15

# Hop-by-hop headers must not be forwarded in either direction (RFC 9110 7.6.1).
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class ConfigError(Exception):
    pass


def _env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or v == "":
        raise ConfigError(f"{name} is required and unset")
    return v


class Config:
    """Read and validate every setting up front, so a misconfiguration is a
    refusal to start rather than a redirect loop discovered by a teammate."""

    def __init__(self, env: dict | None = None) -> None:
        env = os.environ if env is None else env
        prev, os.environ = os.environ, env  # _env reads os.environ; keep it simple
        try:
            self.client_id = _env("JLENS_OAUTH_CLIENT_ID")
            self.client_secret = _env("JLENS_OAUTH_CLIENT_SECRET")
            self.github_token = _env("JLENS_GITHUB_TOKEN")
            secret = _env("JLENS_SESSION_SECRET")
            self.public_url = _env("JLENS_PUBLIC_URL").rstrip("/")
            self.upstream = _env("JLENS_UPSTREAM", "http://127.0.0.1:7860").rstrip("/")
            self.listen_host = _env("JLENS_LISTEN_HOST", "127.0.0.1")
            self.listen_port = int(_env("JLENS_LISTEN_PORT", "7870"))
            self.ttl_s = int(_env("JLENS_SESSION_TTL_S", "28800"))
            repos = _env("JLENS_AUTH_REPOS", "Nathan2589/jacobian-lens")
            self.insecure_cookies = env.get("JLENS_INSECURE_COOKIES") == "1"
            self.model_label = env.get("JLENS_MODEL_ID", "")
            self.branch_label = env.get("JLENS_BRANCH", "")
        finally:
            os.environ = prev

        # A short secret is a forgeable session. 32 bytes of entropy, please.
        if len(secret) < 32:
            raise ConfigError(
                "JLENS_SESSION_SECRET must be at least 32 characters "
                f"(got {len(secret)}). Generate one: openssl rand -hex 32"
            )
        self.secret = secret.encode()

        self.repos = [r.strip() for r in repos.split(",") if r.strip()]
        if not self.repos:
            raise ConfigError("JLENS_AUTH_REPOS is empty: nobody could ever be let in")
        for r in self.repos:
            if r.count("/") != 1 or not all(r.split("/")):
                raise ConfigError(f"JLENS_AUTH_REPOS entry {r!r} is not owner/repo")

        # Secure cookies are silently dropped by the browser over plain http, and
        # the symptom is an infinite login loop that looks like an OAuth bug. Make
        # the mismatch impossible to deploy instead of hard to diagnose.
        if not self.public_url.startswith("https://") and not self.insecure_cookies:
            raise ConfigError(
                f"JLENS_PUBLIC_URL is {self.public_url!r}, not https. The session "
                "cookie is Secure, so a browser would discard it and every request "
                "would bounce back to /login forever. Put TLS in front, or set "
                "JLENS_INSECURE_COOKIES=1 for a local test only."
            )
        if self.ttl_s < 60:
            raise ConfigError("JLENS_SESSION_TTL_S below 60s is not usable")

    @property
    def redirect_uri(self) -> str:
        return f"{self.public_url}{PREFIX}/callback"


# ------------------------------------------------------------ signed payloads
# A signed cookie is the session cache: no server-side store, so the proxy can
# restart (or be replaced) without logging everyone out, and there is nothing to
# grow unboundedly. Tamper-evidence comes from the HMAC, not from secrecy - the
# payload is readable by the holder, which is fine, it is their own login.


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(txt: str) -> bytes:
    return base64.urlsafe_b64decode(txt + "=" * (-len(txt) % 4))


def sign(payload: dict, secret: bytes) -> str:
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    mac = hmac.new(secret, body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64e(mac)}"


def unsign(token: str, secret: bytes, now: float | None = None) -> dict | None:
    """Return the payload, or None for anything at all wrong with it: bad shape,
    bad signature, expired. The caller must not be able to tell those apart."""
    now = time.time() if now is None else now
    try:
        body, mac = token.split(".", 1)
        expected = hmac.new(secret, body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(mac), expected):
            return None
        payload = json.loads(_b64d(body))
    except Exception:  # noqa: BLE001 - every malformed cookie is just "no session"
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or exp <= now:
        return None
    return payload


# ------------------------------------------------------------------- GitHub


class GitHub:
    """The two calls we make, and nothing else. Split out so the tests can
    replace it without a network."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def _request(self, req: urllib.request.Request) -> tuple[int, bytes]:
        req.add_header("User-Agent", "jlens-authproxy")
        try:
            with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def exchange_code(self, code: str) -> str | None:
        """Swap the callback code for the visitor's access token."""
        data = urllib.parse.urlencode(
            {
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "code": code,
                "redirect_uri": self.cfg.redirect_uri,
            }
        ).encode()
        req = urllib.request.Request(GITHUB_TOKEN_URL, data=data)
        req.add_header("Accept", "application/json")
        status, body = self._request(req)
        if status != 200:
            return None
        try:
            return json.loads(body).get("access_token") or None
        except Exception:  # noqa: BLE001
            return None

    def login_for(self, user_token: str) -> str | None:
        """The visitor's login. This is the ONLY thing the visitor's token is
        ever used for, which is why the OAuth app can request no scopes."""
        req = urllib.request.Request(f"{GITHUB_API}/user")
        req.add_header("Authorization", f"Bearer {user_token}")
        req.add_header("Accept", "application/vnd.github+json")
        status, body = self._request(req)
        if status != 200:
            return None
        try:
            login = json.loads(body).get("login")
        except Exception:  # noqa: BLE001
            return None
        return login if isinstance(login, str) and login else None

    def is_collaborator(self, repo: str, login: str) -> bool | None:
        """204 yes, 404 no, anything else None meaning 'we do not know'.

        None must be treated as denied by the caller, but reported differently:
        a rate-limited or mis-scoped server token locking everyone out is an
        outage to fix, not a person to turn away.
        """
        url = f"{GITHUB_API}/repos/{repo}/collaborators/{urllib.parse.quote(login)}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.cfg.github_token}")
        req.add_header("Accept", "application/vnd.github+json")
        status, _ = self._request(req)
        if status == 204:
            return True
        if status == 404:
            return False
        return None


def authorize(gh: GitHub, repos: list[str], login: str) -> tuple[bool, str | None]:
    """(allowed, repo_that_granted). Fails closed: if every repo answered with an
    error rather than yes/no, the second element is None and allowed is False,
    and the caller renders the outage page rather than the rejection page."""
    saw_definite_no = False
    for repo in repos:
        answer = gh.is_collaborator(repo, login)
        if answer is True:
            return True, repo
        if answer is False:
            saw_definite_no = True
    return False, ("denied" if saw_definite_no else None)


# --------------------------------------------------------------------- pages

_PAGE = """<!doctype html><meta charset=utf-8>
<title>{title} - J-lens</title>
<style>
 body{{background:#0e1116;color:#c9d1d9;font:15px/1.6 ui-sans-serif,system-ui,sans-serif;
      display:flex;min-height:100vh;margin:0;align-items:center;justify-content:center}}
 main{{max-width:34rem;padding:2rem}}
 h1{{font-size:1.25rem;margin:0 0 .75rem;color:#e6edf3}}
 code{{background:#161b22;padding:.1rem .35rem;border-radius:4px;font-size:.9em}}
 a{{color:#58a6ff}} .m{{color:#8b949e;font-size:.9em;margin-top:1.25rem}}
</style>
<main><h1>{title}</h1>{body}</main>
"""


def page(title: str, body: str) -> bytes:
    return _PAGE.format(title=html.escape(title), body=body).encode()


def denied_page(login: str, repos: list[str]) -> bytes:
    """Terminal 403. Deliberately NOT a redirect: bouncing a rejected user back
    to /login is the loop this page exists to avoid."""
    where = " or ".join(f"<code>{html.escape(r)}</code>" for r in repos)
    return page(
        "Not a collaborator on this repo",
        f"<p>You are signed in to GitHub as <code>{html.escape(login)}</code>, but "
        f"that account is not a collaborator on {where}, so the J-lens dashboard "
        "is not available to you.</p>"
        "<p>If that is wrong, ask Nathan to add you to the repository, then "
        f'<a href="{PREFIX}/login">try again</a>.</p>'
        f'<p class=m>Signed in as the wrong account? <a href="{PREFIX}/logout">Sign '
        "out here</a>, then sign out of github.com, then try again.</p>",
    )


# --------------------------------------------------------------------- server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "jlens-authproxy"
    sys_version = ""

    cfg: Config
    gh: GitHub

    # ------------------------------------------------------------- plumbing
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write(
            f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n"
        )

    def _cookies(self) -> SimpleCookie:
        jar = SimpleCookie()
        raw = self.headers.get("Cookie")
        if raw:
            try:
                jar.load(raw)
            except Exception:  # noqa: BLE001 - a junk Cookie header is no cookie
                return SimpleCookie()
        return jar

    def _cookie_header(self, name: str, value: str, max_age: int) -> str:
        bits = [
            f"{name}={value}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",  # Lax, not Strict: the OAuth callback is a cross-site
            f"Max-Age={max_age}",  # top-level GET and Strict would drop the cookie
        ]
        if not self.cfg.insecure_cookies:
            bits.append("Secure")
        return "; ".join(bits)

    def _send(
        self, status: int, body: bytes, ctype: str = "text/html; charset=utf-8",
        cookies: list[str] | None = None, location: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        if location:
            self.send_header("Location", location)
        for c in cookies or []:
            self.send_header("Set-Cookie", c)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _clear_session(self) -> str:
        return self._cookie_header(SESSION_COOKIE, "", 0)

    # --------------------------------------------------------------- routing
    def do_GET(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == f"{PREFIX}/health":
                # Proxy liveness ONLY. It must never touch the upstream: the GPU
                # instance reaps itself after JLENS_IDLE_KILL_S with no dashboard
                # traffic, and a health check on a timer would look like traffic
                # and keep a $0.43/hr card alive until the credit ran out.
                self._send(200, b"ok\n", "text/plain; charset=utf-8")
            elif path == f"{PREFIX}/login":
                self._login()
            elif path == f"{PREFIX}/callback":
                self._callback()
            elif path == f"{PREFIX}/logout":
                self._logout()
            elif path == f"{PREFIX}/whoami":
                self._whoami()
            else:
                self._guarded_proxy()
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the web
            self.log_message("unhandled %s on %s", exc.__class__.__name__, path)
            self._send(500, page("Something went wrong", "<p>Check the proxy log.</p>"))

    # ----------------------------------------------------------------- auth
    def _session(self) -> dict | None:
        morsel = self._cookies().get(SESSION_COOKIE)
        return unsign(morsel.value, self.cfg.secret) if morsel else None

    def _login(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        nxt = q.get("next", ["/"])[0]
        # Open-redirect guard: only same-site absolute paths. "//evil.test" is a
        # protocol-relative URL, not a path, and must not survive this.
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        state = secrets.token_urlsafe(24)
        signed = sign(
            {"s": state, "n": nxt, "exp": time.time() + 600}, self.cfg.secret
        )
        params = urllib.parse.urlencode(
            {
                "client_id": self.cfg.client_id,
                "redirect_uri": self.cfg.redirect_uri,
                "state": state,
                # No scope parameter at all. We only need the login, which the
                # bare user endpoint gives on an unscoped token, and the
                # authorization decision is made with our own credential.
                "scope": "",
                "allow_signup": "false",
            }
        )
        self._send(
            302,
            b"",
            cookies=[self._cookie_header(STATE_COOKIE, signed, 600)],
            location=f"{GITHUB_AUTHORIZE}?{params}",
        )

    def _callback(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        clear_state = self._cookie_header(STATE_COOKIE, "", 0)

        if "error" in q:
            self._send(
                403,
                page(
                    "GitHub sign-in was cancelled",
                    f"<p>GitHub returned <code>{html.escape(q['error'][0])}</code>.</p>"
                    f'<p><a href="{PREFIX}/login">Try again</a>.</p>',
                ),
                cookies=[clear_state],
            )
            return

        morsel = self._cookies().get(STATE_COOKIE)
        state_payload = unsign(morsel.value, self.cfg.secret) if morsel else None
        if state_payload is None:
            # The state cookie did not come back. Terminal on purpose: redirecting
            # to /login here is exactly how an OAuth proxy ends up in a loop that
            # the user experiences as a browser hang.
            self._send(
                400,
                page(
                    "Sign-in could not be completed",
                    "<p>The browser did not return the sign-in cookie, so this "
                    "callback cannot be verified.</p>"
                    "<p>Usual causes: the page was reached over plain http (the "
                    "cookie is <code>Secure</code>), cookies are blocked for this "
                    "site, or the sign-in was left open for over ten minutes.</p>"
                    f'<p><a href="{PREFIX}/login">Start again</a>.</p>',
                ),
                cookies=[clear_state],
            )
            return

        got = q.get("state", [""])[0]
        if not hmac.compare_digest(got, str(state_payload.get("s", ""))):
            self._send(400, page("Sign-in could not be verified", "<p>State mismatch.</p>"),
                       cookies=[clear_state])
            return

        code = q.get("code", [""])[0]
        token = self.gh.exchange_code(code) if code else None
        if not token:
            self._send(
                502,
                page(
                    "GitHub did not complete sign-in",
                    "<p>The authorization code could not be exchanged. This is "
                    f'usually transient - <a href="{PREFIX}/login">try again</a>.</p>',
                ),
                cookies=[clear_state],
            )
            return

        login = self.gh.login_for(token)
        if not login:
            self._send(502, page("GitHub did not return a username", "<p>Try again.</p>"),
                       cookies=[clear_state])
            return

        allowed, why = authorize(self.gh, self.cfg.repos, login)
        if not allowed:
            if why is None:
                # Nobody said no; the API never answered. Fail closed, but say so,
                # because this is a broken server token, not a rejected person.
                self.log_message("collaborator check inconclusive for %s", login)
                self._send(
                    503,
                    page(
                        "Access could not be checked",
                        "<p>GitHub did not answer the collaborator check, so nobody "
                        "is being let in until it does.</p>"
                        "<p class=m>For the operator: this is normally an expired or "
                        "under-scoped <code>JLENS_GITHUB_TOKEN</code>, or a rate "
                        "limit. See the proxy log.</p>",
                    ),
                    cookies=[clear_state, self._clear_session()],
                )
                return
            self.log_message("denied %s (not a collaborator)", login)
            self._send(403, denied_page(login, self.cfg.repos),
                       cookies=[clear_state, self._clear_session()])
            return

        self.log_message("allowed %s via %s", login, why)
        session = sign(
            {"login": login, "repo": why, "iat": time.time(),
             "exp": time.time() + self.cfg.ttl_s},
            self.cfg.secret,
        )
        nxt = state_payload.get("n", "/")
        if not isinstance(nxt, str) or not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        self._send(
            302,
            b"",
            cookies=[clear_state,
                     self._cookie_header(SESSION_COOKIE, session, self.cfg.ttl_s)],
            location=nxt,
        )

    def _logout(self) -> None:
        self._send(
            200,
            page(
                "Signed out",
                f'<p>Your J-lens session is cleared. <a href="{PREFIX}/login">Sign '
                "in again</a>.</p>"
                '<p class=m>This does not sign you out of GitHub itself. To come '
                'back as a different account, sign out at '
                '<a href="https://github.com/logout">github.com/logout</a> first.</p>',
            ),
            cookies=[self._clear_session()],
        )

    def _whoami(self) -> None:
        sess = self._session()
        body = json.dumps(
            {
                "login": sess.get("login") if sess else None,
                "repo": sess.get("repo") if sess else None,
                "expires_in": int(sess["exp"] - time.time()) if sess else 0,
                "model": self.cfg.model_label or None,
                "branch": self.cfg.branch_label or None,
            }
        ).encode()
        self._send(200 if sess else 401, body, "application/json")

    # ---------------------------------------------------------------- proxy
    def _guarded_proxy(self) -> None:
        sess = self._session()
        if sess is None:
            parsed = urllib.parse.urlparse(self.path)
            nxt = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            # An unauthenticated XHR gets a 401, not a 302 to github.com: the
            # dashboard polls /status from JS and a redirect there would be
            # swallowed as a CORS failure instead of surfacing as "log in again".
            if self._wants_json():
                self._send(401, b'{"error":"not signed in"}', "application/json",
                           cookies=[self._clear_session()])
                return
            self._send(
                302, b"", cookies=[self._clear_session()],
                location=f"{PREFIX}/login?next={urllib.parse.quote(nxt, safe='')}",
            )
            return
        self._forward(sess)

    def _wants_json(self) -> bool:
        if self.headers.get("X-Requested-With") == "XMLHttpRequest":
            return True
        accept = self.headers.get("Accept", "")
        return "application/json" in accept or "text/event-stream" in accept

    def _forward(self, sess: dict) -> None:
        up = urllib.parse.urlparse(self.cfg.upstream)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP
        }
        headers.pop("Cookie", None)  # our session cookie is not the app's business
        headers.pop("Host", None)
        # The dashboard has no notion of users, but the log should still say who
        # ran a slice, and this is the header a future audit would read.
        headers["X-Jlens-User"] = str(sess.get("login", "?"))

        conn_cls = (
            http.client.HTTPSConnection if up.scheme == "https"
            else http.client.HTTPConnection
        )
        try:
            conn = conn_cls(up.hostname, up.port, timeout=UPSTREAM_TIMEOUT_S)
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            self.log_message("upstream unreachable: %s", exc)
            self._send(502, self._upstream_down_page())
            return

        try:
            self._relay(resp)
        finally:
            conn.close()

    def _upstream_down_page(self) -> bytes:
        return page(
            "The GPU instance is not reachable",
            "<p>You are signed in, but nothing is answering on the dashboard "
            "port. The GPU is rented on demand and is probably not running.</p>"
            "<p class=m>For the operator: check the tunnel "
            "(<code>systemctl status jlens-tunnel</code>) and that an instance "
            "exists (<code>deploy/destroy.sh --list</code>). If there is none, "
            "<code>deploy/provision.sh</code> rents one; bootstrap then takes "
            "15-35 minutes before this page works.</p>",
        )

    def _relay(self, resp) -> None:
        """Stream the upstream response through without buffering it.

        /flood/events is server-sent events that stay open for the length of a
        run, so anything that waits for the whole body here would show the user
        a blank page for minutes and then dump the entire run at once.
        """
        out = [(k, v) for k, v in resp.getheaders() if k.lower() not in HOP_BY_HOP]
        has_length = any(k.lower() == "content-length" for k, _ in out)
        streaming = not has_length

        self.send_response(resp.status)
        for k, v in out:
            self.send_header(k, v)
        if streaming:
            # No length and we are not going to chunk it by hand: mark the
            # response as delimited by connection close, which is what an
            # unbounded SSE stream is anyway.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

        if self.command == "HEAD":
            return
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                if streaming:
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # the browser left mid-stream; normal


def build_server(cfg: Config, gh: GitHub | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"cfg": cfg, "gh": gh or GitHub(cfg)})
    srv = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), handler)
    srv.daemon_threads = True
    return srv


def main() -> int:
    try:
        cfg = Config()
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if cfg.listen_host not in ("127.0.0.1", "::1", "localhost"):
        # Not fatal - someone may genuinely be running without Caddy - but it
        # means TLS is not being terminated in front, so say it out loud.
        print(
            f"WARNING: listening on {cfg.listen_host}, not loopback. Session "
            "cookies are Secure; there must be TLS in front of this.",
            file=sys.stderr,
        )
    srv = build_server(cfg)
    print(
        f"authproxy on {cfg.listen_host}:{cfg.listen_port} -> {cfg.upstream}\n"
        f"  public   {cfg.public_url}\n"
        f"  gate     collaborator on {', '.join(cfg.repos)}\n"
        f"  session  {cfg.ttl_s}s, Secure={not cfg.insecure_cookies}",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the auth proxy in front of the J-lens dashboard.

Run from deploy/web:  python3 -m pytest test_authproxy.py -q

These are mostly tests of the authorization boundary, because that is the part
where being wrong is not a bug report but an open dashboard. The GitHub client
is replaced with a fake; nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import authproxy  # noqa: E402
import hubapi  # noqa: E402

SECRET = "x" * 40


def base_env(tmp_path, **over):
    env = {
        "JLENS_OAUTH_CLIENT_ID": "cid",
        "JLENS_OAUTH_CLIENT_SECRET": "csecret",
        "JLENS_GITHUB_TOKEN": "ghp_servertoken",
        "JLENS_SESSION_SECRET": SECRET,
        "JLENS_PUBLIC_URL": "https://jlens.example.test",
        "JLENS_UPSTREAM": "http://127.0.0.1:1",  # nothing listens: 502 on purpose
        "JLENS_LISTEN_HOST": "127.0.0.1",
        "JLENS_LISTEN_PORT": "0",
        "JLENS_RUN_DB": str(tmp_path / "runs.db"),
        "JLENS_HUB_DIST": str(tmp_path / "dist"),
    }
    env.update(over)
    return env


# ------------------------------------------------------------------- config


def test_config_rejects_a_short_secret(tmp_path):
    with pytest.raises(authproxy.ConfigError, match="SESSION_SECRET"):
        authproxy.Config(base_env(tmp_path, JLENS_SESSION_SECRET="short"))


def test_config_rejects_http_public_url(tmp_path):
    """A Secure cookie over plain http is silently dropped by the browser, and
    the symptom is an endless login loop. Refuse to start instead."""
    with pytest.raises(authproxy.ConfigError, match="not https"):
        authproxy.Config(base_env(tmp_path, JLENS_PUBLIC_URL="http://jlens.example.test"))


def test_config_allows_http_only_with_the_explicit_escape_hatch(tmp_path):
    cfg = authproxy.Config(
        base_env(tmp_path, JLENS_PUBLIC_URL="http://localhost:7870",
                 JLENS_INSECURE_COOKIES="1")
    )
    assert cfg.insecure_cookies is True


def test_config_rejects_a_malformed_repo(tmp_path):
    with pytest.raises(authproxy.ConfigError, match="owner/repo"):
        authproxy.Config(base_env(tmp_path, JLENS_AUTH_REPOS="jacobian-lens"))


def test_config_defaults_to_the_public_fork(tmp_path):
    assert authproxy.Config(base_env(tmp_path)).repos == ["Nathan2589/jacobian-lens"]


# ------------------------------------------------------------------ signing


def test_unsign_rejects_a_tampered_payload():
    token = authproxy.sign({"login": "nobody", "exp": time.time() + 60}, SECRET.encode())
    body, mac = token.split(".", 1)
    forged = authproxy.sign({"login": "admin", "exp": time.time() + 60}, b"other-secret")
    assert authproxy.unsign(f"{forged.split('.')[0]}.{mac}", SECRET.encode()) is None
    assert authproxy.unsign(token, SECRET.encode())["login"] == "nobody"
    assert body  # sanity


def test_unsign_rejects_an_expired_session():
    token = authproxy.sign({"login": "nathan", "exp": time.time() - 1}, SECRET.encode())
    assert authproxy.unsign(token, SECRET.encode()) is None


def test_unsign_rejects_a_session_with_no_expiry():
    """A payload with no exp must not be treated as 'never expires'."""
    token = authproxy.sign({"login": "nathan"}, SECRET.encode())
    assert authproxy.unsign(token, SECRET.encode()) is None


@pytest.mark.parametrize("junk", ["", "no-dot", "a.b", "....", "x" * 200])
def test_unsign_survives_junk(junk):
    assert authproxy.unsign(junk, SECRET.encode()) is None


# ------------------------------------------------------------ authorization


class FakeGitHub:
    def __init__(self, collaborators: dict[str, set[str]], fail: set[str] | None = None):
        self.collaborators = collaborators
        self.fail = fail or set()
        self.calls: list[tuple[str, str]] = []

    def is_collaborator(self, repo, login):
        self.calls.append((repo, login))
        if repo in self.fail:
            return None
        return login in self.collaborators.get(repo, set())


def test_authorize_allows_a_collaborator():
    gh = FakeGitHub({"o/public": {"nathan"}})
    assert authproxy.authorize(gh, ["o/public"], "nathan") == (True, "o/public")


def test_authorize_denies_a_stranger():
    gh = FakeGitHub({"o/public": {"nathan"}})
    allowed, why = authproxy.authorize(gh, ["o/public"], "torvalds")
    assert allowed is False and why == "denied"


def test_authorize_accepts_any_one_of_several_repos():
    """The measured reality: the teammate is on the private parent but not on
    the public fork, so the list is an any-of."""
    gh = FakeGitHub({"o/public": {"nathan"}, "o/private": {"nathan", "junior"}})
    assert authproxy.authorize(gh, ["o/public", "o/private"], "junior") == (True, "o/private")


def test_authorize_fails_closed_when_github_errors():
    """A rate-limited or expired server token must never read as 'allowed', and
    must be distinguishable from a real rejection so the operator can tell an
    outage from a stranger."""
    gh = FakeGitHub({"o/public": {"nathan"}}, fail={"o/public"})
    allowed, why = authproxy.authorize(gh, ["o/public"], "nathan")
    assert allowed is False
    assert why is None  # None => "could not check", not "not a collaborator"


def test_authorize_prefers_a_definite_yes_over_an_error():
    gh = FakeGitHub({"o/private": {"junior"}}, fail={"o/public"})
    assert authproxy.authorize(gh, ["o/public", "o/private"], "junior") == (True, "o/private")


# ------------------------------------------------------- end-to-end over HTTP


class Server:
    def __init__(self, cfg, gh):
        self.srv = authproxy.build_server(cfg, gh)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path, cookie=None, follow=False):
        req = urllib.request.Request(self.url(path))
        if cookie:
            req.add_header("Cookie", cookie)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **kw):
                return None

        opener = urllib.request.build_opener(*([] if follow else [NoRedirect]))
        try:
            with opener.open(req, timeout=10) as resp:
                return resp.status, dict(resp.getheaders()), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def server(tmp_path):
    cfg = authproxy.Config(base_env(tmp_path))
    gh = FakeGitHub({"Nathan2589/jacobian-lens": {"Nathan2589"}})
    s = Server(cfg, gh)
    yield s, cfg, gh
    s.stop()


def session_cookie(cfg, login="Nathan2589"):
    token = authproxy.sign(
        {"login": login, "repo": cfg.repos[0], "exp": time.time() + 600}, cfg.secret
    )
    return f"{authproxy.SESSION_COOKIE}={token}"


def test_health_needs_no_session_and_never_touches_the_gpu(server):
    """Caddy's health check hits this. If it proxied to the dashboard it would
    write an access line on the GPU every few seconds and defeat the idle
    reaper that stops the instance billing."""
    s, _, _ = server
    status, _, body = s.get("/_jlens/health")
    assert status == 200 and body == b"ok\n"


def test_an_anonymous_request_is_sent_to_github_not_to_the_dashboard(server):
    s, _, _ = server
    status, headers, _ = s.get("/")
    assert status == 302
    assert headers["Location"].startswith("/_jlens/login")


def test_login_requests_no_scopes(server):
    """Authorization is done with our own token, so the visitor's grant needs
    nothing. Asking for `repo` or `read:org` here would be over-requesting."""
    s, _, _ = server
    status, headers, _ = s.get("/_jlens/login")
    assert status == 302
    q = urllib.parse.parse_qs(urllib.parse.urlparse(headers["Location"]).query)
    assert q.get("scope", [""])[0] == ""
    assert "github.com/login/oauth/authorize" in headers["Location"]


def test_login_sets_a_secure_httponly_lax_state_cookie(server):
    s, _, _ = server
    _, headers, _ = s.get("/_jlens/login")
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie


def test_login_refuses_an_offsite_next(server):
    """?next=//evil.test is a protocol-relative URL, not a path."""
    s, cfg, _ = server
    _, headers, _ = s.get("/_jlens/login?next=//evil.test/x")
    state = None
    for part in headers["Set-Cookie"].split(";"):
        if part.strip().startswith(authproxy.STATE_COOKIE):
            state = part.split("=", 1)[1]
    assert authproxy.unsign(state, cfg.secret)["n"] == "/"


def test_callback_without_the_state_cookie_is_terminal_not_a_redirect(server):
    """The loop this project must not ship: bouncing a failed callback back to
    /login, forever, with no explanation."""
    s, _, _ = server
    status, headers, body = s.get("/_jlens/callback?code=x&state=y")
    assert status == 400
    assert "Location" not in headers
    assert b"did not return the sign-in cookie" in body


def test_a_valid_session_reaches_the_proxy_and_gets_a_readable_502(server):
    """Nothing listens upstream in the tests. A signed-in user must get the
    'GPU is not reachable' page, not a traceback."""
    s, cfg, _ = server
    status, _, body = s.get("/status", cookie=session_cookie(cfg))
    assert status == 502
    assert b"GPU instance is not reachable" in body


def test_root_falls_back_to_the_hub_when_no_gpu_is_rented(server):
    """With no instance the dashboard cannot answer, but the hub still can, and
    that is where a person can see cost and history."""
    s, cfg, _ = server
    status, headers, _ = s.get("/", cookie=session_cookie(cfg))
    assert status == 302 and headers["Location"] == "/hub/"


def test_a_forged_session_is_not_a_session(server):
    s, _, _ = server
    forged = authproxy.sign(
        {"login": "attacker", "exp": time.time() + 600}, b"a-different-secret-entirely"
    )
    status, headers, _ = s.get("/", cookie=f"{authproxy.SESSION_COOKIE}={forged}")
    assert status == 302 and headers["Location"].startswith("/_jlens/login")


def test_the_api_returns_401_json_not_a_redirect(server):
    """The hub fetches this. A 302 to github.com would surface in the browser as
    an opaque CORS error rather than 'your session expired'."""
    s, _, _ = server
    status, headers, _ = s.get("/_jlens/api/state")
    assert status == 401
    assert headers["Content-Type"].startswith("application/json")


def test_state_reports_absent_gpu_without_pretending_to_know_more(server):
    s, cfg, _ = server
    status, _, body = s.get("/_jlens/api/state", cookie=session_cookie(cfg))
    assert status == 200
    state = json.loads(body)
    assert state["gpu"]["reachable"] is False
    assert state["gpu"]["status"] == "absent"
    assert state["gpu"]["dashboardUrl"] is None
    assert state["user"]["login"] == "Nathan2589"
    # No vast key in the tests, so the panel must say why rather than show zeros.
    assert state["instance"] is None
    assert "vast.ai API key" in state["instanceUnavailableReason"]


def test_runs_start_empty_with_a_zeroed_summary(server):
    s, cfg, _ = server
    status, _, body = s.get("/_jlens/api/runs", cookie=session_cookie(cfg))
    assert status == 200
    payload = json.loads(body)
    assert payload["runs"] == []
    # The hub renders these tiles before any run exists, so an empty log must
    # produce zeros and nulls rather than a missing key.
    assert payload["stats"] == {
        "total": 0, "ok": 0, "failed": 0, "last24h": 0,
        "lastAt": None, "medianMs": None,
    }


def test_run_store_stats_summarise_the_log(tmp_path):
    store = hubapi.RunStore(str(tmp_path / "r.db"))
    now = time.time()
    for i, (ok, dur) in enumerate([(1, 100), (1, 300), (1, 200), (0, None)]):
        store.record(at=now - i, user="u", kind="slice", prompt="p", ok=ok,
                     duration_ms=dur, error=None if ok else "boom")
    store.record(at=now - 200_000, user="u", kind="slice", prompt="old", ok=1,
                 duration_ms=50)
    stats = store.stats()
    assert stats["total"] == 5 and stats["ok"] == 4 and stats["failed"] == 1
    assert stats["last24h"] == 4  # the 200_000s-old one is outside the window
    # Median of the successful durations (50, 100, 200, 300), not the mean.
    assert stats["medianMs"] == 200


def test_hub_requires_a_session(server):
    s, _, _ = server
    status, headers, _ = s.get("/hub/")
    assert status == 302 and headers["Location"].startswith("/_jlens/login")


def test_hub_says_so_when_the_bundle_is_missing(server):
    s, cfg, _ = server
    status, _, body = s.get("/hub/", cookie=session_cookie(cfg))
    assert status == 503 and b"not built" in body


# --------------------------------------------------------------- hub internals


def test_static_site_refuses_to_escape_its_root(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html>")
    (root / "assets" / "app.js").write_text("//")
    (tmp_path / "secret.txt").write_text("nope")

    site = hubapi.StaticSite(str(root))
    assert site.present
    assert site.resolve("/hub/assets/app.js").endswith("assets/app.js")
    # Traversal must not reach a sibling file; it falls back to index.html.
    assert site.resolve("/hub/../secret.txt").endswith("index.html")
    assert site.resolve("/hub/%2e%2e/secret.txt").endswith("index.html")
    # Unknown client-side route -> index.html, not a 404.
    assert site.resolve("/hub/anything").endswith("index.html")


def test_static_site_caches_assets_but_never_index(tmp_path):
    _, cache = hubapi.StaticSite.headers_for("/x/dist/assets/index-abc123.js")
    assert "immutable" in cache
    _, cache = hubapi.StaticSite.headers_for("/x/dist/index.html")
    assert cache == "no-store"


def test_summarise_request_only_matches_actual_runs():
    assert hubapi.summarise_request("/status", b"{}") is None
    assert hubapi.summarise_request("/flood/events", None) is None
    got = hubapi.summarise_request("/run", json.dumps({"prompt": "hi", "max_seq": 64}).encode())
    assert got == {"kind": "slice", "prompt": "hi", "max_seq": 64}
    assert hubapi.summarise_request("/flood/start", b"{}")["kind"] == "flood"


def test_summarise_request_truncates_a_huge_prompt():
    body = json.dumps({"prompt": "a" * 5000}).encode()
    assert len(hubapi.summarise_request("/run", body)["prompt"]) == 400


def test_summarise_request_survives_a_malformed_body():
    assert hubapi.summarise_request("/run", b"not json") == {"kind": "slice"}


def test_run_store_round_trip(tmp_path):
    store = hubapi.RunStore(str(tmp_path / "r.db"))
    store.record(at=time.time(), user="nathan", kind="slice", prompt="p",
                 model="m", branch="b", max_seq=64, top_n=5, gen_tokens=32,
                 duration_ms=1200, ok=1, error=None)
    runs = store.recent()
    assert len(runs) == 1 and runs[0]["ok"] is True and runs[0]["user"] == "nathan"


def test_run_store_newest_first_and_limited(tmp_path):
    store = hubapi.RunStore(str(tmp_path / "r.db"))
    for i in range(5):
        store.record(at=1000 + i, user="u", kind="slice", prompt=f"p{i}", ok=1)
    runs = store.recent(limit=3)
    assert [r["prompt"] for r in runs] == ["p4", "p3", "p2"]


def test_status_cache_goes_stale(tmp_path):
    cache = hubapi.StatusCache()
    assert cache.snapshot() is None
    cache.observe(json.dumps({"status": "ready", "detail": "27B"}).encode())
    assert cache.snapshot()["status"] == "ready"
    # Stale means "we no longer know", not "still ready".
    assert cache.snapshot(max_age_s=0.0) is None


def test_status_cache_ignores_junk(tmp_path):
    cache = hubapi.StatusCache()
    cache.observe(b"<html>not json</html>")
    assert cache.snapshot() is None


def test_vast_client_without_a_key_explains_itself(tmp_path):
    client = hubapi.VastClient(str(tmp_path / "absent-key"), str(tmp_path / "absent-id"))
    snap = client.snapshot()
    assert "reason" in snap and "API key" in snap["reason"]

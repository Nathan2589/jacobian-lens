"""State, run history and static assets for the J-lens experiment hub.

Imported by authproxy.py. Everything here runs on the always-on EC2 box, never
on the GPU.

The one rule that shapes this whole module
------------------------------------------
**Nothing in here may send an HTTP request to the dashboard.**

bootstrap.sh arms a reaper on the GPU instance that destroys it after
JLENS_IDLE_KILL_S with no dashboard traffic, and its heartbeat is the mtime of
the uvicorn access log - which uvicorn writes one line to per request. A hub
that polled `/status` every ten seconds to draw a health badge would look
exactly like a human using the dashboard, the instance would never go idle, and
a $0.43/hr card would run until the vast.ai credit was gone. The cost section of
deploy/README.md exists because forgetting to destroy is the real failure mode
here; a status widget that quietly disables the safety net is the same bug with
better manners.

So reachability is a bare TCP connect (opens a socket, sends no request, writes
no access line) and the richer detail is learned *passively*: when someone has
the dashboard open, their own `/status` polls pass through this proxy, and we
keep the last body we saw. Traffic we are already carrying is free to read.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VAST_API = "https://console.vast.ai/api/v0"

# A TCP connect is cheap but not free, and the hub polls every 10s. Cache it.
_PROBE_TTL_S = 5.0
# vast.ai's API is rate limited and the numbers move slowly (cost accrues by the
# second but nobody needs it to the second).
_VAST_TTL_S = 30.0


# ------------------------------------------------------------------ run store


class RunStore:
    """Every slice and flood run that passed through the proxy.

    On the proxy box rather than the GPU on purpose: the GPU is destroyed
    routinely - by destroy.sh, by the idle reaper, by a host failure - and a
    history that dies with it is not a history. dashboard.py says "no
    persistence here by design"; this is where the persistence went.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # check_same_thread=False + an explicit lock: ThreadingHTTPServer serves
        # each request on its own thread and sqlite3 connections are not
        # thread-safe by default.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     at REAL NOT NULL,
                     user TEXT NOT NULL,
                     kind TEXT NOT NULL,
                     prompt TEXT,
                     model TEXT,
                     branch TEXT,
                     max_seq INTEGER,
                     top_n INTEGER,
                     gen_tokens INTEGER,
                     duration_ms INTEGER,
                     ok INTEGER NOT NULL,
                     error TEXT
                   )"""
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS runs_at ON runs (at DESC)")
            self._db.commit()

    def record(self, **kw) -> None:
        cols = (
            "at", "user", "kind", "prompt", "model", "branch",
            "max_seq", "top_n", "gen_tokens", "duration_ms", "ok", "error",
        )
        values = [kw.get(c) for c in cols]
        with self._lock:
            self._db.execute(
                f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                values,
            )
            self._db.commit()

    def stats(self) -> dict:
        """Summary of the experiment log itself.

        Deliberately independent of vast.ai: this is what the hub can always
        show, including on a box with no API key and with no GPU rented, which
        is the resting state most of the time.
        """
        day_ago = time.time() - 86400
        with self._lock:
            row = self._db.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok,
                          MAX(at) AS last_at
                   FROM runs"""
            ).fetchone()
            recent = self._db.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE at >= ?", (day_ago,)
            ).fetchone()
            # Median, not mean: one 40-minute flood run would drag a mean into
            # uselessness as a description of a typical slice.
            durations = [
                r[0] for r in self._db.execute(
                    "SELECT duration_ms FROM runs WHERE duration_ms IS NOT NULL AND ok"
                ).fetchall()
            ]
        total = row["total"] or 0
        durations.sort()
        median = durations[len(durations) // 2] if durations else None
        return {
            "total": total,
            "ok": row["ok"] or 0,
            "failed": total - (row["ok"] or 0),
            "last24h": recent["n"] or 0,
            "lastAt": row["last_at"],
            "medianMs": median,
        }

    def recent(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM runs ORDER BY at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "at": r["at"],
                "user": r["user"],
                "kind": r["kind"],
                "prompt": r["prompt"] or "",
                "model": r["model"],
                "branch": r["branch"],
                "maxSeq": r["max_seq"],
                "topN": r["top_n"],
                "genTokens": r["gen_tokens"],
                "durationMs": r["duration_ms"],
                "ok": bool(r["ok"]),
                "error": r["error"],
            }
            for r in rows
        ]


def summarise_request(path: str, body: bytes | None) -> dict | None:
    """Turn a proxied dashboard request into a run record, or None if it is not
    one. Only /run and /flood/start are runs; /status polls are not."""
    route = urllib.parse.urlparse(path).path
    if route not in ("/run", "/flood/start"):
        return None
    fields: dict = {"kind": "slice" if route == "/run" else "flood"}
    try:
        payload = json.loads(body or b"{}")
    except Exception:  # noqa: BLE001 - a malformed body is still a run attempt
        return fields
    if not isinstance(payload, dict):
        return fields
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        # The history is a scannable list, not an archive. Long prompts are
        # truncated here rather than at render time so the database stays small.
        fields["prompt"] = prompt[:400]
    for src, dst in (("max_seq", "max_seq"), ("top_n", "top_n"), ("gen_tokens", "gen_tokens")):
        v = payload.get(src)
        if isinstance(v, (int, float)):
            fields[dst] = int(v)
    return fields


# --------------------------------------------------------------- gpu presence


class UpstreamProbe:
    """Is anything listening on the far end of the tunnel?

    Deliberately a bare TCP connect. It opens a socket and closes it without
    sending bytes, so uvicorn logs nothing and the GPU's idle reaper keeps
    counting. See the module docstring.
    """

    def __init__(self, upstream: str) -> None:
        parsed = urllib.parse.urlparse(upstream)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._cached: tuple[float, bool] = (0.0, False)
        self._lock = threading.Lock()

    def reachable(self) -> bool:
        with self._lock:
            at, value = self._cached
            if time.time() - at < _PROBE_TTL_S:
                return value
        try:
            with socket.create_connection((self.host, self.port), timeout=1.5):
                ok = True
        except OSError:
            ok = False
        with self._lock:
            self._cached = (time.time(), ok)
        return ok


class StatusCache:
    """The last /status body the proxy saw go past, and when.

    Costs nothing: it reads a response we were forwarding anyway. Goes stale
    once nobody has the dashboard open, which is correct - that is also when the
    instance is heading for the idle reaper.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = 0.0
        self._body: dict = {}

    def observe(self, body: bytes) -> None:
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001
            return
        if not isinstance(parsed, dict):
            return
        with self._lock:
            self._at, self._body = time.time(), parsed

    def snapshot(self, max_age_s: float = 120.0) -> dict | None:
        with self._lock:
            if self._at and time.time() - self._at < max_age_s:
                return dict(self._body)
        return None


# ------------------------------------------------------------------- vast.ai


class VastClient:
    """Instance cost, uptime and account credit.

    Optional. With no key the hub still works; it just shows the instance panel
    in a reduced form saying why, rather than rendering zeros that look like
    real numbers.
    """

    def __init__(self, key_file: str | None, instance_file: str | None) -> None:
        self.key_file = key_file
        self.instance_file = instance_file
        self._lock = threading.Lock()
        self._cached: tuple[float, dict | None] = (0.0, None)

    @property
    def key(self) -> str | None:
        env = os.environ.get("VAST_API_KEY")
        if env:
            return env.strip()
        if self.key_file and os.path.exists(self.key_file):
            try:
                with open(self.key_file) as fh:
                    return fh.read().strip() or None
            except OSError:
                return None
        return None

    def instance_id(self) -> str | None:
        env = os.environ.get("JLENS_INSTANCE_ID")
        if env:
            return env.strip()
        if self.instance_file and os.path.exists(self.instance_file):
            try:
                with open(self.instance_file) as fh:
                    return fh.read().strip() or None
            except OSError:
                return None
        return None

    def _call(self, path: str, method: str = "GET") -> tuple[int, dict]:
        key = self.key
        if not key:
            return 0, {}
        req = urllib.request.Request(f"{VAST_API}{path}", method=method)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Accept", "application/json")
        if method == "DELETE":
            req.add_header("Content-Type", "application/json")
            req.data = b"{}"
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except Exception:  # noqa: BLE001
                return exc.code, {}
        except Exception:  # noqa: BLE001 - network flake is "unknown", not a crash
            return 0, {}

    def snapshot(self) -> dict:
        """{"reason": str} when unavailable, otherwise the instance figures."""
        with self._lock:
            at, value = self._cached
            if value is not None and time.time() - at < _VAST_TTL_S:
                return value

        result = self._compute()
        with self._lock:
            self._cached = (time.time(), result)
        return result

    def invalidate(self) -> None:
        with self._lock:
            self._cached = (0.0, None)

    def _compute(self) -> dict:
        if not self.key:
            return {"reason": "No vast.ai API key on the proxy box."}
        iid = self.instance_id()
        if not iid:
            return {"reason": "No instance id recorded - nothing has been provisioned."}

        status, body = self._call(f"/instances/{iid}/?owner=me")
        if status != 200:
            return {"reason": f"vast.ai API returned HTTP {status or 'no response'}."}
        inst = body.get("instances")
        if not inst:
            # 200 with a null instance is how vast reports "already destroyed".
            return {"id": None, "destroyed": True}

        dph = inst.get("dph_total")
        start = inst.get("start_date")
        uptime = time.time() - start if isinstance(start, (int, float)) else None
        cost = (dph / 3600.0 * uptime) if (dph and uptime) else None

        credit = None
        st, user = self._call("/users/current/")
        if st == 200:
            credit = user.get("credit")

        return {
            "id": str(iid),
            "gpuName": inst.get("gpu_name"),
            "dphTotal": dph,
            "uptimeS": uptime,
            "costSoFar": cost,
            "credit": credit,
            "runwayH": (credit / dph) if (credit and dph) else None,
            "actualStatus": inst.get("actual_status"),
        }

    def destroy(self, instance_id: str) -> tuple[bool, str]:
        if not self.key:
            return False, "no vast.ai API key on the proxy box"
        status, _ = self._call(f"/instances/{instance_id}/", method="DELETE")
        self.invalidate()
        if status == 200:
            return True, f"instance {instance_id} destroyed"
        return False, f"vast.ai returned HTTP {status or 'no response'}"


# --------------------------------------------------------------------- state


def build_state(
    *,
    session: dict,
    repos_label: str,
    probe: UpstreamProbe,
    status_cache: StatusCache,
    vast: VastClient,
    idle_kill_s: int | None,
) -> dict:
    reachable = probe.reachable()
    observed = status_cache.snapshot()

    if not reachable:
        gpu_status, detail = "absent", "No instance is answering on the dashboard port."
    elif observed:
        gpu_status = observed.get("status") or "loading"
        detail = observed.get("detail") or ""
        if gpu_status not in ("ready", "loading", "error"):
            gpu_status = "loading"
    else:
        # The port is open but nobody has loaded the dashboard recently enough
        # for us to have seen a /status go past. Say that, rather than guessing.
        gpu_status, detail = "loading", "Instance is up; open the dashboard for live detail."

    inst = vast.snapshot()
    reason = inst.get("reason")

    return {
        "user": {
            "login": session.get("login"),
            "repo": session.get("repo") or repos_label,
            "expiresIn": int(session.get("exp", 0) - time.time()),
        },
        "deployment": {
            "model": os.environ.get("JLENS_MODEL_ID") or None,
            "branch": os.environ.get("JLENS_BRANCH") or None,
            "lensRepo": os.environ.get("JLENS_LENS_REPO") or None,
            "precision": os.environ.get("JLENS_PRECISION") or None,
        },
        "gpu": {
            "reachable": reachable,
            "status": gpu_status,
            "detail": detail,
            "dashboardUrl": "/" if reachable else None,
        },
        "instance": None if reason else {
            "id": inst.get("id"),
            "gpuName": inst.get("gpuName"),
            "dphTotal": inst.get("dphTotal"),
            "uptimeS": inst.get("uptimeS"),
            "costSoFar": inst.get("costSoFar"),
            "credit": inst.get("credit"),
            "runwayH": inst.get("runwayH"),
            "idleKillS": idle_kill_s,
            "idleForS": None,  # only the GPU box knows; asking it would cost an access line
        },
        "instanceUnavailableReason": reason,
    }


# ------------------------------------------------------------ static assets


class StaticSite:
    """Serves the built hub bundle from disk.

    Small and hand-rolled because the alternative is another dependency in the
    one process that decides who gets in.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    @property
    def present(self) -> bool:
        return os.path.isfile(os.path.join(self.root, "index.html"))

    def resolve(self, url_path: str) -> str | None:
        """Map a /hub/... URL to a file, or None. Returns index.html for paths
        with no file so client-side routing works."""
        rel = url_path.split("?", 1)[0]
        rel = urllib.parse.unquote(rel)
        rel = rel[len("/hub"):] if rel.startswith("/hub") else rel
        rel = rel.lstrip("/")
        # posixpath.normpath collapses ".." before we join, so a crafted
        # /hub/../../etc/passwd cannot escape the bundle directory.
        candidate = os.path.abspath(os.path.join(self.root, posixpath.normpath("/" + rel).lstrip("/")))
        if not (candidate == self.root or candidate.startswith(self.root + os.sep)):
            return None
        if os.path.isfile(candidate):
            return candidate
        index = os.path.join(self.root, "index.html")
        return index if os.path.isfile(index) else None

    @staticmethod
    def headers_for(filepath: str) -> tuple[str, str]:
        ctype, _ = mimetypes.guess_type(filepath)
        ctype = ctype or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        # Vite fingerprints everything under assets/, so those are immutable.
        # index.html must never be cached or a deploy is invisible until a hard
        # reload, which is the kind of thing that eats an afternoon.
        if "/assets/" in filepath.replace(os.sep, "/"):
            cache = "public, max-age=31536000, immutable"
        else:
            cache = "no-store"
        return ctype, cache

/** The hub's whole server contract. Everything is served by authproxy on the
 *  always-on box, behind the collaborator gate, so there is no auth handling
 *  here - a 401 means the session expired and the only sane response is to send
 *  the browser back through the OAuth flow. */

export type GpuStatus = "ready" | "loading" | "error" | "absent";

export interface HubState {
  user: { login: string; repo: string; expiresIn: number } | null;
  deployment: { model: string | null; branch: string | null; lensRepo: string | null; precision: string | null };
  gpu: {
    reachable: boolean;
    status: GpuStatus;
    detail: string;
    /** null when no instance is rented - the common case, and not an error. */
    dashboardUrl: string | null;
  };
  instance: {
    id: string | null;
    gpuName: string | null;
    dphTotal: number | null;
    uptimeS: number | null;
    costSoFar: number | null;
    credit: number | null;
    /** Hours of runway left at the current rate. */
    runwayH: number | null;
    idleKillS: number | null;
    idleForS: number | null;
  } | null;
  /** Set when the proxy holds no vast.ai key: the hub then shows the GPU panel
   *  in a reduced form instead of pretending the numbers are zero. */
  instanceUnavailableReason: string | null;
}

export interface RunRecord {
  id: number;
  at: number;
  user: string;
  kind: "slice" | "flood";
  prompt: string;
  model: string | null;
  branch: string | null;
  maxSeq: number | null;
  topN: number | null;
  genTokens: number | null;
  durationMs: number | null;
  ok: boolean;
  error: string | null;
}

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" }, credentials: "same-origin" });
  if (res.status === 401) {
    // Session gone. A full navigation, not a fetch: the OAuth flow needs the
    // browser's address bar, and an XHR redirect to github.com would be
    // swallowed as a CORS failure with no visible cause.
    window.location.href = `/_jlens/login?next=${encodeURIComponent(window.location.pathname)}`;
    throw new ApiError("session expired", 401);
  }
  if (!res.ok) throw new ApiError(`${path} returned ${res.status}`, res.status);
  return res.json() as Promise<T>;
}

export const api = {
  state: () => get<HubState>("/_jlens/api/state"),
  runs: (limit = 50) => get<{ runs: RunRecord[] }>(`/_jlens/api/runs?limit=${limit}`),
  destroyInstance: async (id: string) => {
    const res = await fetch("/_jlens/api/instance/destroy", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest" },
      credentials: "same-origin",
      body: JSON.stringify({ id }),
    });
    if (!res.ok) throw new ApiError(await res.text(), res.status);
    return res.json() as Promise<{ destroyed: boolean; detail: string }>;
  },
};

export { ApiError };

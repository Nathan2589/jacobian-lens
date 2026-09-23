import { useCallback, useEffect, useRef, useState } from "react";

/** Poll a loader on an interval, pausing while the tab is hidden.
 *
 *  The pause is not a nicety. The GPU instance reaps itself after
 *  JLENS_IDLE_KILL_S with no dashboard traffic, and the hub's own polling goes
 *  to the proxy, not the GPU - but a tab left open for days still costs the
 *  proxy a request every few seconds for nothing. Hidden tabs stop.
 */
export function usePoll<T>(loader: () => Promise<T>, intervalMs: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const refresh = useCallback(async () => {
    try {
      setData(await loaderRef.current());
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      if (cancelled || document.hidden) return;
      await refresh();
    };

    void refresh();
    timer = window.setInterval(tick, intervalMs);
    const onVisible = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh, intervalMs]);

  return { data, error, loading, refresh };
}

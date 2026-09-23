import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** "2h 14m" / "47s". Durations here are instance uptimes and slice latencies, and
 *  a bare seconds count is unreadable at both ends of that range. */
export function humanDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "--";
  // Slices come back in a few hundred ms. Flooring those to "0s" made every fast
  // run look like a no-op, which is the opposite of what the number is for.
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

export function usd(n: number | null | undefined, places = 2): string {
  if (n == null || !Number.isFinite(n)) return "--";
  return `$${n.toFixed(places)}`;
}

export function relativeTime(epochSeconds: number): string {
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

import { Activity, CircleCheck, Gauge, History } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import type { RunStats } from "@/lib/api";
import { humanDuration, relativeTime } from "@/lib/utils";
import StatTile from "./StatTile";

/** The always-available half of the hub. Every figure here comes from the
 *  proxy's own run log, so it renders with no vast.ai key and no GPU rented —
 *  which is the resting state, and exactly when someone is deciding whether to
 *  spend money renting one. */
export default function ExperimentStats({ stats }: { stats: RunStats | null }) {
  if (!stats) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-[7.5rem] w-full" />
        ))}
      </div>
    );
  }

  const successPct = stats.total ? Math.round((stats.ok / stats.total) * 100) : null;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatTile
        label="Runs recorded"
        value={stats.total}
        icon={History}
        footnote={stats.lastAt ? `last ${relativeTime(stats.lastAt)}` : "nothing yet"}
      />
      <StatTile label="Last 24 hours" value={stats.last24h} icon={Activity} />
      <StatTile
        label="Succeeded"
        value={successPct}
        unit={successPct === null ? undefined : "%"}
        icon={CircleCheck}
        tone={successPct !== null && successPct < 80 ? "warning" : "default"}
        footnote={stats.failed ? `${stats.failed} failed` : undefined}
      />
      <StatTile
        label="Median slice"
        value={stats.medianMs === null ? null : humanDuration(stats.medianMs / 1000)}
        precise
        icon={Gauge}
        footnote="successful runs only"
      />
    </div>
  );
}

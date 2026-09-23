import { CircleAlert, CircleCheck, FlaskConical, Waves } from "lucide-react";
import AnimatedList from "@/components/reactbits/AnimatedList";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { RunRecord } from "@/lib/api";
import { humanDuration, relativeTime } from "@/lib/utils";
import EmptyState from "./EmptyState";

/** The reason the hub exists on the always-on box rather than on the GPU: this
 *  list outlives the instance. Every slice and flood run that passed through the
 *  proxy is here, including ones from instances that were destroyed weeks ago. */
export default function RunHistory({ runs, loading }: { runs: RunRecord[] | null; loading: boolean }) {
  if (loading && !runs) {
    return (
      <div className="flex flex-col gap-2 p-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    );
  }

  if (!runs || runs.length === 0) {
    return (
      <EmptyState icon={FlaskConical} title="No runs recorded yet">
        Every prompt sent through the dashboard is logged here with the model and branch that
        served it. Run a slice and it will appear.
      </EmptyState>
    );
  }

  return (
    <AnimatedList
      items={runs}
      className="divide-y divide-border"
      render={(run) => (
        <div className="flex items-start gap-3 px-4 py-3 transition-colors hover:bg-accent/40">
          <div className="mt-0.5 shrink-0">
            {run.ok ? (
              <CircleCheck className="size-4 text-emerald-500/80" aria-label="succeeded" />
            ) : (
              <CircleAlert className="size-4 text-rose-500/80" aria-label="failed" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              {run.kind === "flood" ? (
                <Badge variant="secondary" className="gap-1">
                  <Waves className="size-3" aria-hidden />
                  flood
                </Badge>
              ) : null}
              <p className="min-w-0 truncate font-mono text-[13px] text-foreground">
                {run.prompt || <span className="text-muted-foreground">(no prompt)</span>}
              </p>
            </div>
            <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span>{run.user}</span>
              <span>{relativeTime(run.at)}</span>
              {run.branch ? <span className="font-mono">{run.branch}</span> : null}
              {run.maxSeq ? <span className="tabular-nums">{run.maxSeq} tok</span> : null}
              {run.error ? <span className="text-rose-400">{run.error}</span> : null}
            </p>
          </div>
          <span className="shrink-0 tabular-nums text-xs text-muted-foreground">
            {humanDuration(run.durationMs ? run.durationMs / 1000 : null)}
          </span>
        </div>
      )}
    />
  );
}

import { CircleDollarSign, Clock, Cpu, KeyRound, Loader2, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type HubState } from "@/lib/api";
import { humanDuration, usd } from "@/lib/utils";
import StatTile from "./StatTile";

/** Rented-GPU lifecycle. The destroy control is here and nowhere else, and it is
 *  deliberately a two-step: the README's own cost section says forgetting to
 *  destroy is the real risk, which makes this the highest-value button on the
 *  page - and the one most worth making impossible to hit by accident. */
export default function InstancePanel({
  state,
  onChanged,
}: {
  state: HubState;
  onChanged: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inst = state.instance;

  if (state.instanceUnavailableReason) {
    return (
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-base">
            <KeyRound className="size-4 text-muted-foreground" aria-hidden />
            Instance details unavailable
          </CardTitle>
          <CardDescription>{state.instanceUnavailableReason}</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Cost, uptime and the destroy control are the only things that need it — run history and
          deployment state below are unaffected.
        </CardContent>
      </Card>
    );
  }

  if (!inst || !inst.id) {
    return null;
  }

  const runwayTone = inst.runwayH == null ? "default" : inst.runwayH < 4 ? "danger" : inst.runwayH < 12 ? "warning" : "default";

  return (
    <section className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Accrued cost"
          value={inst.costSoFar}
          prefix="$"
          decimals={2}
          icon={CircleDollarSign}
          footnote={inst.dphTotal != null ? `${usd(inst.dphTotal, 3)}/hr including storage` : undefined}
        />
        <StatTile
          label="Uptime"
          value={inst.uptimeS == null ? null : humanDuration(inst.uptimeS)}
          precise
          icon={Clock}
          footnote={
            inst.idleKillS != null && inst.idleForS != null
              ? `idle ${humanDuration(inst.idleForS)} of ${humanDuration(inst.idleKillS)} before self-destruct`
              : "idle self-destruct not armed"
          }
        />
        <StatTile
          label="Credit left"
          value={inst.credit}
          prefix="$"
          decimals={2}
          icon={CircleDollarSign}
        />
        <StatTile
          label="Runway"
          value={inst.runwayH}
          unit="hr"
          decimals={1}
          tone={runwayTone}
          icon={Cpu}
          footnote={inst.gpuName ?? undefined}
        />
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-center justify-between gap-4 p-5">
          <div className="min-w-0">
            <p className="text-sm font-medium">
              Instance <span className="font-mono text-muted-foreground">{inst.id}</span>
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Storage bills whether the instance is running or stopped. Destroying it is the only
              thing that stops the burn.
            </p>
            {error ? <p className="mt-2 text-sm text-rose-400">{error}</p> : null}
          </div>
          {confirming ? (
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setConfirming(false)} disabled={busy}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="sm"
                disabled={busy}
                onClick={async () => {
                  setBusy(true);
                  setError(null);
                  try {
                    await api.destroyInstance(inst.id!);
                    setConfirming(false);
                    onChanged();
                  } catch (e) {
                    setError((e as Error).message || "destroy failed");
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                {busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Trash2 aria-hidden />}
                Yes, destroy {inst.id}
              </Button>
            </div>
          ) : (
            <Button variant="outline" size="sm" onClick={() => setConfirming(true)}>
              <Trash2 aria-hidden />
              Destroy instance
            </Button>
          )}
        </CardContent>
      </Card>
    </section>
  );
}

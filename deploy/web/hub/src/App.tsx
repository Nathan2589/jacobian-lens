import {
  BookOpen,
  ExternalLink,
  GitBranch,
  LayoutDashboard,
  LogOut,
  Microscope,
  RefreshCw,
  ServerCrash,
} from "lucide-react";
import EmptyState from "@/components/hub/EmptyState";
import ExperimentStats from "@/components/hub/ExperimentStats";
import InstancePanel from "@/components/hub/InstancePanel";
import RunHistory from "@/components/hub/RunHistory";
import StatusPill from "@/components/hub/StatusPill";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { usePoll } from "@/lib/usePoll";

export default function App() {
  // 10s for state: it is a cheap proxy-local call plus a cached vast lookup, and
  // it never touches the GPU, so it cannot hold a rented instance alive.
  const { data: state, error, loading, refresh } = usePoll(api.state, 10_000);
  const { data: runsData, loading: runsLoading, refresh: refreshRuns } = usePoll(() => api.runs(50), 30_000);

  const refreshAll = () => {
    void refresh();
    void refreshRuns();
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-6">
          <div className="flex min-w-0 items-center gap-3">
            <Microscope className="size-4 shrink-0 text-primary" aria-hidden />
            <span className="truncate text-sm font-semibold tracking-tight">
              J-lens <span className="font-normal text-muted-foreground">· experiment hub</span>
            </span>
          </div>
          <nav className="flex items-center gap-1.5">
            {state?.gpu.dashboardUrl ? (
              <Button variant="ghost" size="sm" asChild>
                <a href={state.gpu.dashboardUrl}>
                  <LayoutDashboard aria-hidden />
                  <span className="hidden sm:inline">Open dashboard</span>
                  <ExternalLink className="size-3 opacity-60" aria-hidden />
                </a>
              </Button>
            ) : null}
            <Button variant="ghost" size="icon" onClick={refreshAll} aria-label="Refresh">
              <RefreshCw className={loading ? "animate-spin" : undefined} aria-hidden />
            </Button>
            {state?.user ? (
              <Button variant="ghost" size="sm" asChild>
                <a href="/_jlens/logout">
                  <LogOut aria-hidden />
                  <span className="hidden font-mono text-xs sm:inline">{state.user.login}</span>
                </a>
              </Button>
            ) : null}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        {error && !state ? (
          <Card>
            <CardContent className="p-0">
              <EmptyState
                icon={ServerCrash}
                title="The hub API is not answering"
                action={
                  <Button variant="outline" size="sm" onClick={refreshAll}>
                    <RefreshCw aria-hidden />
                    Try again
                  </Button>
                }
              >
                This page is served by the auth proxy on the always-on box, so this is a problem
                with the proxy itself rather than with the rented GPU.
              </EmptyState>
            </CardContent>
          </Card>
        ) : (
          <div className="flex flex-col gap-10">
            <section className="flex flex-col gap-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <h1 className="text-2xl font-semibold tracking-tight">Deployment</h1>
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    One model per branch. This box serves whichever branch it was deployed from.
                  </p>
                </div>
                {state ? <StatusPill status={state.gpu.status} detail={state.gpu.detail} /> : <Skeleton className="h-7 w-32" />}
              </div>

              {state ? (
                <Card>
                  <CardHeader className="pb-4">
                    <CardTitle className="flex flex-wrap items-center gap-2 text-base">
                      <span className="font-mono">{state.deployment.model ?? "model not reported"}</span>
                      {state.deployment.precision ? (
                        <Badge variant="secondary">{state.deployment.precision}</Badge>
                      ) : null}
                    </CardTitle>
                    <CardDescription className="flex flex-wrap items-center gap-x-4 gap-y-1">
                      {state.deployment.branch ? (
                        <span className="inline-flex items-center gap-1.5">
                          <GitBranch className="size-3.5" aria-hidden />
                          <span className="font-mono">{state.deployment.branch}</span>
                        </span>
                      ) : null}
                      {state.deployment.lensRepo ? (
                        <span className="font-mono">{state.deployment.lensRepo}</span>
                      ) : null}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    {state.gpu.reachable ? (
                      state.gpu.detail
                    ) : (
                      <>
                        No GPU is rented right now, which is the normal resting state — the card is
                        rented on demand and destroys itself when idle. Run{" "}
                        <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                          deploy/provision.sh
                        </code>{" "}
                        to rent one; bootstrap then takes 15–35 minutes.
                      </>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Skeleton className="h-40 w-full" />
              )}
            </section>

            <section className="flex flex-col gap-5">
              <div>
                <h2 className="text-xl font-semibold tracking-tight">Experiments</h2>
                <p className="mt-1.5 text-sm text-muted-foreground">
                  From the proxy's own log, so these hold whether or not a GPU is rented.
                </p>
              </div>
              <ExperimentStats stats={runsData?.stats ?? null} />
            </section>

            {state ? <InstancePanel state={state} onChanged={refreshAll} /> : null}

            <section className="flex flex-col gap-5">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight">Run history</h2>
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    Recorded by the proxy, so it survives the instance being destroyed.
                  </p>
                </div>
                <Button variant="ghost" size="sm" asChild>
                  <a href="https://github.com/Nathan2589/jacobian-lens/blob/main/deploy/README.md">
                    <BookOpen aria-hidden />
                    Deploy docs
                  </a>
                </Button>
              </div>
              <Card className="overflow-hidden">
                <CardContent className="p-0">
                  <RunHistory runs={runsData?.runs ?? null} loading={runsLoading} />
                </CardContent>
              </Card>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

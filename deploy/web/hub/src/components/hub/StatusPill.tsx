import ShinyText from "@/components/reactbits/ShinyText";
import StarBorder from "@/components/reactbits/StarBorder";
import { Badge } from "@/components/ui/badge";
import type { GpuStatus } from "@/lib/api";

/** The one element on the page allowed to move on its own, because it is the one
 *  element that means money is being spent. When no instance is rented this is a
 *  plain static badge - a dashboard that shimmers at rest teaches you to ignore
 *  the shimmer. */
export default function StatusPill({ status, detail }: { status: GpuStatus; detail: string }) {
  if (status === "absent") {
    return (
      <Badge variant="outline" className="gap-1.5 py-1">
        <span className="size-1.5 rounded-full bg-muted-foreground" aria-hidden />
        No GPU rented
      </Badge>
    );
  }
  if (status === "error") {
    return (
      <Badge variant="danger" className="py-1">
        <span className="size-1.5 rounded-full bg-rose-400" aria-hidden />
        {detail || "Model failed to load"}
      </Badge>
    );
  }
  if (status === "loading") {
    return (
      <Badge variant="warning" className="py-1">
        <span className="size-1.5 animate-pulse rounded-full bg-amber-400" aria-hidden />
        {detail || "Loading model"}
      </Badge>
    );
  }
  return (
    <StarBorder speed="5s">
      <span className="flex items-center gap-2 text-xs font-medium">
        <span className="size-1.5 rounded-full bg-emerald-400" aria-hidden />
        <ShinyText text="Live · billing" shineColor="hsl(var(--primary))" speed={4} />
      </span>
    </StarBorder>
  );
}

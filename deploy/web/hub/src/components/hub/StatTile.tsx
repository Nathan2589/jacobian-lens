import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import CountUp from "@/components/reactbits/CountUp";
import SpotlightCard from "@/components/reactbits/SpotlightCard";
import { cn } from "@/lib/utils";

/** Label, one big number, one line of context. The number animates in with
 *  react-bits CountUp; `precise` turns that off for values where a spring
 *  sweeping through wrong intermediate numbers would be misread (an instance id,
 *  a layer count). */
export default function StatTile({
  label,
  value,
  unit,
  icon: Icon,
  footnote,
  decimals,
  prefix,
  precise = false,
  tone = "default",
  className,
}: {
  label: string;
  value: number | string | null;
  unit?: string;
  icon?: LucideIcon;
  footnote?: ReactNode;
  decimals?: number;
  prefix?: string;
  precise?: boolean;
  tone?: "default" | "success" | "warning" | "danger";
  className?: string;
}) {
  const toneRing = {
    default: "",
    success: "ring-1 ring-emerald-900/50",
    warning: "ring-1 ring-amber-900/50",
    danger: "ring-1 ring-rose-900/50",
  }[tone];

  const numeric = typeof value === "number" && Number.isFinite(value);

  return (
    <SpotlightCard className={cn("p-5", toneRing, className)}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
        {Icon ? <Icon className="size-4 shrink-0 text-muted-foreground" aria-hidden /> : null}
      </div>
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-3xl font-semibold tabular-nums tracking-tight">
          {value === null ? (
            <span className="text-muted-foreground">--</span>
          ) : numeric && !precise ? (
            <CountUp to={value as number} decimals={decimals} prefix={prefix} separator="," />
          ) : (
            <>
              {prefix}
              {value}
            </>
          )}
        </span>
        {unit ? <span className="text-sm text-muted-foreground">{unit}</span> : null}
      </div>
      {footnote ? <p className="mt-2 text-sm text-muted-foreground">{footnote}</p> : null}
    </SpotlightCard>
  );
}

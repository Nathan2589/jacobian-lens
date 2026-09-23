import type { ElementType, ReactNode } from "react";
import { cn } from "@/lib/utils";

/** react-bits StarBorder. A light travels the top and bottom edge.
 *  Retokenised and shrunk: upstream is a 20px-radius 16x26 button; here it wraps
 *  the "instance running" pill, which is the one thing on the page that should
 *  pull the eye because it is the thing that costs money. */
export default function StarBorder({
  as: Component = "div" as ElementType,
  className = "",
  color = "hsl(var(--primary))",
  speed = "5s",
  thickness = 1,
  children,
}: {
  as?: ElementType;
  className?: string;
  color?: string;
  speed?: string;
  thickness?: number;
  children: ReactNode;
}) {
  return (
    <Component
      className={cn("relative inline-block overflow-hidden rounded-md", className)}
      style={{ padding: `${thickness}px 0` }}
    >
      <div
        className="absolute -bottom-3 -right-[250%] z-0 h-1/2 w-[300%] rounded-[50%] opacity-70 motion-safe:animate-[star-bottom_linear_infinite_alternate] motion-reduce:hidden"
        style={{ background: `radial-gradient(circle, ${color}, transparent 10%)`, animationDuration: speed }}
      />
      <div
        className="absolute -left-[250%] -top-3 z-0 h-1/2 w-[300%] rounded-[50%] opacity-70 motion-safe:animate-[star-top_linear_infinite_alternate] motion-reduce:hidden"
        style={{ background: `radial-gradient(circle, ${color}, transparent 10%)`, animationDuration: speed }}
      />
      <div className="relative z-10 rounded-md border bg-card px-3 py-1.5 text-center">{children}</div>
    </Component>
  );
}

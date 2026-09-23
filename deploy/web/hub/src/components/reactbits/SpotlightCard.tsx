import { useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** react-bits SpotlightCard. A radial highlight tracks the cursor across the card.
 *  Retokenised: the upstream version hard-codes #111/#222 and a 1.5rem radius. */
export default function SpotlightCard({
  children,
  className = "",
  spotlightColor = "hsl(var(--primary) / 0.14)",
}: {
  children: ReactNode;
  className?: string;
  spotlightColor?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  return (
    <div
      ref={ref}
      onMouseMove={(e) => {
        const el = ref.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        el.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
        el.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
      }}
      style={{ "--spotlight-color": spotlightColor } as React.CSSProperties}
      className={cn(
        "group relative overflow-hidden rounded-lg border bg-card text-card-foreground shadow-sm",
        "before:pointer-events-none before:absolute before:inset-0 before:opacity-0",
        "before:transition-opacity before:duration-500 before:content-['']",
        "before:bg-[radial-gradient(circle_at_var(--mouse-x,50%)_var(--mouse-y,50%),var(--spotlight-color),transparent_70%)]",
        "hover:before:opacity-100 focus-within:before:opacity-100",
        className,
      )}
    >
      {children}
    </div>
  );
}

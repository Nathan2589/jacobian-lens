import { motion, useAnimationFrame, useMotionValue, useTransform } from "motion/react";
import { useRef } from "react";
import { cn } from "@/lib/utils";

/** react-bits ShinyText, typed and simplified to the one mode this hub uses:
 *  a single sweep on a loop. Reserved for the "GPU is live and billing" label,
 *  so motion means money is moving - not decoration. */
export default function ShinyText({
  text,
  disabled = false,
  speed = 3,
  className = "",
  color = "hsl(var(--muted-foreground))",
  shineColor = "hsl(var(--foreground))",
  spread = 120,
}: {
  text: string;
  disabled?: boolean;
  speed?: number;
  className?: string;
  color?: string;
  shineColor?: string;
  spread?: number;
}) {
  const progress = useMotionValue(0);
  const elapsed = useRef(0);
  const last = useRef<number | null>(null);
  const durationMs = speed * 1000;

  useAnimationFrame((time) => {
    if (disabled) {
      last.current = null;
      return;
    }
    if (last.current === null) {
      last.current = time;
      return;
    }
    elapsed.current += time - last.current;
    last.current = time;
    progress.set(((elapsed.current % durationMs) / durationMs) * 100);
  });

  const backgroundPosition = useTransform(progress, (p) => `${150 - p * 2}% center`);

  if (disabled) return <span className={className}>{text}</span>;

  return (
    <motion.span
      className={cn("inline-block", className)}
      style={{
        backgroundImage: `linear-gradient(${spread}deg, ${color} 0%, ${color} 35%, ${shineColor} 50%, ${color} 65%, ${color} 100%)`,
        backgroundSize: "200% auto",
        WebkitBackgroundClip: "text",
        backgroundClip: "text",
        WebkitTextFillColor: "transparent",
        backgroundPosition,
      }}
    >
      {text}
    </motion.span>
  );
}

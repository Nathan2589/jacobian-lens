import { useInView, useMotionValue, useSpring } from "motion/react";
import { useCallback, useEffect, useRef } from "react";

/** react-bits CountUp, typed. Spring-animates a number into place.
 *  Used for the stat tiles - a cost or an uptime that lands rather than snaps
 *  reads as live data instead of a static render. */
export default function CountUp({
  to,
  from = 0,
  duration = 1.4,
  delay = 0,
  className = "",
  separator = "",
  decimals,
  prefix = "",
  suffix = "",
}: {
  to: number;
  from?: number;
  duration?: number;
  delay?: number;
  className?: string;
  separator?: string;
  decimals?: number;
  prefix?: string;
  suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const motionValue = useMotionValue(from);
  const springValue = useSpring(motionValue, {
    damping: 20 + 40 * (1 / duration),
    stiffness: 100 * (1 / duration),
  });
  const isInView = useInView(ref, { once: true, margin: "0px" });

  const decimalPlaces = (n: number) => {
    const [, frac] = n.toString().split(".");
    return frac && parseInt(frac) !== 0 ? frac.length : 0;
  };
  const places = decimals ?? Math.max(decimalPlaces(from), decimalPlaces(to));

  const format = useCallback(
    (latest: number) => {
      const out = Intl.NumberFormat("en-US", {
        useGrouping: !!separator,
        minimumFractionDigits: places,
        maximumFractionDigits: places,
      }).format(latest);
      return prefix + (separator ? out.replace(/,/g, separator) : out) + suffix;
    },
    [places, separator, prefix, suffix],
  );

  useEffect(() => {
    if (ref.current) ref.current.textContent = format(from);
  }, [from, format]);

  useEffect(() => {
    if (!isInView) return;
    const id = setTimeout(() => motionValue.set(to), delay * 1000);
    return () => clearTimeout(id);
  }, [isInView, motionValue, to, delay]);

  useEffect(
    () => springValue.on("change", (v) => {
      if (ref.current) ref.current.textContent = format(v);
    }),
    [springValue, format],
  );

  return <span className={className} ref={ref} />;
}

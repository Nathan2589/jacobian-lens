import { motion, useInView } from "motion/react";
import { useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

function AnimatedItem({ children, delay, index }: { children: ReactNode; delay: number; index: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.4, once: true });
  return (
    <motion.div
      ref={ref}
      data-index={index}
      initial={{ opacity: 0, y: 8 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 }}
      transition={{ duration: 0.22, delay }}
    >
      {children}
    </motion.div>
  );
}

/** react-bits AnimatedList, reduced to the staggered-reveal half.
 *  Upstream also ships keyboard selection and a scroll-gradient overlay; the run
 *  history here is a table of links, not a picker, so that half is dropped rather
 *  than carried unused. Stagger is capped so a 200-run history does not spend
 *  four seconds animating. */
export default function AnimatedList<T>({
  items,
  render,
  className,
  stagger = 0.03,
  maxStaggerItems = 12,
}: {
  items: T[];
  render: (item: T, index: number) => ReactNode;
  className?: string;
  stagger?: number;
  maxStaggerItems?: number;
}) {
  return (
    <div className={cn("flex flex-col", className)}>
      {items.map((item, i) => (
        <AnimatedItem key={i} index={i} delay={Math.min(i, maxStaggerItems) * stagger}>
          {render(item, i)}
        </AnimatedItem>
      ))}
    </div>
  );
}

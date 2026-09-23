import type * as React from "react";
import { cn } from "@/lib/utils";

/** Sized to the real content box, never a generic spinner: the shape of this
 *  page is known before the data arrives, so the page should not reflow. */
export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("animate-pulse rounded-md bg-muted/50", className)} {...props} />;
}

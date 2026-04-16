export function clampK(value: number) {
  return Math.max(1, Math.min(14, Math.floor(value)));
}

export function formatShare(value: number, total: number) {
  if (total <= 0) {
    return String(value);
  }
  const pct = Math.round((value / total) * 100);
  return `${value}/${total} (${pct}%)`;
}

export function getNetDirection(upDays: number, downDays: number) {
  if (upDays === downDays) {
    return { label: "Mixed", className: "text-slate-700" };
  }
  if (upDays > downDays) {
    return { label: `Mostly Up (+${upDays - downDays})`, className: "text-emerald-700" };
  }
  return { label: `Mostly Down (${upDays - downDays})`, className: "text-rose-700" };
}

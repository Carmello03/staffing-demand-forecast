export function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white/70 px-4 py-3 text-sm text-slate-600">
      <span className="h-3 w-3 animate-pulse rounded-full bg-brand-500" />
      <span>{label}</span>
    </div>
  );
}

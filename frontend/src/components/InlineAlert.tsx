type InlineAlertProps = {
  message: string;
  kind?: "error" | "success" | "info";
};

const styleByKind: Record<NonNullable<InlineAlertProps["kind"]>, string> = {
  error: "border-rose-200 bg-rose-50 text-rose-800",
  success: "border-emerald-200 bg-emerald-50 text-emerald-800",
  info: "border-sky-200 bg-sky-50 text-sky-800",
};

export function InlineAlert({ message, kind = "info" }: InlineAlertProps) {
  return (
    <div className={`rounded-xl border px-4 py-3 text-sm font-medium shadow-sm ${styleByKind[kind]}`}>
      {message}
    </div>
  );
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

export function parseIsoDateLocal(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
  if (!match) {
    return null;
  }

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const parsed = new Date(year, month - 1, day);

  if (
    parsed.getFullYear() !== year
    || parsed.getMonth() !== month - 1
    || parsed.getDate() !== day
  ) {
    return null;
  }

  return parsed;
}

export function formatIsoDateLocal(date: Date): string {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

export function addDaysIsoLocal(baseIsoDate: string, days: number): string | null {
  const base = parseIsoDateLocal(baseIsoDate);
  if (!base) {
    return null;
  }
  const shifted = new Date(base);
  shifted.setDate(base.getDate() + days);
  return formatIsoDateLocal(shifted);
}

export function weekdayFromIsoLocal(dateIso: string): string {
  const parsed = parseIsoDateLocal(dateIso);
  if (!parsed) {
    return "";
  }
  return parsed.toLocaleDateString(undefined, { weekday: "long" });
}

export function todayIsoLocal(): string {
  return formatIsoDateLocal(new Date());
}

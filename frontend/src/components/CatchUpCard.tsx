import { useMemo, useState, type ChangeEvent } from "react";
import { downloadCatchUpTemplate, postCatchUp, uploadStoreCsv } from "../lib/api";
import type { CatchUpDayIn, StoreStatus } from "../types";
import { formatIsoDateLocal, parseIsoDateLocal, weekdayFromIsoLocal } from "../utils/date";
import { InlineAlert } from "./InlineAlert";

type Props = {
  storeId: string;
  status: StoreStatus;
  onComplete: () => void;
};

type DayRow = {
  date: string;
  weekday: string;
  customers: string;
  open: 0 | 1;
  promo: 0 | 1;
};

function toWeekday(dateStr: string) {
  return weekdayFromIsoLocal(dateStr);
}

function buildMissingDates(lastUploaded: string, currentDate: string): string[] {
  const startDate = parseIsoDateLocal(lastUploaded);
  const endDate = parseIsoDateLocal(currentDate);
  if (!startDate || !endDate) {
    return [];
  }

  const dates: string[] = [];
  const end = new Date(endDate);
  const cursor = new Date(startDate);
  cursor.setDate(cursor.getDate() + 1);
  while (cursor <= end) {
    dates.push(formatIsoDateLocal(cursor));
    cursor.setDate(cursor.getDate() + 1);
  }
  return dates;
}
const QUICK_FILL_THRESHOLD = 5;

export function CatchUpCard({ storeId, status, onComplete }: Props) {
  const missingDates = useMemo(() => {
    if (!status.last_uploaded_date || !status.current_date) return [];
    return buildMissingDates(status.last_uploaded_date, status.current_date);
  }, [status.last_uploaded_date, status.current_date]);

  const defaultMode = missingDates.length <= QUICK_FILL_THRESHOLD ? "quick" : "csv";
  const [mode, setMode] = useState<"quick" | "csv">(defaultMode);
  const [rows, setRows] = useState<DayRow[]>(() =>
    missingDates.map((date) => ({
      date,
      weekday: toWeekday(date),
      customers: "",
      open: 1,
      promo: 0,
    })),
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvSuccess, setCsvSuccess] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  if (missingDates.length === 0) return null;

  function updateRow(index: number, field: keyof DayRow, value: string | number) {
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, [field]: value } : r)));
  }

  function setAllOpen(open: 0 | 1) {
    setRows((prev) => prev.map((r) => ({ ...r, open })));
  }

  function setAllPromo(promo: 0 | 1) {
    setRows((prev) => prev.map((r) => ({ ...r, promo })));
  }

  async function handleQuickSubmit() {
    setError(null);
    setSuccess(null);

    const payload: CatchUpDayIn[] = rows.map((r) => ({
      date: r.date,
      customers: r.open === 0 ? 0 : parseFloat(r.customers) || 0,
      open: r.open,
      promo: r.promo,
    }));
    const missingCount = payload.some((p) => p.open === 1 && !p.customers && p.customers !== 0);
    if (missingCount) {
      setError("Please enter customer counts for all open days (or mark them as closed).");
      return;
    }

    setSubmitting(true);
    try {
      const result = await postCatchUp(storeId, payload);
      setSuccess(`${result.rows_written} day${result.rows_written === 1 ? "" : "s"} saved successfully.`);
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save days.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCsvUpload() {
    if (!file) return;
    setCsvError(null);
    setCsvSuccess(null);
    setUploading(true);
    try {
      const result = await uploadStoreCsv(storeId, file);
      setCsvSuccess(`${result.rows_written ?? "?"} rows uploaded successfully.`);
      setFile(null);
      onComplete();
    } catch (err) {
      setCsvError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleDownloadTemplate() {
    setDownloading(true);
    try {
      await downloadCatchUpTemplate(storeId);
    } catch (err) {
      setCsvError(err instanceof Error ? err.message : "Could not download template.");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 space-y-3">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-semibold text-amber-900">
            Warning: {missingDates.length} missing day{missingDates.length === 1 ? "" : "s"} - catch up before forecasting
          </p>
          <p className="text-xs text-amber-700 mt-0.5">
            Last upload: <span className="font-medium">{status.last_uploaded_date}</span>
            {" | "}
            Current date: <span className="font-medium">{status.current_date}</span>
          </p>
        </div>
        <div className="flex rounded-lg border border-amber-200 overflow-hidden text-xs font-semibold self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setMode("quick")}
            className={`px-3 py-1.5 transition ${mode === "quick" ? "bg-amber-600 text-white" : "bg-white text-amber-700 hover:bg-amber-100"}`}
          >
            Quick Fill
          </button>
          <button
            type="button"
            onClick={() => setMode("csv")}
            className={`px-3 py-1.5 transition ${mode === "csv" ? "bg-amber-600 text-white" : "bg-white text-amber-700 hover:bg-amber-100"}`}
          >
            Upload CSV
          </button>
        </div>
      </div>

      {missingDates.length > QUICK_FILL_THRESHOLD && mode === "quick" && (
        <p className="text-xs text-amber-700 bg-amber-100 rounded-lg px-3 py-2">
          You have {missingDates.length} missing days - the CSV upload tab may be quicker.
        </p>
      )}
      {mode === "quick" && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <span className="text-xs font-medium text-amber-800 self-center">Bulk:</span>
            <button type="button" onClick={() => setAllOpen(1)}
              className="rounded-lg border border-amber-300 bg-white px-2 py-1 text-xs text-amber-800 hover:bg-amber-100">
              All open
            </button>
            <button type="button" onClick={() => setAllOpen(0)}
              className="rounded-lg border border-amber-300 bg-white px-2 py-1 text-xs text-amber-800 hover:bg-amber-100">
              All closed
            </button>
            <button type="button" onClick={() => setAllPromo(1)}
              className="rounded-lg border border-amber-300 bg-white px-2 py-1 text-xs text-amber-800 hover:bg-amber-100">
              All promo
            </button>
            <button type="button" onClick={() => setAllPromo(0)}
              className="rounded-lg border border-amber-300 bg-white px-2 py-1 text-xs text-amber-800 hover:bg-amber-100">
              Clear promo
            </button>
          </div>

          <div className="overflow-x-auto rounded-xl border border-amber-200 bg-white/80">
            <table className="min-w-full text-sm">
              <thead className="bg-amber-50 text-left text-xs uppercase tracking-wide text-amber-700">
                <tr>
                  <th className="px-4 py-2">Date</th>
                  <th className="px-4 py-2">Weekday</th>
                  <th className="px-4 py-2">Open</th>
                  <th className="px-4 py-2">Promo</th>
                  <th className="px-4 py-2">Customers</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-amber-100">
                {rows.map((row, i) => (
                  <tr key={row.date} className={row.open === 0 ? "bg-slate-50/80 opacity-60" : ""}>
                    <td className="px-4 py-2 font-mono text-xs text-slate-700">{row.date}</td>
                    <td className="px-4 py-2 text-slate-700">{row.weekday}</td>
                    <td className="px-4 py-2">
                      <input
                        type="checkbox"
                        checked={row.open === 1}
                        onChange={(e) => updateRow(i, "open", e.target.checked ? 1 : 0)}
                        className="h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-300"
                      />
                    </td>
                    <td className="px-4 py-2">
                      <input
                        type="checkbox"
                        checked={row.promo === 1}
                        disabled={row.open === 0}
                        onChange={(e) => updateRow(i, "promo", e.target.checked ? 1 : 0)}
                        className="h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-300 disabled:opacity-40"
                      />
                    </td>
                    <td className="px-4 py-2">
                      {row.open === 0 ? (
                        <span className="text-xs text-slate-400">Closed</span>
                      ) : (
                        <input
                          type="number"
                          min={0}
                          value={row.customers}
                          onChange={(e) => updateRow(i, "customers", e.target.value)}
                          placeholder="e.g. 380"
                          className="w-24 rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-200"
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {error && <InlineAlert kind="error" message={error} />}
          {success && <InlineAlert kind="success" message={success} />}

          <button
            type="button"
            onClick={handleQuickSubmit}
            disabled={submitting}
            className="rounded-xl bg-amber-600 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-amber-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {submitting ? "Saving..." : `Save ${missingDates.length} day${missingDates.length === 1 ? "" : "s"}`}
          </button>
        </div>
      )}
      {mode === "csv" && (
        <div className="space-y-3">
          <p className="text-sm text-amber-800">
            Upload a CSV with columns: <span className="font-mono font-semibold">Date, Customers, Open, Promo</span>.
            <br />
            The file should cover the {missingDates.length} missing day{missingDates.length === 1 ? "" : "s"} from{" "}
            <span className="font-medium">{missingDates[0]}</span> to{" "}
            <span className="font-medium">{missingDates[missingDates.length - 1]}</span>.
          </p>
          <button
            type="button"
            onClick={handleDownloadTemplate}
            disabled={downloading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-800 transition hover:bg-amber-100 disabled:opacity-70"
          >
            {downloading ? "Preparing..." : "Download pre-filled template"}
          </button>

          <p className="text-xs text-amber-700">
            The template has all missing dates pre-filled - just add your customer counts, save, and upload.
          </p>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <input
              type="file"
              accept=".csv"
              onChange={(e: ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] ?? null)}
              className="text-sm text-slate-700 file:mr-3 file:rounded-lg file:border file:border-amber-300 file:bg-white file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-amber-800 hover:file:bg-amber-100"
            />
            <button
              type="button"
              onClick={handleCsvUpload}
              disabled={!file || uploading}
              className="rounded-xl bg-amber-600 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-amber-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </div>

          {csvError && <InlineAlert kind="error" message={csvError} />}
          {csvSuccess && <InlineAlert kind="success" message={csvSuccess} />}
        </div>
      )}
    </div>
  );
}

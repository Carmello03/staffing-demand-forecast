import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "../components/Card";
import { CatchUpCard } from "../components/CatchUpCard";
import { ForecastTrendChart } from "../components/ForecastTrendChart";
import { InlineAlert } from "../components/InlineAlert";
import { LoadingState } from "../components/LoadingState";
import { StoreSelector } from "../components/StoreSelector";
import { useStores } from "../contexts/StoresContext";
import { getForecastRange, getStoreStatus, postForecastRange } from "../lib/api";
import type { ForecastDayOverride, StoreStatus } from "../types";
import {
  buildGlobalTopDrivers,
  buildNextKRows,
  describeDriverLabel,
  formatCustomers,
  formatDate,
  getExplanationLabels,
  mapForecastsByHorizon,
  type ForecastMap,
} from "../utils/forecast";
import {
  readForecastRangeSession,
  writeForecastRangeSession,
} from "../utils/forecastSessionCache";
import { readSelectedStoreId, resolveStoreId, saveSelectedStoreId } from "../utils/storeSelection";
import { clampK, formatShare, getNetDirection } from "../utils/ui";

type ForecastStats = {
  availableCount: number;
  average: number | null;
  peakDay: number | null;
  peakValue: number | null;
};

type DayPlan = {
  open: boolean;
  promo: boolean;
};

function defaultDayPlan(): DayPlan {
  return { open: true, promo: false };
}

function exportForecastCSV(rows: ReturnType<typeof buildNextKRows>, storeName: string) {
  const headers = ["Day", "Date", "Weekday", "Open", "Promo", "Holiday", "Predicted Customers", "Suggested Staff"];
  const data = rows.map((row) => [
    `+${row.day}`,
    row.targetDate ?? "",
    row.weekday ?? "",
    row.plannedOpen === 1 ? "Yes" : "No",
    row.plannedPromo === 1 ? "Yes" : "No",
    row.holidayNote ?? "",
    row.prediction !== null ? Math.round(row.prediction) : "",
    typeof row.suggestedStaff === "number" ? row.suggestedStaff : "",
  ]);
  const csv = [headers, ...data]
    .map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `forecast-${storeName.replace(/\s+/g, "-")}-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function getRowClass(weekday: string, stateHoliday: string, schoolHoliday: number) {
  const isHoliday = stateHoliday !== "0" || schoolHoliday === 1;
  const isWeekend = weekday === "Saturday" || weekday === "Sunday";
  if (isHoliday) return "bg-amber-50/60 hover:bg-amber-50";
  if (isWeekend) return "bg-slate-50/80 hover:bg-slate-100/70";
  return "transition hover:bg-sky-50/70";
}

function buildScenarioKey(customEnabled: boolean, overrides: ForecastDayOverride[]) {
  if (!customEnabled) {
    return "default";
  }
  return overrides
    .map((item) => `${item.horizon}:${item.open ?? 1}:${item.promo ?? 0}`)
    .join("|");
}

export function DashboardPage() {
  const { stores, loading: loadingStores, error: storeError } = useStores();
  const [selectedStoreId, setSelectedStoreId] = useState("");
  const [status, setStatus] = useState<StoreStatus | null>(null);
  const [forecasts, setForecasts] = useState<ForecastMap>({});
  const [k, setK] = useState(7);

  const [loadingStatus, setLoadingStatus] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [loadingForecasts, setLoadingForecasts] = useState(false);
  const [loadingExplanations, setLoadingExplanations] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [hasLoadedForecast, setHasLoadedForecast] = useState(false);
  const [showTechnicalDrivers, setShowTechnicalDrivers] = useState(false);
  const [useCustomScenario, setUseCustomScenario] = useState(false);
  const [showScenarioEditor, setShowScenarioEditor] = useState(false);
  const [dayPlans, setDayPlans] = useState<Record<number, DayPlan>>(() =>
    Object.fromEntries(Array.from({ length: 14 }, (_, i) => [i + 1, defaultDayPlan()])),
  );

  const scenarioOverrides = useMemo<ForecastDayOverride[]>(
    () =>
      useCustomScenario
        ? Array.from({ length: k }, (_, i) => i + 1).map((day) => ({
            horizon: day,
            open: (dayPlans[day]?.open ?? true) ? 1 : 0,
            promo: (dayPlans[day]?.promo ?? false) ? 1 : 0,
          }))
        : [],
    [dayPlans, k, useCustomScenario],
  );

  const scenarioKey = useMemo(
    () => buildScenarioKey(useCustomScenario, scenarioOverrides),
    [scenarioOverrides, useCustomScenario],
  );

  const planningRows = useMemo(() => {
    const source = status?.current_date ? new Date(`${status.current_date}T00:00:00`) : new Date();
    const base = Number.isNaN(source.getTime()) ? new Date() : source;
    return Array.from({ length: k }, (_, index) => {
      const day = index + 1;
      const target = new Date(base);
      target.setDate(base.getDate() + day);
      const targetDate = target.toISOString().slice(0, 10);
      const weekday = target.toLocaleDateString(undefined, { weekday: "long" });
      return { day, targetDate, weekday };
    });
  }, [k, status?.current_date]);

  const scenarioSummary = useMemo(() => {
    if (!useCustomScenario) {
      return null;
    }
    const closedDays = scenarioOverrides.filter((item) => item.open === 0).length;
    const promoDays = scenarioOverrides.filter((item) => item.promo === 1).length;
    return { closedDays, promoDays };
  }, [scenarioOverrides, useCustomScenario]);

  useEffect(() => {
    if (stores.length === 0) {
      setSelectedStoreId("");
      return;
    }

    setSelectedStoreId((current) => resolveStoreId(stores, current, readSelectedStoreId()));
  }, [stores]);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      saveSelectedStoreId(selectedStoreId);

      if (!selectedStoreId) {
        setStatus(null);
        setForecasts({});
        setStatusError(null);
        setForecastError(null);
        setHasLoadedForecast(false);
        setLoadingExplanations(false);
        return;
      }

      setStatus(null);
      setForecasts({});
      setStatusError(null);
      setForecastError(null);
      setHasLoadedForecast(false);
      setLoadingExplanations(false);
      setLoadingStatus(true);

      const statusResult = await getStoreStatus(selectedStoreId)
        .then((value) => ({ ok: true as const, value }))
        .catch((error: unknown) => ({ ok: false as const, error }));

      if (cancelled) {
        return;
      }

      if (statusResult.ok) {
        setStatus(statusResult.value);
      } else {
        setStatus(null);
        setStatusError(
          statusResult.error instanceof Error
            ? statusResult.error.message
            : "Unable to load store status.",
        );
      }
      setLoadingStatus(false);
    }

    void loadStatus();

    return () => {
      cancelled = true;
    };
  }, [selectedStoreId]);

  useEffect(() => {
    if (!selectedStoreId) {
      return;
    }
    if (!status?.current_date) {
      return;
    }

    const cached = readForecastRangeSession(selectedStoreId, k, status.current_date, scenarioKey);
    if (!cached) {
      return;
    }

    setForecasts(mapForecastsByHorizon(cached.forecasts));
    setForecastError(null);
    setHasLoadedForecast(true);
  }, [k, scenarioKey, selectedStoreId, status?.current_date]);

  function updateDayPlan(day: number, patch: Partial<DayPlan>) {
    setDayPlans((current) => ({
      ...current,
      [day]: { ...(current[day] || defaultDayPlan()), ...patch },
    }));
    setForecasts({});
    setForecastError(null);
    setHasLoadedForecast(false);
    setLoadingExplanations(false);
  }

  function setAllOpen(value: boolean) {
    setDayPlans((current) => {
      const next = { ...current };
      for (let day = 1; day <= k; day += 1) {
        next[day] = { ...(next[day] || defaultDayPlan()), open: value };
      }
      return next;
    });
    setForecasts({});
    setForecastError(null);
    setHasLoadedForecast(false);
    setLoadingExplanations(false);
  }

  function setAllPromo(value: boolean) {
    setDayPlans((current) => {
      const next = { ...current };
      for (let day = 1; day <= k; day += 1) {
        next[day] = { ...(next[day] || defaultDayPlan()), promo: value };
      }
      return next;
    });
    setForecasts({});
    setForecastError(null);
    setHasLoadedForecast(false);
    setLoadingExplanations(false);
  }

  async function handleGenerateForecast() {
    if (!selectedStoreId) {
      return;
    }

    setLoadingForecasts(true);
    setLoadingExplanations(false);
    setForecastError(null);
    setHasLoadedForecast(true);

    const baseForecastResult = await (
      useCustomScenario
        ? postForecastRange(selectedStoreId, {
            k,
            day_overrides: scenarioOverrides,
            include_explanations: false,
          })
        : getForecastRange(selectedStoreId, k, false)
    )
      .then((value) => ({ ok: true as const, value }))
      .catch((error: unknown) => ({ ok: false as const, error }));

    if (!baseForecastResult.ok) {
      setForecasts({});
      setForecastError(
        baseForecastResult.error instanceof Error
          ? baseForecastResult.error.message
          : "Unable to load forecast.",
      );
      setHasLoadedForecast(false);
      setLoadingForecasts(false);
      return;
    }

    setForecasts(mapForecastsByHorizon(baseForecastResult.value.forecasts));
    writeForecastRangeSession(selectedStoreId, k, baseForecastResult.value, scenarioKey);
    setLoadingForecasts(false);

    setLoadingExplanations(true);
    const explanationResult = await (
      useCustomScenario
        ? postForecastRange(selectedStoreId, {
            k,
            day_overrides: scenarioOverrides,
            include_explanations: true,
          })
        : getForecastRange(selectedStoreId, k, true)
    )
      .then((value) => ({ ok: true as const, value }))
      .catch(() => ({ ok: false as const }));

    if (explanationResult.ok) {
      setForecasts(mapForecastsByHorizon(explanationResult.value.forecasts));
      writeForecastRangeSession(selectedStoreId, k, explanationResult.value, scenarioKey);
    }
    setLoadingExplanations(false);
  }

  const rows = useMemo(() => buildNextKRows(k, forecasts), [forecasts, k]);
  const forecastStats = useMemo<ForecastStats>(() => {
    const available = rows.filter(
      (row): row is (typeof rows)[number] & { prediction: number } => row.prediction !== null,
    );
    const total = available.reduce((sum, row) => sum + row.prediction, 0);
    const average = available.length > 0 ? total / available.length : null;

    let peakDay: number | null = null;
    let peakValue: number | null = null;

    available.forEach((row) => {
      if (peakValue === null || row.prediction > peakValue) {
        peakValue = row.prediction;
        peakDay = row.day;
      }
    });

    return {
      availableCount: available.length,
      average,
      peakDay,
      peakValue,
    };
  }, [rows]);
  const globalTopDrivers = useMemo(
    () =>
      buildGlobalTopDrivers(rows.map((row) => row.explanation), 10, {
        includeTechnical: showTechnicalDrivers,
      }),
    [rows, showTechnicalDrivers],
  );
  const hasExplanationData = useMemo(
    () => rows.some((row) => Boolean(row.explanation)),
    [rows],
  );
  const explanationDayCount = useMemo(
    () => rows.filter((row) => Boolean(row.explanation)).length,
    [rows],
  );

  return (
    <div className="space-y-5">
      {storeError && <InlineAlert kind="error" message={storeError} />}
      {statusError && <InlineAlert kind="error" message={statusError} />}
      {forecastError && <InlineAlert kind="error" message={forecastError} />}

      <div className="grid gap-5 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Overview">
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1">
              <span className="text-sm font-medium text-slate-700">Select store</span>
              <StoreSelector
                stores={stores}
                value={selectedStoreId}
                onChange={setSelectedStoreId}
                disabled={loadingStores}
              />
            </label>

            <label className="space-y-1">
              <span className="text-sm font-medium text-slate-700">Preview window (K days)</span>
              <select
                value={k}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setK(clampK(next));
                  setForecasts({});
                  setForecastError(null);
                  setHasLoadedForecast(false);
                  setLoadingExplanations(false);
                }}
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm text-slate-800 shadow-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              >
                {Array.from({ length: 14 }, (_, i) => i + 1).map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <div className="sm:col-span-2 rounded-xl border border-slate-200 bg-white/75 p-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <label className="inline-flex items-center gap-2 text-sm font-medium text-slate-700">
                  <input
                    type="checkbox"
                    checked={useCustomScenario}
                    onChange={(event) => {
                      setUseCustomScenario(event.target.checked);
                      if (!event.target.checked) {
                        setShowScenarioEditor(false);
                      }
                      setForecasts({});
                      setForecastError(null);
                      setHasLoadedForecast(false);
                      setLoadingExplanations(false);
                    }}
                    className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-300"
                  />
                  Custom scenario (Open/Promo by day)
                </label>

                {useCustomScenario && (
                  <button
                    type="button"
                    onClick={() => setShowScenarioEditor((value) => !value)}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                  >
                    {showScenarioEditor ? "Hide Day Planner" : "Edit Day Planner"}
                  </button>
                )}
              </div>

              {useCustomScenario && (
                <p className="mt-2 text-xs text-slate-600">
                  Set if the store is open and if promo is active for each forecast day.
                </p>
              )}

              {useCustomScenario && showScenarioEditor && (
                <div className="mt-3 space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => setAllOpen(true)}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                    >
                      Set all open
                    </button>
                    <button
                      type="button"
                      onClick={() => setAllOpen(false)}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                    >
                      Set all closed
                    </button>
                    <button
                      type="button"
                      onClick={() => setAllPromo(true)}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                    >
                      Set all promo
                    </button>
                    <button
                      type="button"
                      onClick={() => setAllPromo(false)}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                    >
                      Clear promo
                    </button>
                  </div>

                  <div className="table-shell">
                    <table className="min-w-full text-sm">
                      <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="px-3 py-2">Day</th>
                          <th className="px-3 py-2">Date</th>
                          <th className="px-3 py-2">Weekday</th>
                          <th className="px-3 py-2">Open</th>
                          <th className="px-3 py-2">Promo</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white/85">
                        {planningRows.map((row) => {
                          const plan = dayPlans[row.day] || defaultDayPlan();
                          return (
                            <tr key={row.day} className="transition hover:bg-sky-50/70">
                              <td className="px-3 py-2 font-medium text-slate-800">+{row.day}</td>
                              <td className="px-3 py-2 text-slate-700">{formatDate(row.targetDate)}</td>
                              <td className="px-3 py-2 text-slate-700">{row.weekday}</td>
                              <td className="px-3 py-2">
                                <input
                                  type="checkbox"
                                  checked={plan.open}
                                  onChange={(event) =>
                                    updateDayPlan(row.day, { open: event.target.checked })
                                  }
                                  className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-300"
                                />
                              </td>
                              <td className="px-3 py-2">
                                <input
                                  type="checkbox"
                                  checked={plan.promo}
                                  onChange={(event) =>
                                    updateDayPlan(row.day, { promo: event.target.checked })
                                  }
                                  className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-300"
                                />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Link
              to={selectedStoreId ? `/upload?storeId=${selectedStoreId}` : "/upload"}
              className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-center text-sm font-medium text-slate-700 transition hover:-translate-y-0.5 hover:border-brand-300 hover:text-brand-700"
            >
              Upload Data
            </Link>
            <Link
              to={selectedStoreId ? `/store/${selectedStoreId}` : "/stores"}
              className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-center text-sm font-medium text-slate-700 transition hover:-translate-y-0.5 hover:border-brand-300 hover:text-brand-700"
            >
              Go to Store Detail
            </Link>
          </div>
        </Card>

        <Card title="Store Status">
          {loadingStatus ? (
            <LoadingState label="Refreshing status..." />
          ) : status ? (
            <div className="space-y-2 text-sm text-slate-700">
              <p>
                <span className="font-medium text-slate-900">Current date:</span>{" "}
                {formatDate(status.current_date)}
              </p>
              <p>
                <span className="font-medium text-slate-900">Last upload:</span>{" "}
                {formatDate(status.last_uploaded_date)}
              </p>
              <p>
                <span className="font-medium text-slate-900">Days uploaded:</span>{" "}
                {status.days_uploaded}
              </p>
              <p>
                <span className="font-medium text-slate-900">Gap days:</span>{" "}
                {status.gap_days ?? "N/A"}
              </p>
              <p className="pt-1">
                <span
                  className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${
                    status.ready_to_forecast
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {status.ready_to_forecast
                    ? "ready_to_forecast = true"
                    : "ready_to_forecast = false"}
                </span>
              </p>
            </div>
          ) : (
            <p className="text-sm text-slate-600">Select a store to view status.</p>
          )}
        </Card>

        {status && !status.ready_to_forecast && status.last_uploaded_date && (
          <div className="lg:col-span-3">
            <CatchUpCard
              storeId={selectedStoreId!}
              status={status}
              onComplete={() => {
                if (selectedStoreId) {
                  getStoreStatus(selectedStoreId).then((s) => setStatus(s)).catch(() => null);
                }
                setForecasts({});
                setHasLoadedForecast(false);
              }}
            />
          </div>
        )}
      </div>

      <Card
        title={`Next ${k} days forecast preview`}
        action={
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-brand-100 bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-700">
              Forecast window: +1 to +14
            </span>
            {scenarioSummary && (
              <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-700">
                Scenario: {scenarioSummary.closedDays} closed, {scenarioSummary.promoDays} promo
              </span>
            )}
            <div className="flex flex-col items-end gap-1">
              <button
                type="button"
                onClick={handleGenerateForecast}
                disabled={!selectedStoreId || loadingForecasts || loadingExplanations || (!!status && !status.ready_to_forecast)}
                className="rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:-translate-y-0.5 hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {loadingForecasts
                  ? "Generating..."
                  : loadingExplanations
                    ? "Loading Drivers..."
                    : "Generate Forecast"}
              </button>
              {status && !status.ready_to_forecast && (
                <p className="text-[11px] text-amber-700 font-medium">
                  Catch up {status.gap_days} missing day{status.gap_days === 1 ? "" : "s"} first
                </p>
              )}
            </div>
          </div>
        }
      >
        {loadingForecasts ? (
          <LoadingState label="Loading forecast preview..." />
        ) : !hasLoadedForecast ? (
          <p className="text-sm text-slate-600">
            Click <span className="font-semibold">Generate Forecast</span> to load forecast data.
          </p>
        ) : (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="metric-chip">
                <p className="subtle-label">Forecast Points</p>
                <p className="mt-1 text-xl font-bold text-ink-900">
                  {forecastStats.availableCount}/{k}
                </p>
              </div>
              <div className="metric-chip">
                <p className="subtle-label">Average Demand</p>
                <p className="mt-1 text-xl font-bold text-ink-900">
                  {forecastStats.average === null ? "N/A" : formatCustomers(forecastStats.average)}
                </p>
              </div>
            </div>

            {forecastStats.peakDay !== null && (() => {
              const peakRow = rows.find((r) => r.day === forecastStats.peakDay);
              return (
                <div className="flex items-center gap-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
                  <span className="text-sm font-semibold text-amber-800">Peak</span>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">Busiest Day</p>
                    <p className="text-base font-bold text-amber-900">
                      {peakRow?.weekday ?? `Day +${forecastStats.peakDay}`}
                      {peakRow?.targetDate ? `, ${formatDate(peakRow.targetDate)}` : ""}
                      {" - "}
                      <span>{formatCustomers(forecastStats.peakValue!)} customers expected</span>
                    </p>
                  </div>
                </div>
              );
            })()}

            <div className="rounded-xl border border-slate-200 bg-white/80 p-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                  Global Top Drivers ({showTechnicalDrivers ? "Technical View" : "Manager View"})
                </p>
                <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-600">
                  <input
                    type="checkbox"
                    checked={showTechnicalDrivers}
                    onChange={(event) => setShowTechnicalDrivers(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-300"
                  />
                  Show technical drivers
                </label>
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                {showTechnicalDrivers
                  ? "Technical view shows raw SHAP feature names used by the model."
                  : "Manager view prioritizes operational factors. Turn on technical drivers to inspect lag and seasonal features."}
              </p>
              {globalTopDrivers.length === 0 ? (
                <p className="mt-2 text-xs text-slate-500">
                  {loadingExplanations
                    ? "Forecast ready. Loading SHAP drivers..."
                    : hasExplanationData
                      ? "This run is mainly explained by technical demand-history signals. Turn on technical drivers to inspect them."
                      : "Generate forecast to see top SHAP drivers."}
                </p>
              ) : (
                <div className="mt-2 overflow-x-auto">
                  <table className="min-w-full text-xs">
                    <thead className="text-left text-slate-500">
                      <tr>
                        <th className="px-2 py-1.5">Driver</th>
                        <th className="px-2 py-1.5">Appears (days)</th>
                        <th className="px-2 py-1.5">Up Lifts</th>
                        <th className="px-2 py-1.5">Down Lowers</th>
                        <th className="px-2 py-1.5">Net</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {globalTopDrivers.map((driver) => {
                        const descriptor = describeDriverLabel(driver.label);
                        const net = getNetDirection(driver.upDays, driver.downDays);
                        return (
                          <tr key={driver.label}>
                            <td className="px-2 py-1.5">
                              <p className="font-medium">{descriptor.label}</p>
                              {descriptor.hint ? (
                                <p className="text-[11px] text-slate-500">{descriptor.hint}</p>
                              ) : null}
                            </td>
                            <td className="px-2 py-1.5">
                              {formatShare(driver.dayMentions, explanationDayCount)}
                            </td>
                            <td className="px-2 py-1.5 text-emerald-700">
                              {formatShare(driver.upDays, explanationDayCount)}
                            </td>
                            <td className="px-2 py-1.5 text-rose-700">
                              {formatShare(driver.downDays, explanationDayCount)}
                            </td>
                            <td className={`px-2 py-1.5 font-semibold ${net.className}`}>
                              {net.label}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <ForecastTrendChart rows={rows} />

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => {
                  const store = stores.find((s) => s.store_id === selectedStoreId);
                  exportForecastCSV(rows, store?.store_name ?? selectedStoreId ?? "store");
                }}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
              >
                Export CSV
              </button>
            </div>

            <div className="table-shell">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="whitespace-nowrap px-4 py-3">Day</th>
                    <th className="px-4 py-3">Target Date</th>
                    <th className="px-4 py-3">Weekday</th>
                    <th className="px-4 py-3">Open</th>
                    <th className="px-4 py-3">Promo</th>
                    <th className="px-4 py-3">Holiday</th>
                    <th className="px-4 py-3">Predicted Customers</th>
                    <th className="px-4 py-3">Suggested Staff</th>
                    <th className="px-4 py-3">Top Drivers</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white/80 text-slate-700">
                  {rows.map((row) => {
                    const drivers = getExplanationLabels(row.explanation, {
                      includeTechnical: showTechnicalDrivers,
                    });
                    return (
                      <tr key={row.day} className={getRowClass(row.weekday ?? "", row.stateHoliday ?? "0", row.schoolHoliday ?? 0)}>
                      <td className="whitespace-nowrap px-4 py-2.5 font-medium">Day +{row.day}</td>
                      <td className="px-4 py-2.5">{formatDate(row.targetDate)}</td>
                      <td className="px-4 py-2.5">{row.weekday || "N/A"}</td>
                      <td className="px-4 py-2.5">{row.plannedOpen === 1 ? "Yes" : "No"}</td>
                      <td className="px-4 py-2.5">{row.plannedPromo === 1 ? "Yes" : "No"}</td>
                      <td className="px-4 py-2.5">{row.holidayNote}</td>
                      <td className="px-4 py-2.5">
                        {row.prediction === null ? (
                          <span className="text-slate-400">N/A</span>
                        ) : (
                          formatCustomers(row.prediction)
                        )}
                      </td>
                      <td className="px-4 py-2.5">
                        {typeof row.suggestedStaff === "number" ? row.suggestedStaff : "N/A"}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="space-y-1 text-xs leading-tight">
                          {drivers.up ? (
                            <div className="text-emerald-700">
                              <span className="font-semibold">Likely lifts demand:</span> {drivers.up}
                            </div>
                          ) : null}
                          {drivers.down ? (
                            <div className="text-rose-700">
                              <span className="font-semibold">Likely lowers demand:</span> {drivers.down}
                            </div>
                          ) : null}
                          {!drivers.up && !drivers.down ? (
                            <span className="text-xs text-slate-400">
                              {loadingExplanations && row.prediction !== null ? "Loading..." : "N/A"}
                            </span>
                          ) : null}
                        </div>
                      </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}


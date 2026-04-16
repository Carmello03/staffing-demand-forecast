import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Card } from "../components/Card";
import { InlineAlert } from "../components/InlineAlert";
import { LoadingState } from "../components/LoadingState";
import {
  getForecastRange,
  getStoreMeta,
  getStoreProfile,
  getStoreStatus,
  patchStoreProfile,
  updateStoreMeta,
} from "../lib/api";
import type {
  StoreMeta,
  StoreMetaPatch,
  StoreStatus,
} from "../types";
import {
  aggregateManagerDrivers,
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
import { useParams } from "react-router-dom";
import { clampK, formatShare, getNetDirection } from "../utils/ui";

type MetaFormState = {
  store: string;
  storeType: "a" | "b" | "c" | "d";
  assortment: "a" | "b" | "c";
  competitionDistance: string;
  promo2: "0" | "1";
  promoInterval: string;
  competitionOpenSinceMonth: string;
  competitionOpenSinceYear: string;
  promo2SinceWeek: string;
  promo2SinceYear: string;
};

function createDefaultMetaForm(): MetaFormState {
  return {
    store: "",
    storeType: "a",
    assortment: "a",
    competitionDistance: "",
    promo2: "0",
    promoInterval: "",
    competitionOpenSinceMonth: "",
    competitionOpenSinceYear: "",
    promo2SinceWeek: "",
    promo2SinceYear: "",
  };
}

function normalizeStoreType(value: string | undefined): MetaFormState["storeType"] {
  const normalized = (value || "").toLowerCase();
  if (normalized === "a" || normalized === "b" || normalized === "c" || normalized === "d") {
    return normalized;
  }
  return "a";
}

function normalizeAssortment(value: string | undefined): MetaFormState["assortment"] {
  const normalized = (value || "").toLowerCase();
  if (normalized === "a" || normalized === "b" || normalized === "c") {
    return normalized;
  }
  return "a";
}

function toNumberText(value: number | undefined) {
  return value === undefined ? "" : String(value);
}

function parseOptionalNumber(label: string, rawValue: string) {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number(trimmed);
  if (Number.isNaN(parsed)) {
    throw new Error(`${label} must be a valid number.`);
  }
  return parsed;
}

const COUNTRY_OPTIONS = [
  { code: "DE", label: "Germany (DE)" },
  { code: "IE", label: "Ireland (IE)" },
  { code: "PL", label: "Poland (PL)" },
  { code: "GB", label: "United Kingdom (GB)" },
  { code: "FR", label: "France (FR)" },
  { code: "ES", label: "Spain (ES)" },
  { code: "IT", label: "Italy (IT)" },
] as const;

function normalizeCountryIso(value: string | undefined) {
  const raw = (value || "").trim();
  if (!raw) {
    return "DE";
  }

  const upper = raw.toUpperCase();
  const aliases: Record<string, string> = {
    GERMANY: "DE",
    DEUTSCHLAND: "DE",
    IRELAND: "IE",
    POLAND: "PL",
    "UNITED KINGDOM": "GB",
    UK: "GB",
    "GREAT BRITAIN": "GB",
    FRANCE: "FR",
    SPAIN: "ES",
    ITALY: "IT",
  };

  if (aliases[upper]) {
    return aliases[upper];
  }
  if (COUNTRY_OPTIONS.some((option) => option.code === upper)) {
    return upper;
  }
  return "DE";
}

export function StoreDetailPage() {
  const { storeId = "" } = useParams();
  const [status, setStatus] = useState<StoreStatus | null>(null);
  const [forecasts, setForecasts] = useState<ForecastMap>({});
  const [k, setK] = useState(7);
  const [loadingForecasts, setLoadingForecasts] = useState(false);
  const [loadingExplanations, setLoadingExplanations] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [hasLoadedForecast, setHasLoadedForecast] = useState(false);
  const [showTechnicalDrivers, setShowTechnicalDrivers] = useState(false);

  const [metaForm, setMetaForm] = useState<MetaFormState>(createDefaultMetaForm);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [savingMeta, setSavingMeta] = useState(false);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [metaSuccess, setMetaSuccess] = useState<string | null>(null);

  const [customersPerStaff, setCustomersPerStaff] = useState("");
  const [country, setCountry] = useState("DE");
  const [holidaySubdivision, setHolidaySubdivision] = useState("");

  const [loading, setLoading] = useState(true);
  const [savingProfile, setSavingProfile] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  function applyMetaToForm(storeMeta: StoreMeta | undefined) {
    const meta = storeMeta || {};
    setMetaForm({
      store: toNumberText(meta.Store),
      storeType: normalizeStoreType(meta.StoreType),
      assortment: normalizeAssortment(meta.Assortment),
      competitionDistance: toNumberText(meta.CompetitionDistance),
      promo2: Number(meta.Promo2 || 0) === 1 ? "1" : "0",
      promoInterval: meta.PromoInterval || "",
      competitionOpenSinceMonth: toNumberText(meta.CompetitionOpenSinceMonth),
      competitionOpenSinceYear: toNumberText(meta.CompetitionOpenSinceYear),
      promo2SinceWeek: toNumberText(meta.Promo2SinceWeek),
      promo2SinceYear: toNumberText(meta.Promo2SinceYear),
    });
  }

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      if (!storeId) {
        return;
      }

      setLoading(true);
      setError(null);
      const statusResult = await getStoreStatus(storeId)
        .then((value) => ({ ok: true as const, value }))
        .catch((fetchError: unknown) => ({ ok: false as const, fetchError }));

      if (cancelled) {
        return;
      }

      if (statusResult.ok) {
        setStatus(statusResult.value);
      } else {
        setStatus(null);
        setError(
          statusResult.fetchError instanceof Error
            ? statusResult.fetchError.message
            : "Unable to fetch store status.",
        );
      }

      setLoading(false);
    }

    void loadStatus();
    return () => {
      cancelled = true;
    };
  }, [storeId]);

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      if (!storeId) {
        return;
      }

      try {
        const response = await getStoreProfile(storeId);
        if (cancelled) {
          return;
        }

        const profile = response.profile || {};
        setCustomersPerStaff(
          typeof profile.customers_per_staff === "number"
            ? String(profile.customers_per_staff)
            : "",
        );
        setCountry(normalizeCountryIso(typeof profile.country === "string" ? profile.country : ""));
        setHolidaySubdivision(
          typeof profile.holiday_subdivision === "string" ? profile.holiday_subdivision : "",
        );
      } catch {
        if (!cancelled) {
          setCustomersPerStaff("");
          setCountry("DE");
          setHolidaySubdivision("");
        }
      }
    }

    void loadProfile();
    return () => {
      cancelled = true;
    };
  }, [storeId]);

  useEffect(() => {
    setForecasts({});
    setForecastError(null);
    setHasLoadedForecast(false);
    setLoadingForecasts(false);
    setLoadingExplanations(false);
  }, [storeId]);

  useEffect(() => {
    if (!storeId) {
      return;
    }
    if (!status?.current_date) {
      return;
    }

    const cached = readForecastRangeSession(storeId, k, status.current_date);
    if (!cached) {
      return;
    }

    setForecasts(mapForecastsByHorizon(cached.forecasts));
    setForecastError(null);
    setHasLoadedForecast(true);
  }, [k, status?.current_date, storeId]);

  useEffect(() => {
    let cancelled = false;

    async function loadMeta() {
      if (!storeId) {
        return;
      }

      setLoadingMeta(true);
      setMetaError(null);
      setMetaSuccess(null);

      try {
        const meta = await getStoreMeta(storeId);
        if (!cancelled) {
          applyMetaToForm(meta.store_meta);
        }
      } catch (fetchError) {
        if (!cancelled) {
          applyMetaToForm(undefined);
          setMetaError(
            fetchError instanceof Error
              ? fetchError.message
              : "Unable to fetch store metadata.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingMeta(false);
        }
      }
    }

    void loadMeta();
    return () => {
      cancelled = true;
    };
  }, [storeId]);

  const rows = useMemo(() => buildNextKRows(k, forecasts), [forecasts, k]);
  const driverInsights = useMemo(
    () =>
      aggregateManagerDrivers(rows.map((row) => row.explanation), {
        includeTechnical: showTechnicalDrivers,
      }),
    [rows, showTechnicalDrivers],
  );
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

  async function handleGenerateForecast() {
    if (!storeId) {
      return;
    }

    setLoadingForecasts(true);
    setLoadingExplanations(false);
    setForecastError(null);
    setHasLoadedForecast(true);

    const baseForecastResult = await getForecastRange(storeId, k, false)
      .then((value) => ({ ok: true as const, value }))
      .catch((fetchError: unknown) => ({ ok: false as const, fetchError }));

    if (!baseForecastResult.ok) {
      setForecasts({});
      setForecastError(
        baseForecastResult.fetchError instanceof Error
          ? baseForecastResult.fetchError.message
          : "Unable to fetch forecast data.",
      );
      setHasLoadedForecast(false);
      setLoadingForecasts(false);
      return;
    }

    setForecasts(mapForecastsByHorizon(baseForecastResult.value.forecasts));
    writeForecastRangeSession(storeId, k, baseForecastResult.value);
    setLoadingForecasts(false);

    setLoadingExplanations(true);
    const explanationResult = await getForecastRange(storeId, k, true)
      .then((value) => ({ ok: true as const, value }))
      .catch(() => ({ ok: false as const }));

    if (explanationResult.ok) {
      setForecasts(mapForecastsByHorizon(explanationResult.value.forecasts));
      writeForecastRangeSession(storeId, k, explanationResult.value);
    }
    setLoadingExplanations(false);
  }

  async function handleProfileSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!storeId) {
      return;
    }

    setSavingProfile(true);
    setError(null);
    setSuccess(null);

    const numericValue = customersPerStaff.trim();
    const payload = {
      customers_per_staff: numericValue ? Number(numericValue) : undefined,
      country: country.trim() || undefined,
      holiday_subdivision: holidaySubdivision.trim() || undefined,
    };

    try {
      if (
        payload.customers_per_staff !== undefined &&
        Number.isNaN(payload.customers_per_staff)
      ) {
        throw new Error("customers_per_staff must be a valid number.");
      }

      const response = await patchStoreProfile(storeId, payload);
      setSuccess(
        `Profile updated. customers_per_staff=${
          response.profile.customers_per_staff ?? "unset"
        }, country=${response.profile.country ?? "unset"}, holiday_subdivision=${
          response.profile.holiday_subdivision ?? "unset"
        }.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save profile.");
    } finally {
      setSavingProfile(false);
    }
  }

  async function handleMetaSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!storeId) {
      return;
    }

    setSavingMeta(true);
    setMetaError(null);
    setMetaSuccess(null);

    try {
      const storeNumber = parseOptionalNumber("Store", metaForm.store);
      if (storeNumber === undefined) {
        throw new Error("Store is required for forecasting.");
      }

      const payload: StoreMetaPatch = {
        Store: storeNumber,
        StoreType: metaForm.storeType,
        Assortment: metaForm.assortment,
        Promo2: Number(metaForm.promo2),
        PromoInterval: metaForm.promoInterval.trim(),
      };

      const competitionDistance = parseOptionalNumber(
        "CompetitionDistance",
        metaForm.competitionDistance,
      );
      if (competitionDistance !== undefined) {
        payload.CompetitionDistance = competitionDistance;
      }

      const competitionOpenSinceMonth = parseOptionalNumber(
        "CompetitionOpenSinceMonth",
        metaForm.competitionOpenSinceMonth,
      );
      if (competitionOpenSinceMonth !== undefined) {
        payload.CompetitionOpenSinceMonth = competitionOpenSinceMonth;
      }

      const competitionOpenSinceYear = parseOptionalNumber(
        "CompetitionOpenSinceYear",
        metaForm.competitionOpenSinceYear,
      );
      if (competitionOpenSinceYear !== undefined) {
        payload.CompetitionOpenSinceYear = competitionOpenSinceYear;
      }

      const promo2SinceWeek = parseOptionalNumber("Promo2SinceWeek", metaForm.promo2SinceWeek);
      if (promo2SinceWeek !== undefined) {
        payload.Promo2SinceWeek = promo2SinceWeek;
      }

      const promo2SinceYear = parseOptionalNumber("Promo2SinceYear", metaForm.promo2SinceYear);
      if (promo2SinceYear !== undefined) {
        payload.Promo2SinceYear = promo2SinceYear;
      }

      const response = await updateStoreMeta(storeId, payload);
      applyMetaToForm(response.store_meta);
      setMetaSuccess("Saved.");
    } catch (err) {
      setMetaError(err instanceof Error ? err.message : "Unable to save metadata.");
    } finally {
      setSavingMeta(false);
    }
  }

  return (
    <div className="space-y-5">
      {error && <InlineAlert kind="error" message={error} />}
      {forecastError && <InlineAlert kind="error" message={forecastError} />}
      {success && <InlineAlert kind="success" message={success} />}

      <Card title={`Store Detail: ${storeId}`}>
        {loading ? (
          <LoadingState label="Loading store status..." />
        ) : status ? (
          <div className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2 lg:grid-cols-4">
            <p>
              <span className="font-medium text-slate-900">Current date:</span>{" "}
              {formatDate(status.current_date)}
            </p>
            <p>
              <span className="font-medium text-slate-900">Last uploaded:</span>{" "}
              {formatDate(status.last_uploaded_date)}
            </p>
            <p>
              <span className="font-medium text-slate-900">Days uploaded:</span>{" "}
              {status.days_uploaded}
            </p>
            <p>
              <span className="font-medium text-slate-900">ready_to_forecast:</span>{" "}
              <span
                className={`rounded-full px-2 py-1 text-xs font-semibold ${
                  status.ready_to_forecast
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-amber-100 text-amber-800"
                }`}
              >
                {String(status.ready_to_forecast)}
              </span>
            </p>
          </div>
        ) : (
          <p className="text-sm text-slate-600">No status available.</p>
        )}
      </Card>

      <Card title="Store Setup / Metadata">
        {metaError && <InlineAlert kind="error" message={metaError} />}
        {metaSuccess && <InlineAlert kind="success" message={metaSuccess} />}

        {loadingMeta ? (
          <LoadingState label="Loading metadata..." />
        ) : (
          <form onSubmit={handleMetaSave} className="mt-3 grid gap-4 md:grid-cols-2">
            <label className="space-y-1">
              <span
                className="block text-sm font-medium text-slate-700"
                title="Required for forecasting. Usually the numeric store code."
              >
                Store
              </span>
              <input
                type="number"
                required
                value={metaForm.store}
                onChange={(event) =>
                  setMetaForm((prev) => ({ ...prev, store: event.target.value }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                placeholder="1001"
              />
            </label>

            <label className="space-y-1">
              <span
                className="block text-sm font-medium text-slate-700"
                title="Category code used for forecasting: Type A/B/C/D."
              >
                StoreType
              </span>
              <select
                value={metaForm.storeType}
                onChange={(event) =>
                  setMetaForm((prev) => ({
                    ...prev,
                    storeType: event.target.value as MetaFormState["storeType"],
                  }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              >
                <option value="a">Type A (forecast category)</option>
                <option value="b">Type B (forecast category)</option>
                <option value="c">Type C (forecast category)</option>
                <option value="d">Type D (forecast category)</option>
              </select>
            </label>

            <label className="space-y-1">
              <span
                className="block text-sm font-medium text-slate-700"
                title="Assortment level: Basic/Extra/Extended."
              >
                Assortment
              </span>
              <select
                value={metaForm.assortment}
                onChange={(event) =>
                  setMetaForm((prev) => ({
                    ...prev,
                    assortment: event.target.value as MetaFormState["assortment"],
                  }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              >
                <option value="a">Basic (a)</option>
                <option value="b">Extra (b)</option>
                <option value="c">Extended (c)</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="block text-sm font-medium text-slate-700">
                CompetitionDistance (meters)
              </span>
              <input
                type="number"
                step="0.01"
                value={metaForm.competitionDistance}
                onChange={(event) =>
                  setMetaForm((prev) => ({
                    ...prev,
                    competitionDistance: event.target.value,
                  }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                placeholder="430.5"
              />
            </label>

            <label className="space-y-1">
              <span className="block text-sm font-medium text-slate-700">
                Promo2 (Yes/No)
              </span>
              <select
                value={metaForm.promo2}
                onChange={(event) =>
                  setMetaForm((prev) => ({
                    ...prev,
                    promo2: event.target.value as MetaFormState["promo2"],
                  }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              >
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </label>

            <label className="space-y-1">
              <span className="block text-sm font-medium text-slate-700">
                PromoInterval (optional)
              </span>
              <input
                type="text"
                value={metaForm.promoInterval}
                onChange={(event) =>
                  setMetaForm((prev) => ({
                    ...prev,
                    promoInterval: event.target.value,
                  }))
                }
                className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                placeholder="Feb,May,Aug,Nov"
              />
            </label>

            <details className="rounded-xl border border-slate-200 bg-white/60 p-3 md:col-span-2">
              <summary className="cursor-pointer text-sm font-semibold text-slate-800">
                Advanced (optional)
              </summary>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="space-y-1">
                  <span className="block text-xs font-medium text-slate-600">
                    CompetitionOpenSinceMonth
                  </span>
                  <input
                    type="number"
                    value={metaForm.competitionOpenSinceMonth}
                    onChange={(event) =>
                      setMetaForm((prev) => ({
                        ...prev,
                        competitionOpenSinceMonth: event.target.value,
                      }))
                    }
                    className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    placeholder="9"
                  />
                </label>

                <label className="space-y-1">
                  <span className="block text-xs font-medium text-slate-600">
                    CompetitionOpenSinceYear
                  </span>
                  <input
                    type="number"
                    value={metaForm.competitionOpenSinceYear}
                    onChange={(event) =>
                      setMetaForm((prev) => ({
                        ...prev,
                        competitionOpenSinceYear: event.target.value,
                      }))
                    }
                    className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    placeholder="2012"
                  />
                </label>

                <label className="space-y-1">
                  <span className="block text-xs font-medium text-slate-600">Promo2SinceWeek</span>
                  <input
                    type="number"
                    value={metaForm.promo2SinceWeek}
                    onChange={(event) =>
                      setMetaForm((prev) => ({
                        ...prev,
                        promo2SinceWeek: event.target.value,
                      }))
                    }
                    className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    placeholder="13"
                  />
                </label>

                <label className="space-y-1">
                  <span className="block text-xs font-medium text-slate-600">Promo2SinceYear</span>
                  <input
                    type="number"
                    value={metaForm.promo2SinceYear}
                    onChange={(event) =>
                      setMetaForm((prev) => ({
                        ...prev,
                        promo2SinceYear: event.target.value,
                      }))
                    }
                    className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    placeholder="2011"
                  />
                </label>
              </div>
            </details>

            <div className="md:col-span-2">
              <button
                type="submit"
                disabled={savingMeta}
                className="rounded-xl bg-brand-500 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {savingMeta ? "Saving..." : "Save Metadata"}
              </button>
            </div>
          </form>
        )}
      </Card>

      <Card title="Profile Editor">
        <form onSubmit={handleProfileSave} className="grid gap-4 md:grid-cols-2">
          <label className="space-y-1">
            <span className="block text-sm font-medium text-slate-700">
              customers_per_staff
            </span>
            <input
              type="number"
              step="0.01"
              value={customersPerStaff}
              onChange={(event) => setCustomersPerStaff(event.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              placeholder="12.5"
            />
          </label>

          <label className="space-y-1">
            <span className="block text-sm font-medium text-slate-700">country</span>
            <select
              value={country}
              onChange={(event) => setCountry(normalizeCountryIso(event.target.value))}
              className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            >
              {COUNTRY_OPTIONS.map((option) => (
                <option key={option.code} value={option.code}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-1">
            <span className="block text-sm font-medium text-slate-700">
              holiday_subdivision (optional)
            </span>
            <input
              type="text"
              value={holidaySubdivision}
              onChange={(event) => setHolidaySubdivision(event.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              placeholder="DE-BY"
            />
          </label>

          <div className="md:col-span-2">
            <button
              type="submit"
              disabled={savingProfile}
              className="rounded-xl bg-brand-500 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {savingProfile ? "Saving..." : "Save Profile"}
            </button>
          </div>
        </form>
      </Card>

      <Card
        title="Forecast - Next K Days"
        action={
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <label htmlFor="detail-k" className="font-medium">
              K:
            </label>
            <select
              id="detail-k"
              value={k}
              onChange={(event) => {
                const next = Number(event.target.value);
                setK(clampK(next));
                setForecasts({});
                setForecastError(null);
                setHasLoadedForecast(false);
                setLoadingExplanations(false);
              }}
              className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs"
            >
              {Array.from({ length: 14 }, (_, i) => i + 1).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <div className="flex flex-col items-end gap-1">
              <button
                type="button"
                onClick={handleGenerateForecast}
                disabled={loadingForecasts || loadingExplanations || !storeId || (!!status && !status.ready_to_forecast)}
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
          <LoadingState label="Loading forecasts..." />
        ) : !hasLoadedForecast ? (
          <p className="text-sm text-slate-600">
            Click <span className="font-semibold">Generate Forecast</span> to load forecast data.
          </p>
        ) : (
          <>
            <p className="mb-3 text-xs text-slate-500">Forecast window supports day +1 through +14.</p>

            <div className="table-shell">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="whitespace-nowrap px-4 py-3">Day</th>
                    <th className="px-4 py-3">Target Date</th>
                    <th className="px-4 py-3">Weekday</th>
                    <th className="px-4 py-3">Holiday</th>
                    <th className="px-4 py-3">Predicted Customers</th>
                    <th className="px-4 py-3">Suggested Staff</th>
                    <th className="px-4 py-3">Top Drivers</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white/80">
                  {rows.map((row) => {
                    const drivers = getExplanationLabels(row.explanation, {
                      includeTechnical: showTechnicalDrivers,
                    });
                    return (
                      <tr key={row.day} className="transition hover:bg-sky-50/70">
                      <td className="whitespace-nowrap px-4 py-2.5 font-medium">Day +{row.day}</td>
                      <td className="px-4 py-2.5 text-slate-700">{formatDate(row.targetDate)}</td>
                      <td className="px-4 py-2.5 text-slate-700">{row.weekday || "N/A"}</td>
                      <td className="px-4 py-2.5 text-slate-700">{row.holidayNote}</td>
                      <td className="px-4 py-2.5 text-slate-700">
                        {row.prediction === null ? (
                          <span className="text-slate-400">-</span>
                        ) : (
                          formatCustomers(row.prediction)
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-slate-700">
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
          </>
        )}
      </Card>

      <Card title="Key Drivers">
        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-4">
            <p className="text-sm font-semibold text-emerald-800">Main Uplift Driver</p>
            <p className="mt-1 text-xs text-emerald-700">
              {driverInsights.topUp
                ? `${driverInsights.topUp.label} (upward on ${driverInsights.topUp.up}/${k} days)`
                : "Generate forecast to see which factor most often lifts demand."}
            </p>
          </div>

          <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4">
            <p className="text-sm font-semibold text-amber-800">Main Downward Driver</p>
            <p className="mt-1 text-xs text-amber-700">
              {driverInsights.topDown
                ? `${driverInsights.topDown.label} (downward on ${driverInsights.topDown.down}/${k} days)`
                : "Generate forecast to see which factor most often reduces demand."}
            </p>
          </div>

          <div className="rounded-xl border border-sky-200 bg-sky-50/70 p-4">
            <p className="text-sm font-semibold text-sky-800">Most Frequent Driver</p>
            <p className="mt-1 text-xs text-sky-700">
              {driverInsights.topOverall
                ? `${driverInsights.topOverall.label} (mentioned on ${driverInsights.topOverall.total}/${k} days)`
                : "Generate forecast to see the most repeated demand driver."}
            </p>
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-slate-200 bg-white/80 p-3">
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
                    <th className="px-2 py-1.5">Times in Top Drivers</th>
                    <th className="px-2 py-1.5">Days Increasing Forecast</th>
                    <th className="px-2 py-1.5">Days Reducing Forecast</th>
                    <th className="px-2 py-1.5">Net Direction</th>
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
                        <td className={`px-2 py-1.5 font-semibold ${net.className}`}>{net.label}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

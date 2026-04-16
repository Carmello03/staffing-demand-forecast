import type { ForecastExplanation, ForecastResponse } from "../types";

const DAY_MS = 24 * 60 * 60 * 1000;
export type ForecastMap = Partial<Record<number, ForecastResponse>>;

function toIsoDate(date: Date) {
  return date.toISOString().slice(0, 10);
}

function toWeekday(dateIso: string) {
  const dt = new Date(dateIso);
  if (Number.isNaN(dt.getTime())) {
    return "";
  }
  return dt.toLocaleDateString(undefined, { weekday: "long" });
}

export function formatDate(value: string | null | undefined) {
  if (!value) {
    return "N/A";
  }
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) {
    return value;
  }
  return dt.toLocaleDateString();
}

export function formatCustomers(value: number) {
  return Math.round(value).toLocaleString();
}

export function mapForecastsByHorizon(forecasts: ForecastResponse[]): ForecastMap {
  const mapped: ForecastMap = {};
  forecasts.forEach((forecast) => {
    mapped[forecast.horizon] = forecast;
  });
  return mapped;
}

export function getIssueDate(forecasts: ForecastMap) {
  return forecasts[1]?.issue_date || forecasts[7]?.issue_date || forecasts[14]?.issue_date;
}

export function buildNextKRows(k: number, forecasts: ForecastMap) {
  const issueDate = getIssueDate(forecasts);
  const issueDateTime = issueDate ? new Date(issueDate).getTime() : Date.now();

  return Array.from({ length: k }, (_, index) => {
    const day = index + 1;
    const targetDate = toIsoDate(new Date(issueDateTime + day * DAY_MS));
    const horizonForecast = forecasts[day];
    const holidayNote = getHolidayNote(horizonForecast);

    return {
      day,
      targetDate,
      weekday: horizonForecast?.weekday || toWeekday(targetDate),
      plannedOpen: horizonForecast?.planned_open ?? 1,
      plannedPromo: horizonForecast?.planned_promo ?? 0,
      stateHoliday: horizonForecast?.state_holiday ?? "0",
      schoolHoliday: horizonForecast?.school_holiday ?? 0,
      holidayNote,
      prediction: horizonForecast ? horizonForecast.prediction_customers : null,
      suggestedStaff: horizonForecast?.suggested_staff ?? null,
      explanation: horizonForecast?.explanation ?? null,
    };
  });
}

const STATE_HOLIDAY_LABEL: Record<string, string> = {
  a: "Public holiday",
  b: "Easter holiday",
  c: "Christmas holiday",
};

function getHolidayNote(forecast: ForecastResponse | undefined) {
  if (!forecast) {
    return "N/A";
  }

  const notes: string[] = [];
  const stateCode = String(forecast.state_holiday || "0").toLowerCase();
  if (stateCode !== "0") {
    const label = forecast.public_holiday_name || STATE_HOLIDAY_LABEL[stateCode] || "Public holiday";
    notes.push(`Public: ${label}`);
  }

  if (Number(forecast.school_holiday || 0) === 1) {
    const label = forecast.school_holiday_name || "School holiday";
    notes.push(`School: ${label}`);
  }

  if (notes.length === 0) {
    return "-";
  }
  return notes.join(" | ");
}

const TECHNICAL_DRIVER_PATTERNS = [
  /^(cat__)?store[\s_=]/i,
  /^storetype[\s_=]/i,
  /^assortment[\s_=]/i,
  /^competitiondistance$/i,
  /^promointerval[\s_=]/i,
  /^promo2$/i,
  /^lag\d+$/i,
  /^roll\d+[\s_]*(mean|avg|average|sum|min|max|std)?$/i,
  /^dow[\s_]*(sin|cos)$/i,
  /^month[\s_]*(sin|cos)$/i,
  /^dayofyear[\s_]*(sin|cos)$/i,
  /^trend$/i,
];

type DriverViewOptions = {
  includeTechnical?: boolean;
};

function isTechnicalDriverLabel(label: string) {
  return TECHNICAL_DRIVER_PATTERNS.some((pattern) => pattern.test(label));
}

const DRIVER_HINT_PATTERNS: Array<{ pattern: RegExp; hint: string }> = [
  { pattern: /^lag1$/i, hint: "Yesterday's demand pattern." },
  { pattern: /^lag7$/i, hint: "Same weekday demand from last week." },
  { pattern: /^lag14$/i, hint: "Same weekday demand from two weeks ago." },
  { pattern: /^roll7[\s_]*mean$/i, hint: "Recent 7-day average demand level." },
  { pattern: /^schoolholiday$/i, hint: "School holiday calendar effect." },
  { pattern: /^stateholiday$/i, hint: "Public holiday calendar effect." },
  { pattern: /^dow[\s_]*cos$/i, hint: "Weekly seasonality pattern component." },
  { pattern: /^dow[\s_]*sin$/i, hint: "Weekly seasonality pattern component." },
  { pattern: /^planned[\s_]*promo$/i, hint: "Planned promotion for that day." },
  { pattern: /^planned[\s_]*open$/i, hint: "Planned open or closed status." },
  { pattern: /^promo$/i, hint: "Promotion activity impact." },
];

export function describeDriverLabel(label: string) {
  const normalized = label.trim();
  const matched = DRIVER_HINT_PATTERNS.find((item) => item.pattern.test(normalized));
  return {
    label: normalized,
    hint: matched?.hint,
    isTechnical: isTechnicalDriverLabel(normalized),
  };
}

function isStoreIdFeature(factor: { feature?: string; label?: string }) {
  const raw = factor.feature ?? "";
  return /^(cat__)?store_\d/i.test(raw);
}

function pickDriverLabel(
  factors: ForecastExplanation["top_positive_factors"] | ForecastExplanation["top_negative_factors"],
  includeTechnical: boolean,
) {
  for (const factor of factors) {
    if (!includeTechnical && isStoreIdFeature(factor)) {
      continue;
    }
    const label = factor.label?.trim();
    if (!label) {
      continue;
    }
    if (isTechnicalDriverLabel(label)) {
      if (includeTechnical) {
        return label;
      }
      continue;
    }
    return label;
  }
  return undefined;
}

export function getExplanationLabels(
  explanation: ForecastExplanation | null | undefined,
  options?: DriverViewOptions,
) {
  if (!explanation) {
    return { up: undefined, down: undefined };
  }
  const includeTechnical = options?.includeTechnical ?? false;
  const up = pickDriverLabel(explanation.top_positive_factors, includeTechnical);
  const down = pickDriverLabel(explanation.top_negative_factors, includeTechnical);
  return { up, down };
}

export function aggregateManagerDrivers(
  explanations: Array<ForecastExplanation | null | undefined>,
  options?: DriverViewOptions,
) {
  const counts: Record<string, { up: number; down: number }> = {};
  const includeTechnical = options?.includeTechnical ?? false;

  explanations.forEach((explanation) => {
    if (!explanation) {
      return;
    }

    const upLabel = pickDriverLabel(explanation.top_positive_factors, includeTechnical);
    const downLabel = pickDriverLabel(explanation.top_negative_factors, includeTechnical);

    if (upLabel) {
      counts[upLabel] = counts[upLabel] || { up: 0, down: 0 };
      counts[upLabel].up += 1;
    }
    if (downLabel) {
      counts[downLabel] = counts[downLabel] || { up: 0, down: 0 };
      counts[downLabel].down += 1;
    }
  });

  const entries = Object.entries(counts).map(([label, value]) => ({
    label,
    up: value.up,
    down: value.down,
    total: value.up + value.down,
  }));

  const topUp = entries.slice().sort((a, b) => b.up - a.up)[0] || null;
  const topDown = entries.slice().sort((a, b) => b.down - a.down)[0] || null;
  const topOverall = entries.slice().sort((a, b) => b.total - a.total)[0] || null;

  return { topUp, topDown, topOverall };
}

export function buildGlobalTopDrivers(
  explanations: Array<ForecastExplanation | null | undefined>,
  topN = 10,
  options?: DriverViewOptions,
) {
  const includeTechnical = options?.includeTechnical ?? false;
  const summary: Record<
    string,
    {
      label: string;
      mentions: number;
      upDays: number;
      downDays: number;
      dayMentions: number;
      absImpactSum: number;
    }
  > = {};

  explanations.forEach((explanation) => {
    if (!explanation) {
      return;
    }

    const dayUpLabels = new Set<string>();
    const dayDownLabels = new Set<string>();

    explanation.top_positive_factors.forEach((factor) => {
      if (!includeTechnical && isStoreIdFeature(factor)) {
        return;
      }
      const mappedLabel = factor.label?.trim();
      if (!mappedLabel) {
        return;
      }
      if (!includeTechnical && isTechnicalDriverLabel(mappedLabel)) {
        return;
      }
      const label = mappedLabel;
      summary[label] = summary[label] || {
        label,
        mentions: 0,
        upDays: 0,
        downDays: 0,
        dayMentions: 0,
        absImpactSum: 0,
      };
      summary[label].mentions += 1;
      summary[label].absImpactSum += Math.abs(factor.abs_shap_value ?? factor.shap_value ?? 0);
      dayUpLabels.add(label);
    });

    explanation.top_negative_factors.forEach((factor) => {
      if (!includeTechnical && isStoreIdFeature(factor)) {
        return;
      }
      const mappedLabel = factor.label?.trim();
      if (!mappedLabel) {
        return;
      }
      if (!includeTechnical && isTechnicalDriverLabel(mappedLabel)) {
        return;
      }
      const label = mappedLabel;
      summary[label] = summary[label] || {
        label,
        mentions: 0,
        upDays: 0,
        downDays: 0,
        dayMentions: 0,
        absImpactSum: 0,
      };
      summary[label].mentions += 1;
      summary[label].absImpactSum += Math.abs(factor.abs_shap_value ?? factor.shap_value ?? 0);
      dayDownLabels.add(label);
    });

    const dayMentionLabels = new Set([...dayUpLabels, ...dayDownLabels]);
    dayUpLabels.forEach((label) => {
      summary[label].upDays += 1;
    });
    dayDownLabels.forEach((label) => {
      summary[label].downDays += 1;
    });
    dayMentionLabels.forEach((label) => {
      summary[label].dayMentions += 1;
    });
  });

  return Object.values(summary)
    .map((item) => ({
      ...item,
      avgAbsImpact: item.mentions > 0 ? item.absImpactSum / item.mentions : 0,
    }))
    .sort((a, b) => {
      if (b.absImpactSum !== a.absImpactSum) {
        return b.absImpactSum - a.absImpactSum;
      }
      return b.mentions - a.mentions;
    })
    .slice(0, topN);
}


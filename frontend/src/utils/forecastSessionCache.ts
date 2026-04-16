import type { ForecastRangeResponse } from "../types";

const SESSION_CACHE_PREFIX = "forecast_range_v5";

function cacheKey(storeId: string, k: number, scenarioKey: string) {
  return `${SESSION_CACHE_PREFIX}:${storeId}:${k}:${scenarioKey}`;
}

export function readForecastRangeSession(
  storeId: string,
  k: number,
  expectedIssueDate: string,
  scenarioKey = "default",
): ForecastRangeResponse | null {
  if (!storeId) {
    return null;
  }

  const raw = sessionStorage.getItem(cacheKey(storeId, k, scenarioKey));
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as ForecastRangeResponse;
    if (
      !parsed ||
      parsed.store_id !== storeId ||
      !Array.isArray(parsed.forecasts) ||
      typeof parsed.issue_date !== "string" ||
      parsed.issue_date !== expectedIssueDate
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function writeForecastRangeSession(
  storeId: string,
  k: number,
  value: ForecastRangeResponse,
  scenarioKey = "default",
) {
  if (!storeId) {
    return;
  }
  sessionStorage.setItem(cacheKey(storeId, k, scenarioKey), JSON.stringify(value));
}

import type {
  CatchUpDayIn,
  CatchUpResponse,
  ForecastRangeRequest,
  ForecastRangeResponse,
  Store,
  StoreMetaPatch,
  StoreMetaResponse,
  StoreProfilePatch,
  StoreProfileResponse,
  StoreStatus,
  UploadResponse,
} from "../types";
import { auth } from "./firebase";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getAuthToken(): Promise<string | null> {
  try {
    if (!auth.currentUser) {
      return null;
    }
    return await auth.currentUser.getIdToken();
  } catch {
    return null;
  }
}

async function buildAuthHeaders(options: RequestInit, includeJsonContentType = true) {
  const token = await getAuthToken();
  const headers = new Headers(options.headers ?? {});
  const isFormData = options.body instanceof FormData;

  if (includeJsonContentType && !isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  return headers;
}

async function parseErrorMessage(response: Response) {
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") {
      return data.detail;
    }
    if (typeof data?.message === "string") {
      return data.message;
    }
  } catch {
  }

  if (response.statusText) {
    return response.statusText;
  }
  return "Request failed. Please try again.";
}

async function request<T>(path: string, options: RequestInit = {}) {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL is missing.");
  }

  const headers = await buildAuthHeaders(options);

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorMessage(response), response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function requestRaw(path: string, options: RequestInit = {}) {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL is missing.");
  }

  const headers = await buildAuthHeaders(options, false);
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    throw new ApiError(await parseErrorMessage(response), response.status);
  }

  return response;
}

export function getStores() {
  return request<Store[]>("/stores");
}

export function createStore(storeName: string) {
  return request<Store>("/stores", {
    method: "POST",
    body: JSON.stringify({ store_name: storeName }),
  });
}

export function uploadStoreCsv(storeId: string, file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return request<UploadResponse>(`/stores/${storeId}/upload`, {
    method: "POST",
    body: formData,
  });
}

export function getStoreStatus(storeId: string) {
  return request<StoreStatus>(`/stores/${storeId}/status`);
}

export function patchStoreProfile(storeId: string, payload: StoreProfilePatch) {
  return request<StoreProfileResponse>(`/stores/${storeId}/profile`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function getStoreProfile(storeId: string) {
  return request<StoreProfileResponse>(`/stores/${storeId}/profile`);
}

export function getForecastRange(storeId: string, k: number, includeExplanations = false) {
  const query = includeExplanations
    ? `k=${k}&include_explanations=true`
    : `k=${k}`;
  return request<ForecastRangeResponse>(`/stores/${storeId}/forecast-range?${query}`);
}

export function postForecastRange(storeId: string, payload: ForecastRangeRequest) {
  return request<ForecastRangeResponse>(`/stores/${storeId}/forecast-range`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getStoreMeta(storeId: string) {
  return request<StoreMetaResponse>(`/stores/${storeId}/meta`);
}

export function updateStoreMeta(storeId: string, payload: StoreMetaPatch) {
  return request<StoreMetaResponse>(`/stores/${storeId}/meta`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function postCatchUp(storeId: string, days: CatchUpDayIn[]) {
  return request<CatchUpResponse>(`/stores/${storeId}/catch-up`, {
    method: "POST",
    body: JSON.stringify(days),
  });
}

export async function downloadCatchUpTemplate(storeId: string): Promise<void> {
  const response = await requestRaw(`/stores/${storeId}/catch-up-template`);

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="([^"]+)"/);
  a.download = match ? match[1] : `catch-up-${storeId.slice(0, 8)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}


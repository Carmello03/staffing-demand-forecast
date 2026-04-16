import type { Store } from "../types";

export const SELECTED_STORE_ID_KEY = "selected_store_id";

export function readSelectedStoreId() {
  return localStorage.getItem(SELECTED_STORE_ID_KEY);
}

export function saveSelectedStoreId(storeId: string) {
  localStorage.setItem(SELECTED_STORE_ID_KEY, storeId);
}

export function clearSelectedStoreId() {
  localStorage.removeItem(SELECTED_STORE_ID_KEY);
}

function isValidStoreId(stores: Store[], storeId: string | null | undefined): storeId is string {
  return Boolean(storeId) && stores.some((store) => store.store_id === storeId);
}

export function resolveStoreId(
  stores: Store[],
  ...candidates: Array<string | null | undefined>
) {
  for (const candidate of candidates) {
    if (isValidStoreId(stores, candidate)) {
      return candidate;
    }
  }

  return stores[0]?.store_id || "";
}

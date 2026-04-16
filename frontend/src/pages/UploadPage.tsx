import { useEffect, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { Card } from "../components/Card";
import { InlineAlert } from "../components/InlineAlert";
import { StoreSelector } from "../components/StoreSelector";
import { useStores } from "../contexts/StoresContext";
import { uploadStoreCsv } from "../lib/api";
import type { UploadResponse } from "../types";
import { readSelectedStoreId, resolveStoreId, saveSelectedStoreId } from "../utils/storeSelection";

export function UploadPage() {
  const [params] = useSearchParams();
  const { stores, loading: loadingStores, error: storesError } = useStores();
  const [selectedStoreId, setSelectedStoreId] = useState(params.get("storeId") || "");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (stores.length === 0) {
      setSelectedStoreId("");
      return;
    }

    const preferredStore = params.get("storeId") || readSelectedStoreId();
    setSelectedStoreId((current) => resolveStoreId(stores, preferredStore, current));
  }, [params, stores]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedStoreId || !file) {
      setError("Select a store and choose a CSV file first.");
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const result: UploadResponse = await uploadStoreCsv(selectedStoreId, file);
      const rows =
        typeof result.rows_written === "number"
          ? ` Uploaded rows: ${result.rows_written}.`
          : "";
      setSuccess(`CSV uploaded successfully.${rows}`);
      saveSelectedStoreId(selectedStoreId);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="space-y-5">
      {storesError && <InlineAlert kind="error" message={storesError} />}
      {error && <InlineAlert kind="error" message={error} />}
      {success && <InlineAlert kind="success" message={success} />}

      <Card title="Upload Store CSV">
        <p className="mb-4 text-sm text-slate-600">
          Required columns: <span className="font-medium">Date, Customers, Open, Promo</span>.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <label className="space-y-1">
            <span className="block text-sm font-medium text-slate-700">Store</span>
            <StoreSelector
              stores={stores}
              value={selectedStoreId}
              onChange={setSelectedStoreId}
              disabled={loadingStores || uploading}
              id="upload-store-select"
            />
          </label>

          <label className="block space-y-1">
            <span className="block text-sm font-medium text-slate-700">CSV file</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const chosen = event.target.files?.[0] ?? null;
                setFile(chosen);
              }}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-brand-500 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:bg-brand-700"
            />
          </label>

          <button
            type="submit"
            disabled={uploading || loadingStores || !selectedStoreId || !file}
            className="rounded-xl bg-brand-500 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {uploading ? "Uploading..." : "Upload Data"}
          </button>
        </form>
      </Card>
    </div>
  );
}

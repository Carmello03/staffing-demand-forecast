import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { Card } from "../components/Card";
import { InlineAlert } from "../components/InlineAlert";
import { LoadingState } from "../components/LoadingState";
import { useStores } from "../contexts/StoresContext";
import { createStore, getStoreStatus } from "../lib/api";
import type { StoreStatus } from "../types";

function ReadinessBadge({ status }: { status: StoreStatus | undefined }) {
  if (!status) return <span className="text-xs text-slate-400">—</span>;
  if (status.ready_to_forecast) {
    return <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">Ready</span>;
  }
  if (!status.last_uploaded_date) {
    return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-500">No data</span>;
  }
  if (status.gap_days !== null && status.gap_days <= 7) {
    return <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-700">{status.gap_days}d behind</span>;
  }
  return <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-700">Outdated</span>;
}

export function StoresPage() {
  const { stores, loading, error: storesError, refreshStores } = useStores();
  const [storeName, setStoreName] = useState("");
  const [creating, setCreating] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Record<string, StoreStatus>>({});

  useEffect(() => {
    if (stores.length === 0) return;
    Promise.all(
      stores.map((s) =>
        getStoreStatus(s.store_id)
          .then((status) => ({ id: s.store_id, status }))
          .catch(() => null),
      ),
    ).then((results) => {
      const map: Record<string, StoreStatus> = {};
      results.forEach((r) => { if (r) map[r.id] = r.status; });
      setStatuses(map);
    });
  }, [stores]);

  async function handleCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setActionError(null);
    setSuccess(null);
    setCreating(true);

    try {
      const created = await createStore(storeName.trim());
      setSuccess(`Created store "${created.store_name}" (${created.store_id.slice(0, 8)}...).`);
      setStoreName("");
      await refreshStores();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to create store.");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="space-y-5">
      {storesError && <InlineAlert kind="error" message={storesError} />}
      {actionError && <InlineAlert kind="error" message={actionError} />}
      {success && <InlineAlert kind="success" message={success} />}

      <Card title="Create Store">
        <form onSubmit={handleCreate} className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex-1 space-y-1">
            <span className="text-sm font-medium text-slate-700">Store name</span>
            <input
              type="text"
              required
              value={storeName}
              onChange={(event) => setStoreName(event.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white/90 px-3 py-2 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              placeholder="Downtown Branch"
            />
          </label>

          <button
            type="submit"
            disabled={creating || !storeName.trim()}
            className="rounded-xl bg-brand-500 px-4 py-2 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {creating ? "Creating..." : "Create"}
          </button>
        </form>
      </Card>

      <Card title="Store List">
        {loading ? (
          <LoadingState label="Loading stores..." />
        ) : stores.length === 0 ? (
          <p className="text-sm text-slate-600">No stores yet. Create your first store above.</p>
        ) : (
          <div className="table-shell">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-3">Store Name</th>
                  <th className="px-4 py-3">Store ID</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white/80">
                {stores.map((store) => (
                  <tr key={store.store_id} className="transition hover:bg-sky-50/70">
                    <td className="px-4 py-2.5 font-medium text-slate-800">{store.store_name}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-600">{store.store_id}</td>
                    <td className="px-4 py-2.5"><ReadinessBadge status={statuses[store.store_id]} /></td>
                    <td className="px-4 py-2.5">
                      <div className="flex gap-2">
                        <Link
                          to={`/store/${store.store_id}`}
                          className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                        >
                          Details
                        </Link>
                        <Link
                          to={`/upload?storeId=${store.store_id}`}
                          className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-700"
                        >
                          Upload CSV
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

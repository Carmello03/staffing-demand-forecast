import type { Store } from "../types";

type StoreSelectorProps = {
  stores: Store[];
  value: string;
  onChange: (storeId: string) => void;
  disabled?: boolean;
  id?: string;
};

export function StoreSelector({
  stores,
  value,
  onChange,
  disabled = false,
  id = "store-id",
}: StoreSelectorProps) {
  return (
    <select
      id={id}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-xl border border-slate-300/90 bg-white px-3 py-2 text-sm font-medium text-slate-800 shadow-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100 disabled:cursor-not-allowed disabled:bg-slate-100"
    >
      <option value="">Select a store</option>
      {stores.map((store) => (
        <option key={store.store_id} value={store.store_id}>
          {store.store_name} ({store.store_id.slice(0, 8)})
        </option>
      ))}
    </select>
  );
}

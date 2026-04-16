import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { getStores } from "../lib/api";
import type { Store } from "../types";
import { useAuth } from "./AuthContext";

type StoresContextValue = {
  stores: Store[];
  loading: boolean;
  error: string | null;
  refreshStores: () => Promise<void>;
};

const StoresContext = createContext<StoresContextValue | undefined>(undefined);

export function StoresProvider({ children }: { children: ReactNode }) {
  const { token } = useAuth();
  const [stores, setStores] = useState<Store[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshStores = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getStores();
      setStores(data);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Unable to fetch stores.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!token) {
      setStores([]);
      setLoading(false);
      setError(null);
      return;
    }

    void refreshStores();
  }, [refreshStores, token]);

  const value = useMemo<StoresContextValue>(
    () => ({
      stores,
      loading,
      error,
      refreshStores,
    }),
    [error, loading, refreshStores, stores],
  );

  return <StoresContext.Provider value={value}>{children}</StoresContext.Provider>;
}

export function useStores() {
  const context = useContext(StoresContext);
  if (!context) {
    throw new Error("useStores must be used inside StoresProvider");
  }
  return context;
}

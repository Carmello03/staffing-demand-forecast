import {
  onIdTokenChanged,
  signInWithEmailAndPassword,
  signOut,
  type User,
} from "firebase/auth";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { clearSelectedStoreId } from "../utils/storeSelection";
import { auth } from "../lib/firebase";

type AuthContextValue = {
  user: User | null;
  token: string | null;
  isInitializing: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    const unsubscribe = onIdTokenChanged(auth, async (nextUser) => {
      setUser(nextUser);

      if (!nextUser) {
        setToken(null);
        setIsInitializing(false);
        return;
      }

      try {
        const nextToken = await nextUser.getIdToken();
        setToken(nextToken);
      } catch (tokenError) {
        console.error("Failed to resolve Firebase ID token:", tokenError);
        setToken(null);
      }
      setIsInitializing(false);
    });

    return unsubscribe;
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const result = await signInWithEmailAndPassword(auth, email, password);
    try {
      const nextToken = await result.user.getIdToken();
      setToken(nextToken);
    } catch (tokenError) {
      console.error("Failed to resolve Firebase ID token after login:", tokenError);
      setToken(null);
    }
  }, []);

  const logout = useCallback(async () => {
    await signOut(auth);
    clearSelectedStoreId();
    setToken(null);
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      isInitializing,
      login,
      logout,
    }),
    [isInitializing, login, logout, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}

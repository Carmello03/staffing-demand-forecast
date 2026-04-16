import { Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { LoadingState } from "./components/LoadingState";
import { useAuth } from "./contexts/AuthContext";
import { StoresProvider } from "./contexts/StoresContext";
import { DashboardPage } from "./pages/DashboardPage";
import { LoginPage } from "./pages/LoginPage";
import { StoreDetailPage } from "./pages/StoreDetailPage";
import { StoresPage } from "./pages/StoresPage";
import { UploadPage } from "./pages/UploadPage";

function ProtectedShell() {
  const { token, isInitializing } = useAuth();

  if (isInitializing) {
    return (
      <div className="mx-auto mt-20 max-w-xl px-4">
        <LoadingState label="Checking login..." />
      </div>
    );
  }

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return (
    <StoresProvider>
      <AppLayout>
        <Outlet />
      </AppLayout>
    </StoresProvider>
  );
}

function LoginRoute() {
  const { token, isInitializing } = useAuth();

  if (isInitializing) {
    return (
      <div className="mx-auto mt-20 max-w-xl px-4">
        <LoadingState label="Checking login..." />
      </div>
    );
  }

  if (token) {
    return <Navigate to="/dashboard" replace />;
  }

  return <LoginPage />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/login" element={<LoginRoute />} />

      <Route element={<ProtectedShell />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/stores" element={<StoresPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/store/:storeId" element={<StoreDetailPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

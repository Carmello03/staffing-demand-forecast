import type { ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const links = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/stores", label: "Stores" },
  { to: "/upload", label: "Upload" },
];

export function AppLayout({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  async function handleLogout() {
    await logout();
    navigate("/login");
  }

  return (
    <div className="min-h-screen bg-app-gradient pb-10">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <header className="glass-card mb-6 overflow-hidden border border-slate-200/80">
          <div className="relative">
            <div className="pointer-events-none absolute -right-10 -top-10 h-28 w-28 rounded-full bg-brand-100/60 blur-2xl" />
            <div className="pointer-events-none absolute -left-10 -bottom-8 h-24 w-24 rounded-full bg-accent-100/60 blur-2xl" />
          </div>

          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-700">
                Staffing Demand Forecast
              </p>
              <h1 className="text-2xl font-extrabold tracking-tight text-ink-900">
                Manager Dashboard
              </h1>
              <p className="text-sm text-slate-600">
                Monitor demand, upload latest data, and review forecasts.
              </p>
            </div>

            <div className="flex flex-col gap-3 lg:items-end">
              <nav className="flex flex-wrap gap-2">
                {links.map((link) => (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    className={({ isActive }) =>
                      `rounded-xl px-3.5 py-1.5 text-sm font-semibold transition ${
                        isActive
                          ? "bg-brand-500 text-white shadow-glow"
                          : "border border-slate-200 bg-white/85 text-slate-700 hover:border-brand-200 hover:text-ink-900"
                      }`
                    }
                  >
                    {link.label}
                  </NavLink>
                ))}
              </nav>

              <div className="flex w-full items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white/90 px-3 py-2 shadow-sm lg:w-auto">
                <div className="min-w-0 max-w-[14rem] sm:max-w-[20rem]">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    Signed In As
                  </p>
                  <p className="truncate text-sm font-semibold text-ink-900">
                    {user?.email || "Logged in"}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleLogout}
                  className="rounded-lg border border-accent-100 bg-accent-50 px-3 py-1.5 text-xs font-bold text-accent-700 transition hover:-translate-y-0.5 hover:bg-accent-100"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </header>

        <main className="space-y-5">{children}</main>
      </div>
    </div>
  );
}

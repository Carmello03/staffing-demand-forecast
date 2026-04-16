import { Component, type ReactNode } from "react";

type AppErrorBoundaryProps = {
  children: ReactNode;
};

type AppErrorBoundaryState = {
  hasError: boolean;
  message: string;
};

export class AppErrorBoundary extends Component<
  AppErrorBoundaryProps,
  AppErrorBoundaryState
> {
  constructor(props: AppErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      message: "",
    };
  }

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return {
      hasError: true,
      message: error.message || "Unexpected app error.",
    };
  }

  override componentDidCatch(error: Error) {
    console.error("App render error:", error);
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-app-gradient px-4 py-10">
          <div className="mx-auto max-w-2xl rounded-2xl border border-rose-200 bg-white p-6 shadow-lg">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-rose-700">
              Frontend Error
            </p>
            <h1 className="mt-2 text-xl font-bold text-slate-900">
              The app failed to render
            </h1>
            <p className="mt-3 rounded-lg border border-rose-100 bg-rose-50 px-3 py-2 text-sm text-rose-800">
              {this.state.message}
            </p>
            <p className="mt-3 text-sm text-slate-600">
              Open browser DevTools console for full stack trace.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

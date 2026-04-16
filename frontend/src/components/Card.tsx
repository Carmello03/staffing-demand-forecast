import type { PropsWithChildren, ReactNode } from "react";

type CardProps = PropsWithChildren<{
  title?: string;
  action?: ReactNode;
  className?: string;
}>;

export function Card({ title, action, className = "", children }: CardProps) {
  return (
    <section className={`glass-card hover-lift ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex items-center justify-between gap-4">
          {title ? (
            <h2 className="text-lg font-bold tracking-tight text-ink-900">{title}</h2>
          ) : (
            <span />
          )}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

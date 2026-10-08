import type { ReactNode } from 'react';

export default function Card({ title, blurb, children }: { title: string; blurb: string; children: ReactNode }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <p className="muted">{blurb}</p>
      {children}
    </section>
  );
}

export function ErrorLine({ error }: { error: string | null }) {
  return error ? <div className="out err">{error}</div> : null;
}

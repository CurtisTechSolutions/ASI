export function fmt(x: number, digits = 3): string {
  return x.toFixed(digits);
}

/** A visible glyph for the space byte in labels. */
export function unitLabel(u: string): string {
  return u === ' ' ? '␣' : u;
}

/** Blank lines separate texts. */
export function splitTexts(raw: string): string[] {
  return raw
    .split(/\n\s*\n/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
}

export async function readFiles(files: FileList | null): Promise<string[]> {
  const out: string[] = [];
  for (const f of files ?? []) out.push(await f.text());
  return out;
}

export function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

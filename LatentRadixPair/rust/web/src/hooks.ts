import { useCallback, useEffect, useState } from 'react';
import { api, Info } from './api';
import { errorText } from './util';

/** The model summary, refreshed on demand. */
export function useInfo() {
  const [info, setInfo] = useState<Info | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      setInfo(await api<Info>('/api/info'));
      setError(null);
    } catch (err) {
      setError(errorText(err));
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { info, error, refresh };
}

/** A button-driven request: its busy flag, its error, and the last result. */
export function useAction<T>() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<T | null>(null);
  const run = useCallback(async (work: () => Promise<T>) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await work());
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }, []);
  return { busy, error, result, run, setResult };
}

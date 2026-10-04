import { useCallback, useEffect, useState } from "react";

export interface StaticState<T> {
  data?: T;
  error?: Error;
  loading: boolean;
  retry: () => void;
}

/** Load a bundled static JSON file; these load instantly and never depend on the API. */
export function useStatic<T>(loader: () => Promise<T>): StaticState<T> {
  const [state, setState] = useState<{ data?: T; error?: Error; loading: boolean }>({ loading: true });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    loader().then(
      (data) => live && setState({ data, loading: false }),
      (error: Error) => live && setState({ error, loading: false }),
    );
    return () => {
      live = false;
    };
  }, [loader, nonce]);

  const retry = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, retry };
}

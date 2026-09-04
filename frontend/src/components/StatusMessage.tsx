import { ApiUnavailableError, NotFoundError } from "../api/ApiError";

interface StatusMessageProps {
  loading: boolean;
  error: Error | null;
  empty: boolean;
  emptyMessage?: string;
  notFoundMessage?: string;
  children?: React.ReactNode;
}

/** Central loading / empty / error / not-found messaging with aria-live. */
export function StatusMessage({
  loading,
  error,
  empty,
  emptyMessage = "Nothing to show yet.",
  notFoundMessage = "Not found.",
  children,
}: StatusMessageProps) {
  if (loading) {
    return (
      <div className="status status--loading" role="status" aria-live="polite">
        <span className="status__spinner" aria-hidden="true" /> Loading…
      </div>
    );
  }
  if (error) {
    if (error instanceof NotFoundError) {
      return (
        <div className="status status--notfound" role="status">
          <h2>Not found</h2>
          <p>{notFoundMessage}</p>
        </div>
      );
    }
    const unavailable = error instanceof ApiUnavailableError;
    return (
      <div className="status status--error" role="alert">
        <h2>{unavailable ? "Backend unavailable" : "Something went wrong"}</h2>
        <p>
          {unavailable
            ? "The RevenueGuard API could not be reached. Check that the backend is running."
            : "The request could not be completed. Please try again."}
        </p>
      </div>
    );
  }
  if (empty) {
    return (
      <div className="status status--empty" role="status">
        <p>{emptyMessage}</p>
      </div>
    );
  }
  return <>{children}</>;
}

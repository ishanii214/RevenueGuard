/** Typed API errors so the UI can distinguish states without string matching. */

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export class NotFoundError extends ApiError {
  constructor(path: string) {
    super(404, `Not found: ${path}`);
    this.name = "NotFoundError";
  }
}

export class ApiUnavailableError extends ApiError {
  constructor() {
    super(503, "Backend unavailable");
    this.name = "ApiUnavailableError";
  }
}

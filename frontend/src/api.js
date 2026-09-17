export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

let csrfToken = "";

function errorMessage(data, status) {
  if (typeof data?.detail === "string") return data.detail;
  if (typeof data?.error === "string") return data.error;
  const messages = Object.values(data || {}).filter((value) => Array.isArray(value) && typeof value[0] === "string");
  if (messages.length) return messages.flat().join(" ");
  return `The server returned an unexpected response (HTTP ${status}). Please try again.`;
}

export async function apiRequest(path, { method = "GET", body, signal, timeoutMs = 15_000 } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  const timeout = setTimeout(abort, timeoutMs);
  try {
    const response = await fetch(`${API_BASE}/api/v1${path}`, {
      method,
      credentials: "include",
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        ...(method !== "GET" && csrfToken ? { "X-CSRFToken": csrfToken } : {}),
      },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      signal: controller.signal,
    });
    if (response.status === 204) return null;
    const data = await response.json().catch(() => null);
    if (!response.ok || !data) {
      const error = new Error(errorMessage(data, response.status));
      error.status = response.status;
      throw error;
    }
    if (data.csrf_token) csrfToken = data.csrf_token;
    return data;
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      const timeoutError = new Error("The request took too long. Please try again.");
      timeoutError.name = "TimeoutError";
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

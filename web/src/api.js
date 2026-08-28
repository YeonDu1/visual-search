// Client for the Flask API in api/app.py.
//
// Every endpoint there answers with JSON, including errors ({error, status}),
// so a failed response still carries a message worth showing. A fetch that
// rejects outright is a different failure -- the server is down, the port is
// wrong, or CORS blocked us -- and the UI needs to say so rather than print
// "Failed to fetch".

export const API_URL = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'
).replace(/\/+$/, '')

/** Thrown when the API answered but with a non-2xx status. */
export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Thrown when the API could not be reached at all. */
export class NetworkError extends Error {
  constructor(cause) {
    super(`Cannot reach the API at ${API_URL}`)
    this.name = 'NetworkError'
    this.cause = cause
  }
}

async function request(path, init) {
  let response
  try {
    response = await fetch(`${API_URL}${path}`, init)
  } catch (cause) {
    throw new NetworkError(cause)
  }

  let payload = null
  try {
    payload = await response.json()
  } catch {
    // Non-JSON body: fall through to the status-based message below.
  }

  if (!response.ok) {
    const detail = payload?.error || `${response.status} ${response.statusText}`
    throw new ApiError(detail, response.status)
  }
  if (payload === null) {
    throw new ApiError('API returned a non-JSON body', response.status)
  }
  return payload
}

export function health({ signal } = {}) {
  return request('/health', { signal })
}

export function searchText(query, k, { signal } = {}) {
  return request('/search/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k }),
    signal,
  })
}

function imageForm(file, k) {
  const form = new FormData()
  form.append('file', file)
  form.append('k', String(k))
  return form
}

export function searchImage(file, k, { signal } = {}) {
  return request('/search/image', {
    method: 'POST',
    body: imageForm(file, k),
    signal,
  })
}

export function classify(file, k = 5, { signal } = {}) {
  return request('/classify', {
    method: 'POST',
    body: imageForm(file, k),
    signal,
  })
}

/** Message for any error the calls above can raise. */
export function errorMessage(error) {
  if (error instanceof NetworkError) {
    return `${error.message}. Start it with: flask --app api.app run --debug`
  }
  return error?.message || String(error)
}

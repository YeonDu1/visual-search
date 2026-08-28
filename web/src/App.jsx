import { useCallback, useEffect, useRef, useState } from 'react'
import {
  API_URL,
  classify,
  errorMessage,
  health,
  searchImage,
  searchText,
} from './api'
import './App.css'

const K_OPTIONS = [5, 10, 20]
const DEFAULT_K = 10
const CLASSIFY_K = 5
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

/** Aborts from an effect cleanup are not failures; they must not reach the UI. */
function isAbort(error) {
  return error?.name === 'AbortError' || error?.cause?.name === 'AbortError'
}

function formatScore(score) {
  return score.toFixed(3)
}

function formatPercent(probability) {
  return `${(probability * 100).toFixed(1)}%`
}

function ResultCard({ product }) {
  return (
    <li className="card">
      <div className="card-image">
        <img
          src={product.image_url}
          alt={product.productDisplayName || ''}
          loading="lazy"
        />
        <span className="score" title="cosine similarity">
          {formatScore(product.score)}
        </span>
      </div>
      <p className="article-type">{product.articleType || '—'}</p>
      <p className="product-name">
        {product.productDisplayName || `Product ${product.id}`}
      </p>
    </li>
  )
}

function Classification({ predictions }) {
  return (
    <section className="classification" aria-label="Zero-shot subCategory">
      <h2>
        Zero-shot subCategory{' '}
        <span className="hint">top {predictions.length}</span>
      </h2>
      <ul>
        {predictions.map((prediction, index) => (
          <li
            key={prediction.label}
            className={index === 0 ? 'top' : undefined}
          >
            <span className="label">{prediction.label}</span>
            <span className="bar" aria-hidden="true">
              <span
                style={{
                  width: `${Math.max(prediction.probability * 100, 1)}%`,
                }}
              />
            </span>
            <span className="probability">
              {formatPercent(prediction.probability)}
            </span>
            <span className="cosine">cos {formatScore(prediction.score)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Dropzone({ onFile, disabled, preview }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const take = (file) => {
    if (file) onFile(file)
  }

  return (
    <div
      className={`dropzone${dragging ? ' dragging' : ''}`}
      onDragEnter={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragOver={(event) => {
        event.preventDefault()
        event.dataTransfer.dropEffect = 'copy'
      }}
      onDragLeave={(event) => {
        // Ignore the leave events fired when crossing into a child element.
        if (!event.currentTarget.contains(event.relatedTarget))
          setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        take(event.dataTransfer.files?.[0])
      }}
    >
      {preview ? (
        <img className="preview" src={preview.url} alt={preview.name} />
      ) : (
        <span className="dropzone-icon" aria-hidden="true">
          &#8681;
        </span>
      )}
      <p className="dropzone-text">
        <span className="dropzone-title">
          {preview ? preview.name : 'Drop an image here'}
        </span>
        <button
          type="button"
          className="link"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          {preview ? 'choose another' : 'or choose a file'}
        </button>
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          take(event.target.files?.[0])
          // Reset so re-selecting the same file fires change again.
          event.target.value = ''
        }}
      />
    </div>
  )
}

export default function App() {
  const [status, setStatus] = useState({ state: 'checking' })
  const [text, setText] = useState('')
  const [k, setK] = useState(DEFAULT_K)
  const [query, setQuery] = useState(null) // {kind:'text',text} | {kind:'image',file}
  const [preview, setPreview] = useState(null) // {url, name}
  const [results, setResults] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // --- health -------------------------------------------------------------
  const checkHealth = useCallback(
    (signal) =>
      health({ signal })
        .then((info) => setStatus({ state: 'ok', info }))
        .catch((cause) => {
          if (isAbort(cause)) return
          setStatus({ state: 'down', message: errorMessage(cause) })
        }),
    [],
  )

  useEffect(() => {
    const controller = new AbortController()
    checkHealth(controller.signal)
    return () => controller.abort()
  }, [checkHealth])

  // --- searching ------------------------------------------------------------
  // Driven by the events that change a query -- submit, drop, k -- rather than
  // by an effect on [query, k]: an effect would also fire on mount and twice
  // under StrictMode. Only the newest request may write state, so the previous
  // one is aborted and its handlers bail out on the abort.
  const inFlight = useRef(null)

  const statusState = status.state

  const runSearch = useCallback(
    async (nextQuery, nextK) => {
      inFlight.current?.abort()
      const controller = new AbortController()
      inFlight.current = controller
      const { signal } = controller
      setLoading(true)
      setError(null)

      try {
        let response
        let classification = null
        if (nextQuery.kind === 'text') {
          response = await searchText(nextQuery.text, nextK, { signal })
        } else {
          ;[response, classification] = await Promise.all([
            searchImage(nextQuery.file, nextK, { signal }),
            classify(nextQuery.file, CLASSIFY_K, { signal }),
          ])
        }
        setResults(response.results)
        setPredictions(classification ? classification.predictions : null)
        // A search that succeeds proves the API came back; refresh the badge.
        if (statusState === 'down') checkHealth()
      } catch (cause) {
        if (isAbort(cause)) return
        setError(errorMessage(cause))
        setResults(null)
        setPredictions(null)
      } finally {
        // A superseded search must not clear the newer one's spinner.
        if (!signal.aborted) setLoading(false)
      }
    },
    [checkHealth, statusState],
  )

  useEffect(() => () => inFlight.current?.abort(), [])

  // Object URLs leak unless revoked once the preview is replaced.
  useEffect(() => {
    if (!preview) return undefined
    return () => URL.revokeObjectURL(preview.url)
  }, [preview])

  const submitText = (event) => {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed) return
    const next = { kind: 'text', text: trimmed }
    setPreview(null)
    setPredictions(null)
    setQuery(next)
    runSearch(next, k)
  }

  const submitFile = useCallback(
    (file) => {
      if (!file.type.startsWith('image/')) {
        setError(
          `${file.name} is not an image (${file.type || 'unknown type'})`,
        )
        return
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        const mb = (file.size / 1024 / 1024).toFixed(1)
        setError(`${file.name} is ${mb}MB; the API accepts at most 10MB`)
        return
      }
      const next = { kind: 'image', file }
      setText('')
      setPreview({ url: URL.createObjectURL(file), name: file.name })
      setQuery(next)
      runSearch(next, k)
    },
    [k, runSearch],
  )

  const changeK = (event) => {
    const nextK = Number(event.target.value)
    setK(nextK)
    if (query) runSearch(query, nextK)
  }

  const unreachable = status.state === 'down' && !results

  return (
    <div className="app">
      <header>
        <h1>Visual search</h1>
        <p className="subtitle">
          CLIP bidirectional search over the Fashion Product Images catalog.
          {status.state === 'ok' && (
            <span className="badge ok">
              {status.info.index_size.toLocaleString()} products &middot;{' '}
              {status.info.device}
            </span>
          )}
          {status.state === 'down' && (
            <span className="badge down">API offline</span>
          )}
        </p>
      </header>

      {unreachable && (
        <div className="banner error" role="alert">
          <strong>Cannot reach the API at {API_URL}</strong>
          <span>
            Start it with <code>flask --app api.app run --debug</code>, or set{' '}
            <code>VITE_API_URL</code> in <code>web/.env</code> if it runs
            somewhere else.
          </span>
        </div>
      )}

      <div className="controls">
        <form className="text-search" onSubmit={submitText}>
          <input
            type="search"
            value={text}
            placeholder="red running shoes for men"
            aria-label="Text query"
            onChange={(event) => setText(event.target.value)}
          />
          <button type="submit" disabled={!text.trim() || loading}>
            Search
          </button>
        </form>

        <label className="k-selector">
          Results
          <select value={k} onChange={changeK} disabled={loading}>
            {K_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      <Dropzone onFile={submitFile} disabled={loading} preview={preview} />

      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}

      {predictions && <Classification predictions={predictions} />}

      <section className="results" aria-live="polite" aria-busy={loading}>
        {loading && <p className="status">Searching&hellip;</p>}

        {!loading && !query && !unreachable && (
          <p className="status">
            Search by text, or drop a product image to find visually similar
            items.
          </p>
        )}

        {!loading && results?.length === 0 && (
          <p className="status">No results.</p>
        )}

        {results?.length > 0 && (
          <>
            <h2 className="results-heading">
              Top {results.length}
              {query?.kind === 'text' ? (
                <span className="hint"> for &ldquo;{query.text}&rdquo;</span>
              ) : (
                <span className="hint"> visually similar</span>
              )}
            </h2>
            <ul className={`grid${loading ? ' stale' : ''}`}>
              {results.map((product) => (
                <ResultCard key={product.id} product={product} />
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  )
}

// Every call to the Python server goes through here.
//
// The backend answers {ok: true, ...} or {ok: false, error} and almost never
// throws -- the agent degrades rather than raising -- so the useful failure mode
// for the UI is "show me what went wrong", not an empty panel. `request` turns
// both transport failures and {ok:false} into one thrown Error carrying the
// message, and every caller surfaces it.

async function request(path, options) {
  let response
  try {
    response = await fetch(path, options)
  } catch (cause) {
    throw new Error(`cannot reach the distil server — is it running? (${cause.message})`)
  }
  let payload
  try {
    payload = await response.json()
  } catch {
    throw new Error(`${response.status} ${response.statusText}: response was not JSON`)
  }
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`)
  }
  return payload
}

const query = (params) => {
  const usable = Object.entries(params || {}).filter(
    ([, v]) => v !== undefined && v !== null && v !== '',
  )
  return usable.length ? `?${new URLSearchParams(usable)}` : ''
}

export const get = (name, params) => request(`/api/${name}${query(params)}`)

export const post = (name, body) =>
  request(`/api/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })

// Colour by trace kind, shared by every visualisation so a `failure` is the same
// red in the scatter plot, the recall list and the memory table.
export const KIND_COLOUR = {
  query: '#7aa2f7',
  answer: '#9ece6a',
  goal: '#e0af68',
  chain: '#bb9af7',
  tool: '#2ac3de',
  fact: '#73daca',
  failure: '#f7768e',
  idea: '#ff9e64',
  game: '#c0caf5',
  solution: '#41a6b5',
  digest: '#565f89',
}

export const colourOf = (kind) => KIND_COLOUR[kind] || '#9aa5ce'

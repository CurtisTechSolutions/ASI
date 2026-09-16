// Everything the system can do, as things you type.
//
// This replaced a row of tabs. Tabs make you navigate to a system; a command
// line lets you talk to it, and -- the part that matters here -- the result of
// every one lands in the same transcript as everything else, in the order it
// happened. What used to be six views you had to switch between and correlate by
// memory is now one scrollable record.
//
// Anything that is not a command is a question, which is the common case, so the
// common case costs no syntax at all.

import { get, post } from './api.js'

export const COMMANDS = [
  {
    name: 'tools',
    args: '',
    blurb: 'what it can do, and what each capability solved',
    run: async () => ({ kind: 'tools', data: await get('tools') }),
  },
  {
    name: 'forge',
    args: '<what the tool should do>',
    blurb: 'write, verify and register a new capability',
    needs: 'say what the tool should do, e.g. /forge count words in a string',
    run: async (rest) => ({ kind: 'forge', data: await post('forge', { goal: rest }) }),
  },
  {
    name: 'memory',
    args: '',
    blurb: 'every trace, projected onto a plane',
    run: async () => ({ kind: 'memory', data: await get('memory') }),
  },
  {
    name: 'recall',
    args: '<query>',
    blurb: 'search memory and show why each hit ranked where it did',
    needs: 'say what to search for, e.g. /recall merging dictionaries',
    run: async (rest) => ({ kind: 'recall', data: await get('recall', { q: rest, k: 8 }) }),
  },
  {
    name: 'explore',
    args: '[n]',
    blurb: 'brainstorm and run experiments, ranked by what they would teach',
    run: async (rest) => ({
      kind: 'explore',
      data: await post('explore', { steps: Math.min(20, Math.max(1, parseInt(rest, 10) || 3)) }),
    }),
  },
  {
    name: 'cases',
    args: '[like what]',
    blurb: 'problems, and what actually solved them',
    run: async (rest) => ({ kind: 'cases', data: await get('cases', rest ? { like: rest } : {}) }),
  },
  {
    name: 'clarify',
    args: '<task>',
    blurb: 'sharpen a task without committing to solving it',
    needs: 'give a task to sharpen, e.g. /clarify make the thing better',
    run: async (rest) => ({ kind: 'clarify', data: await post('clarify', { task: rest }) }),
  },
  {
    name: 'trace',
    args: '<id>',
    blurb: 'one memory, and everything it is linked to',
    needs: 'give a trace id -- every answer and every recall hit shows one',
    run: async (rest) => ({ kind: 'trace', data: await get('trace', { id: rest }) }),
  },
  {
    name: 'mcp',
    args: '[add <name> <command…> | remove <name>]',
    blurb: 'remote tool servers, in the same embedding layer as local ones',
    run: async (rest) => {
      const [verb, name, ...command] = rest.split(/\s+/).filter(Boolean)
      if (verb === 'add') {
        if (!name || !command.length) throw new Error('usage: /mcp add <name> <command> [args…]')
        return { kind: 'mcp', data: await post('mcp_attach', { action: 'add', name, command }) }
      }
      if (verb === 'remove') {
        if (!name) throw new Error('usage: /mcp remove <name>')
        return { kind: 'mcp', data: await post('mcp_attach', { action: 'remove', name }) }
      }
      return { kind: 'mcp', data: await get('mcp') }
    },
  },
  {
    name: 'seed',
    args: '',
    blurb: 'plant the starter toolkit (runs by itself on an empty memory)',
    run: async () => ({ kind: 'seed', data: await post('seed', {}) }),
  },
  {
    name: 'compress',
    args: '[now]',
    blurb: 'consolidate cold memory into digests; previews unless you say now',
    run: async (rest) => ({
      kind: 'compress',
      data: await post('compress', { dry_run: rest.trim() !== 'now' }),
    }),
  },
  {
    name: 'tune',
    args: '',
    blurb: 'tune its own policy against measured outcomes',
    run: async () => ({ kind: 'tune', data: await post('upgrade', { trials: 6 }) }),
  },
  {
    name: 'system',
    args: '',
    blurb: 'providers, policy, and every number it may change about itself',
    run: async () => ({ kind: 'system', data: await get('state') }),
  },
  {
    name: 'help',
    args: '',
    blurb: 'this list',
    run: async () => ({ kind: 'help', data: { commands: COMMANDS } }),
  },
]

const BY_NAME = new Map(COMMANDS.map((c) => [c.name, c]))

// A leading slash and a known word. `/` alone, or an unknown word, is left as
// ordinary text -- guessing at a near-miss would silently run something else.
export function parse(input) {
  const text = input.trim()
  if (!text.startsWith('/')) return null
  const [word, ...rest] = text.slice(1).split(/\s+/)
  const command = BY_NAME.get(word.toLowerCase())
  if (!command) return { unknown: word }
  return { command, rest: rest.join(' ').trim() }
}

// What to offer while someone is part-way through typing a command.
export function suggest(input) {
  const text = input.trim()
  if (!text.startsWith('/') || text.includes(' ')) return []
  const typed = text.slice(1).toLowerCase()
  return COMMANDS.filter((c) => c.name.startsWith(typed))
}

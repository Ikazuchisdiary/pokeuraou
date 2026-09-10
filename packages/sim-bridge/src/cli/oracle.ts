/**
 * JSONL-over-stdio oracle server.
 *
 * One JSON request per line in, one JSON response per line out. Requests carry batches
 * so the Python side never pays a round trip per item.
 *
 *   {"id":1,"op":"spread_stats","format":"...","rows":[{species,nature,sp}, ...]}
 *   {"id":2,"op":"create","format":"...","teams":[[set,...],[set,...]],"policy":{...}}
 *   {"id":3,"op":"step","session":0,"choices":["move 1 1","move 2 2"]}
 *   {"id":4,"op":"probe_choices","session":0,"side":0,"candidates":["move 1 1", ...]}
 *   {"id":5,"op":"action_speeds","session":0}
 *   {"id":5,"op":"validate_team","format":"...","team":[set, ...]}
 *   {"id":6,"op":"close","session":0}
 *   {"id":7,"op":"ping"}
 *
 * Responses: {"id":N,"ok":true,"result":...} or {"id":N,"ok":false,"error":"..."}
 */
import * as readline from 'readline';
import { OracleSession, spreadStats, validateTeam, importTeam, type CreateRequest } from '../oracle';

const sessions = new Map<number, OracleSession>();
let nextSession = 0;

function session(id: unknown): OracleSession {
	const s = sessions.get(Number(id));
	if (!s) throw new Error(`No such session: ${String(id)}`);
	return s;
}

function handle(req: Record<string, unknown>): unknown {
	switch (req.op) {
		case 'ping':
			return { pong: true, pid: process.pid, node: process.version };

		case 'spread_stats':
			return spreadStats(
				String(req.format),
				req.rows as { species: string; nature: string; sp: Record<string, number> }[]
			);

		case 'validate_team': {
			const problems = validateTeam(String(req.format), req.team as never);
			return { ok: problems === null, problems: problems ?? [] };
		}

		case 'import_team':
			return importTeam(String(req.text));

		case 'create': {
			const s = new OracleSession(req as unknown as CreateRequest);
			const id = nextSession++;
			sessions.set(id, s);
			return { session: id, ...s.result() };
		}

		case 'step':
			return session(req.session).step(req.choices as (string | null)[]);

		case 'state':
			return session(req.session).result();

		case 'action_speeds':
			return session(req.session).actionSpeeds();

		case 'probe_choices':
			return session(req.session).probeChoices(Number(req.side), req.candidates as string[]);

		case 'close':
			sessions.delete(Number(req.session));
			return { closed: true };

		default:
			throw new Error(`Unknown op: ${String(req.op)}`);
	}
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', line => {
	const text = line.trim();
	if (!text) return;
	let id: unknown = null;
	try {
		const req = JSON.parse(text) as Record<string, unknown>;
		id = req.id ?? null;
		process.stdout.write(`${JSON.stringify({ id, ok: true, result: handle(req) })}\n`);
	} catch (err) {
		const message = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
		process.stdout.write(`${JSON.stringify({ id, ok: false, error: message })}\n`);
	}
});
rl.on('close', () => process.exit(0));

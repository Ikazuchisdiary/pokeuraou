/**
 * Writes configs/regulations/<formatId>.json from the pinned Showdown dex.
 *
 *   node dist/cli/dump-regulation.js [formatId ...]
 */
import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import { buildRegulationConfig, coverageSummary } from '../regulation';

const DEFAULT_FORMATS = ['gen9championsvgc2026regmc', 'gen9championsvgc2026regmb'];

function repoRoot(): string {
	return path.resolve(__dirname, '..', '..', '..', '..');
}

function showdownCommit(root: string): string {
	try {
		return execFileSync('git', ['-C', path.join(root, 'vendor', 'pokemon-showdown'), 'rev-parse', 'HEAD'], {
			encoding: 'utf8',
		}).trim();
	} catch {
		return 'unknown';
	}
}

function main() {
	const root = repoRoot();
	const commit = showdownCommit(root);
	const outDir = path.join(root, 'configs', 'regulations');
	fs.mkdirSync(outDir, { recursive: true });

	const formats = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_FORMATS;
	for (const formatId of formats) {
		const cfg = buildRegulationConfig(formatId, commit);
		const out = path.join(outDir, `${cfg.meta.formatId}.json`);
		fs.writeFileSync(out, `${JSON.stringify(cfg, null, '\t')}\n`);
		const cov = coverageSummary(cfg);
		console.log(
			`${cfg.meta.formatId}  ->  ${path.relative(root, out)}  (${(fs.statSync(out).size / 1e6).toFixed(2)} MB)`
		);
		console.log(
			`  species ${cov.species.total} (team-legal ${cov.species.teamLegal})` +
			`  moves ${cov.moves.total} (custom code ${cov.moves.withCode})` +
			`  items ${cov.items.total} (custom code ${cov.items.withCode})` +
			`  abilities ${cov.abilities.total} (custom code ${cov.abilities.withCode})`
		);
	}
}

main();

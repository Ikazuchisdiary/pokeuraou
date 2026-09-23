/**
 * Writes configs/regulations/<formatId>.json from the pinned Showdown dex.
 *
 *   node dist/cli/dump-regulation.js [formatId ...]
 */
import * as fs from 'fs';
import * as path from 'path';
import { buildRegulationConfig, coverageSummary } from '../regulation';
import { showdownCommit, showdownGitlink } from '../showdown-commit';

const DEFAULT_FORMATS = ['gen9championsvgc2026regmc', 'gen9championsvgc2026regmb'];

function repoRoot(): string {
	return path.resolve(__dirname, '..', '..', '..', '..');
}

function main() {
	const root = repoRoot();
	// The dex comes from whatever `pokemon-showdown` resolves to, so that is the checkout
	// whose commit is recorded -- not a path assumed to hold it (IKA-152).
	const showdownDir = path.dirname(require.resolve('pokemon-showdown/package.json'));
	const commit = showdownCommit(root, showdownDir);
	const outDir = path.join(root, 'configs', 'regulations');
	fs.mkdirSync(outDir, { recursive: true });

	const formats = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_FORMATS;
	for (const formatId of formats) {
		const cfg = buildRegulationConfig(formatId, commit);
		const out = path.join(outDir, `${cfg.meta.formatId}.json`);
		fs.writeFileSync(out, `${JSON.stringify(cfg, null, '\t')}\n`);
		// Self-check: what landed on disk names the submodule's gitlink.
		const written = JSON.parse(fs.readFileSync(out, 'utf8')).meta.showdownCommit;
		const gitlink = showdownGitlink(root);
		if (written !== gitlink) {
			throw new Error(`${out}: meta.showdownCommit ${written} != vendor gitlink ${gitlink}`);
		}
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

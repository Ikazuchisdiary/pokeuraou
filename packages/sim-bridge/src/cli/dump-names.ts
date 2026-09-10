/**
 * Writes configs/names/ja.json from Showdown's own Japanese text data.
 *
 *   node dist/cli/dump-names.js [locale ...]
 *
 * The names are not typed out here. Showdown ships `data/text/<locale>/` keyed by exactly
 * the ids this project already uses, so the mapping is generated from the pinned vendor
 * commit and moves with it -- which is the same rule the regulation dump follows, and the
 * reason a regulation rotation cannot leave a stale hand-written table behind.
 *
 * Two things are recorded rather than smoothed over:
 *
 * - an entry whose translation is `null` (Showdown marks these `NEEDS TRANSLATION`) is
 *   omitted, so the Python side can count what is missing instead of being handed an
 *   English string that looks translated;
 * - mega formes carry their own `name` ("メガフシギバナ") even where `forme` is null, so
 *   the species table is keyed by the full species id and needs no composition.
 */
import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';

const DEFAULT_LOCALES = ['ja'];

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

type NameTable = { [id: string]: string };

/** Pulls `name` out of a text table, dropping the entries Showdown left untranslated. */
function namesOf(table: { [id: string]: { name?: string | null } }): { names: NameTable; missing: string[] } {
	const names: NameTable = {};
	const missing: string[] = [];
	for (const [id, entry] of Object.entries(table ?? {})) {
		const name = entry?.name;
		if (typeof name === 'string' && name.length > 0) names[id] = name;
		else missing.push(id);
	}
	return { names, missing };
}

type SpeciesEntry = { name?: string; base?: string; forme?: string };

/**
 * Species names in the three pieces Showdown actually stores.
 *
 * A species with no formes has `name`. A species with formes puts the shared part on the
 * base entry's `baseSpecies` and the distinguishing part on each forme entry's `forme`.
 * All three are emitted; composing them needs the English forme as a fallback, which lives
 * on the regulation side.
 */
function speciesNames(table: {
	[id: string]: { name?: string | null; baseSpecies?: string | null; forme?: string | null };
}): { entries: { [id: string]: SpeciesEntry }; missing: string[] } {
	const entries: { [id: string]: SpeciesEntry } = {};
	const missing: string[] = [];
	for (const [id, entry] of Object.entries(table ?? {})) {
		const out: SpeciesEntry = {};
		if (typeof entry?.name === 'string' && entry.name) out.name = entry.name;
		if (typeof entry?.baseSpecies === 'string' && entry.baseSpecies) out.base = entry.baseSpecies;
		if (typeof entry?.forme === 'string' && entry.forme) out.forme = entry.forme;
		if (Object.keys(out).length) entries[id] = out;
		else missing.push(id);
	}
	return { entries, missing };
}

function plain(table: { [id: string]: string | null | undefined }): NameTable {
	const out: NameTable = {};
	for (const [id, value] of Object.entries(table ?? {})) {
		if (typeof value === 'string' && value.length > 0) out[id] = value;
	}
	return out;
}

function main() {
	const root = repoRoot();
	const commit = showdownCommit(root);
	const outDir = path.join(root, 'configs', 'names');
	fs.mkdirSync(outDir, { recursive: true });

	const locales = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_LOCALES;
	for (const locale of locales) {
		// The compiled dist is what the simulator itself loads, so read that rather than the
		// TypeScript source: same data, and no build step of our own to keep in sync.
		const dir = path.join(root, 'vendor', 'pokemon-showdown', 'dist', 'data', 'text', locale);
		if (!fs.existsSync(dir)) {
			console.error(`no text data for locale ${locale} at ${path.relative(root, dir)}`);
			process.exitCode = 1;
			continue;
		}

		/* eslint-disable @typescript-eslint/no-var-requires */
		const pokedex = require(path.join(dir, 'pokedex')).PokedexText;
		const moves = require(path.join(dir, 'moves')).MovesText;
		const items = require(path.join(dir, 'items')).ItemsText;
		const abilities = require(path.join(dir, 'abilities')).AbilitiesText;
		const names = require(path.join(dir, 'names'));
		/* eslint-enable @typescript-eslint/no-var-requires */

		const species = speciesNames(pokedex);
		const move = namesOf(moves);
		const item = namesOf(items);
		const ability = namesOf(abilities);

		const config = {
			meta: {
				locale,
				source: `vendor/pokemon-showdown/dist/data/text/${locale}`,
				showdownCommit: commit,
				generatedBy: 'packages/sim-bridge/src/cli/dump-names.ts',
				note:
					'Showdown が name: null と書いた項目は落としている。訳が無いことを ' +
					'英語名で埋めると「訳せている」と見分けがつかなくなるため、' +
					'Python 側で不足を数えて報告する。',
				untranslated: {
					speciesEntries: species.missing.length,
					moves: move.missing.length,
					items: item.missing.length,
					abilities: ability.missing.length,
				},
			},
			// {id: {name?, base?, forme?}} -- see speciesNames above.
			species: species.entries,
			moves: move.names,
			items: item.names,
			abilities: ability.names,
			natures: plain(names.NatureNames),
			types: plain(names.TypeNames),
			stats: plain(names.StatNames),
			statsShort: plain(names.StatShortNames),
			statuses: plain(names.StatusNames),
			genders: plain(names.GenderNames),
		};

		const out = path.join(outDir, `${locale}.json`);
		fs.writeFileSync(out, `${JSON.stringify(config, null, '\t')}\n`);
		console.log(
			`${locale}  ->  ${path.relative(root, out)}  ` +
				`(species ${Object.keys(config.species).length}, moves ${Object.keys(config.moves).length}, ` +
				`items ${Object.keys(config.items).length}, abilities ${Object.keys(config.abilities).length}, ` +
				`natures ${Object.keys(config.natures).length})`
		);
	}
}

main();

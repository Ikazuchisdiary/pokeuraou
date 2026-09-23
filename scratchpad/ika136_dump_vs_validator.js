// IKA-136: does the regulation dump's legality agree with Showdown's own team validator?
//
//   node scratchpad/ika136_dump_vs_validator.js [formatId] [standings.json.gz]
//
// For every species (non-mega, not battle-only), move, item and ability the format's dex
// knows, asks the validator (`checkSpecies` / `checkMove` / `checkItem` / `checkAbility`)
// and compares with what configs/regulations/<formatId>.json contains. Then lists the
// champions-mod entries that change a field but leave `isNonstandard` to be inherited from
// base, and, given a standings file, every move there that the dump lacks, with the
// validator's own reason and whether the champions learnset teaches it to that species.
//
// The Showdown build is read from vendor/pokemon-showdown/dist, or from $SHOWDOWN_ROOT
// (a worktree's submodule is not initialised, so point it at the main checkout's).
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const repo = path.resolve(__dirname, '..');
const PS = process.env.SHOWDOWN_ROOT || path.join(repo, 'vendor', 'pokemon-showdown');
const formatId = process.argv[2] || 'gen9championsvgc2026regmc';
const standingsPath = process.argv[3];

// The mod tables are read before the dex loads: loading merges base entries into them.
const modDir = path.join(PS, 'dist', 'data', 'mods', 'champions');
const modKeys = {};
for (const [kind, file, name] of [['moves', 'moves.js', 'Moves'], ['items', 'items.js', 'Items'], ['abilities', 'abilities.js', 'Abilities']]) {
	const table = require(path.join(modDir, file))[name];
	modKeys[kind] = Object.fromEntries(Object.entries(table).map(([id, e]) => [id, Object.keys(e)]));
}
const learnsets = require(path.join(modDir, 'learnsets.js')).Learnsets;

const { Dex, TeamValidator, toID } = require(path.join(PS, 'dist', 'sim'));
const format = Dex.formats.get(formatId);
const dex = Dex.forFormat(format);
const v = TeamValidator.get(format);
const dump = JSON.parse(fs.readFileSync(path.join(repo, 'configs', 'regulations', `${formatId}.json`), 'utf8'));

const probe = { name: 'probe', species: 'Pikachu', moves: [], item: '', ability: '' };
const inDump = {
	species: new Set(dump.species.filter(s => s.teamLegal).map(s => s.id)),
	moves: new Set(dump.moves.map(m => m.id)),
	items: new Set(dump.items.map(m => m.id)),
	abilities: new Set(dump.abilities.map(m => m.id)),
};
const kinds = {
	species: [dex.species.all().filter(s => !s.isMega && !s.battleOnly),
		s => v.checkSpecies({ ...probe, species: s.name }, s, s, {})],
	moves: [dex.moves.all().filter(m => !m.isZ && !m.isMax), m => v.checkMove(probe, m, {})],
	items: [dex.items.all(), i => v.checkItem(probe, i, {})],
	abilities: [dex.abilities.all(), a => v.checkAbility(probe, a, {})],
};
const agreement = {};
for (const [kind, [all, check]] of Object.entries(kinds)) {
	const out = { agree: 0, validatorOnly: [], dumpOnly: [] };
	for (const t of all) {
		if (!t.exists) continue;
		const problem = check(t);
		const ok = problem === null || problem === undefined;
		const d = inDump[kind].has(t.id);
		if (ok && !d) out.validatorOnly.push(`${t.id} (${t.isNonstandard})`);
		else if (!ok && d) out.dumpOnly.push(`${t.id}: ${problem}`);
		else out.agree++;
	}
	agreement[kind] = out;
}
console.log('dump vs validator', JSON.stringify(agreement, null, 1));

const learners = {};
for (const [sp, e] of Object.entries(learnsets)) {
	for (const mv of Object.keys(e.learnset || {})) (learners[mv] ??= []).push(sp);
}
const inherited = {};
for (const [kind, getter] of [['moves', id => dex.moves.get(id)], ['items', id => dex.items.get(id)], ['abilities', id => dex.abilities.get(id)]]) {
	inherited[kind] = [];
	for (const [id, keys] of Object.entries(modKeys[kind])) {
		if (keys.includes('isNonstandard')) continue;
		const t = getter(id);
		if (!t.isNonstandard) continue;
		const row = { id, isNonstandard: t.isNonstandard, changes: keys.filter(k => k !== 'inherit') };
		if (kind === 'moves') {
			row.learners = (learners[id] || []).map(sp => `${sp}${inDump.species.has(sp) ? ' (team-legal)' : ''}`);
		}
		inherited[kind].push(row);
	}
}
console.log('mod entries that change a field and inherit isNonstandard', JSON.stringify(inherited, null, 1));

if (standingsPath) {
	const data = JSON.parse(zlib.gunzipSync(fs.readFileSync(standingsPath)).toString('utf8'));
	const missing = {};
	for (const entry of Object.values(data.standings || {})) {
		for (const p of entry.team || []) {
			const sp = dex.species.get(p.code || p.name);
			for (const mv of p.moves || []) {
				const id = toID(mv.name);
				if (inDump.moves.has(id)) continue;
				const m = dex.moves.get(id);
				const key = `${id} on ${sp.id}`;
				missing[key] ??= {
					count: 0,
					isNonstandard: m.isNonstandard,
					validator: v.checkMove(probe, m, {}),
					learnset: v.checkCanLearn(m, sp, v.allSources(sp), { ...probe, species: sp.name }),
				};
				missing[key].count++;
			}
		}
	}
	console.log('standings moves the dump lacks', JSON.stringify(missing, null, 1));
}

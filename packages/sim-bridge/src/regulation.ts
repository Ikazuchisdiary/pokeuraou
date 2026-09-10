/**
 * Regulation config extraction.
 *
 * Everything the Python resolver needs about a Champions regulation is derived here from
 * the pinned Showdown dex and written to configs/regulations/<id>.json. The Python side
 * never reads Showdown -- it only reads that JSON. Nothing about a regulation (legal
 * species, mega-capable species, item pool) is hardcoded anywhere.
 */
import { Dex } from 'pokemon-showdown';
import type { Species } from 'pokemon-showdown/dist/sim/dex-species';

/** Neutral natures that exist in Showdown but NOT in the real Champions game. */
export const NATURES_NOT_IN_REAL_GAME = ['hardy', 'docile', 'bashful', 'quirky'] as const;

/**
 * Showdown rejects a 0-SP set only when the nature is Serious, and tells you to pick a
 * different neutral nature. Real Champions has Serious as its only neutral nature, so a
 * 0-SP set can only be expressed to Showdown by substituting a stat-identical neutral
 * nature. Hardy has the same (empty) plus/minus as Serious.
 */
export const ZERO_SP_NATURE_SUBSTITUTE = 'Hardy';

export interface RegulationMeta {
	formatId: string;
	formatName: string;
	mod: string;
	gameType: string;
	generation: number;
	/** Total stat points per Pokemon. */
	spLimit: number;
	/** Stat points per individual stat. */
	spPerStatMax: number;
	/** IVs are fixed in Champions; the validator requires maxed IVs. */
	fixedIv: number;
	level: number;
	/**
	 * Whether the format carries Showdown's `Level Clause Mod`. The champions mod picks
	 * its stat formula on this flag, not on the Pokemon's level: without it, stats use a
	 * level-50 closed form (`base + SP + 75` for HP, `base + SP + 20` otherwise);
	 * with it, the general level-dependent form. Champions VGC formats do not have it.
	 */
	usesLevelClauseMod: boolean;
	teamSize: number;
	pickedTeamSize: number;
	activePerSide: number;
	openTeamSheets: boolean;
	/** Open Team Sheets reveals nature in Champions; only the SP spread stays hidden. */
	teamSheetRevealsNature: boolean;
	itemClause: number | null;
	maxMoveCount: number;
	/** Terastallization is disabled in Champions (canTerastallize returns null). */
	teraEnabled: boolean;
	/** Mega Evolution: once per battle per side. */
	megaPerSide: number;
	showdownCommit: string;
	generatedAt: string;
}

export interface SpeciesEntry {
	id: string;
	name: string;
	num: number;
	baseSpecies: string;
	forme: string;
	types: string[];
	baseStats: Record<string, number>;
	abilities: string[];
	weightkg: number;
	genderRatio: { M: number; F: number };
	gender?: string;
	isMega: boolean;
	/** Item that triggers the forme change into this species, if any. */
	requiredItem?: string;
	battleOnly?: string[];
	/** Base forme this one reverts to / derives from. */
	changesFrom?: string;
	/** True if this forme can be brought to team preview (i.e. is team-legal). */
	teamLegal: boolean;
}

export interface MoveSecondary {
	chance?: number;
	status?: string;
	volatileStatus?: string;
	boosts?: Record<string, number>;
	self?: { boosts?: Record<string, number> };
}

export interface MoveEntry {
	id: string;
	name: string;
	num: number;
	type: string;
	category: 'Physical' | 'Special' | 'Status';
	basePower: number;
	/** `true` means "never misses". */
	accuracy: number | true;
	pp: number;
	priority: number;
	target: string;
	critRatio: number;
	flags: Record<string, number>;
	boosts?: Record<string, number>;
	self?: { boosts?: Record<string, number>; volatileStatus?: string; sideCondition?: string };
	status?: string;
	volatileStatus?: string;
	sideCondition?: string;
	slotCondition?: string;
	pseudoWeather?: string;
	terrain?: string;
	weather?: string;
	heal?: number[];
	drain?: number[];
	recoil?: number[];
	multihit?: number | number[];
	secondaries?: MoveSecondary[];
	damage?: number | 'level';
	ohko?: boolean | 'Ice';
	forceSwitch?: boolean;
	selfSwitch?: string | boolean;
	selfdestruct?: string | boolean;
	breaksProtect?: boolean;
	willCrit?: boolean;
	stallingMove?: boolean;
	thawsTarget?: boolean;
	smartTarget?: boolean;
	tracksTarget?: boolean;
	spreadModifier?: number;
	noDamageVariance?: boolean;
	ignoreAbility?: boolean;
	ignoreDefensive?: boolean;
	ignoreEvasion?: boolean;
	ignoreImmunity?: boolean | Record<string, boolean>;
	overrideOffensiveStat?: string;
	overrideDefensiveStat?: string;
	overrideOffensivePokemon?: string;
	overrideDefensivePokemon?: string;
	forceSTAB?: boolean;
	nonGhostTarget?: string;
	/** Stat changes applied to the user once, after the move succeeds. */
	selfBoost?: { boosts?: Record<string, number> };
	/**
	 * Move has JS event handlers in Showdown (onHit, basePowerCallback, ...) whose behaviour
	 * is NOT captured by the declarative fields above. The Python resolver must either
	 * implement it explicitly or report it as uncovered. This is what makes coverage
	 * measurable instead of guessed.
	 */
	hasCustomCode: boolean;
	customHooks: string[];
}

export interface ItemEntry {
	id: string;
	name: string;
	num: number;
	isBerry: boolean;
	naturalGift?: { basePower: number; type: string };
	fling?: { basePower: number; status?: string; volatileStatus?: string };
	/** { "Charizard": "Charizard-Mega-Y" } */
	megaStone?: Record<string, string>;
	itemUser?: string[];
	onPlate?: string;
	zMove?: boolean;
	hasCustomCode: boolean;
	customHooks: string[];
}

export interface AbilityEntry {
	id: string;
	name: string;
	num: number;
	rating: number;
	flags: Record<string, number>;
	suppressWeather?: boolean;
	hasCustomCode: boolean;
	customHooks: string[];
}

export interface NatureEntry {
	name: string;
	plus?: string;
	minus?: string;
	existsInRealGame: boolean;
}

export interface RegulationConfig {
	meta: RegulationMeta;
	natures: NatureEntry[];
	statIds: string[];
	types: string[];
	/** typechart[attackingType][defendingType]: 0 immune, 0.5 resist, 1 neutral, 2 super */
	typechart: Record<string, Record<string, number>>;
	species: SpeciesEntry[];
	moves: MoveEntry[];
	items: ItemEntry[];
	abilities: AbilityEntry[];
	/** itemId -> { fromSpeciesId: toSpeciesId } */
	megaMap: Record<string, Record<string, string>>;
	boostNames: string[];
}

/** Keys on dex entries that hold JS behaviour we cannot serialize. */
const CODE_KEY_RE = /^(on[A-Z]|.*Callback$)/;

function codeHooks(obj: object): string[] {
	const out: string[] = [];
	for (const key in obj) {
		if (!CODE_KEY_RE.test(key)) continue;
		if (typeof (obj as Record<string, unknown>)[key] === 'function') out.push(key);
	}
	// `condition` holds a nested effect with its own handlers (e.g. Protect, Tailwind).
	const cond = (obj as { condition?: object }).condition;
	if (cond) {
		for (const key of codeHooks(cond)) out.push(`condition.${key}`);
	}
	return out.sort();
}

/** Showdown stores type effectiveness as an index into [neutral, weak, resist, immune]. */
const TYPE_MOD: Record<number, number> = { 0: 1, 1: 2, 2: 0.5, 3: 0 };

const DECLARATIVE_MOVE_KEYS = [
	'boosts', 'status', 'volatileStatus', 'sideCondition', 'slotCondition', 'pseudoWeather',
	'terrain', 'weather', 'heal', 'drain', 'recoil', 'multihit', 'damage', 'ohko',
	'forceSwitch', 'selfSwitch', 'selfdestruct', 'breaksProtect', 'willCrit', 'stallingMove',
	'thawsTarget', 'smartTarget', 'tracksTarget', 'spreadModifier', 'noDamageVariance',
	'ignoreAbility', 'ignoreDefensive', 'ignoreEvasion', 'ignoreImmunity',
	'overrideOffensiveStat', 'overrideDefensiveStat', 'overrideOffensivePokemon',
	'overrideDefensivePokemon', 'forceSTAB', 'nonGhostTarget',
	// A separate field from `self`, applied once after the move succeeds. Only three moves
	// use it (Clanging Scales, Clangorous Soul, Scale Shot), and dropping it silently lost
	// their drawback entirely.
	'selfBoost',
] as const;

/** Copies the listed keys when they carry a meaningful value. */
function pickDeclarative(src: Record<string, unknown>, keys: readonly string[]): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const k of keys) {
		const v = src[k];
		if (v === undefined || v === null || v === false) continue;
		out[k] = typeof v === 'object' ? structuredClone(v) : v;
	}
	return out;
}

export function buildRegulationConfig(formatId: string, showdownCommit: string): RegulationConfig {
	const format = Dex.formats.get(formatId);
	if (!format.exists) throw new Error(`Unknown format: ${formatId}`);
	const dex = Dex.forFormat(format);
	const ruleTable = Dex.formats.getRuleTable(format);

	const activePerSide = format.gameType === 'doubles' ? 2 : format.gameType === 'triples' ? 3 : 1;

	const meta: RegulationMeta = {
		formatId: format.id,
		formatName: format.name,
		mod: format.mod,
		gameType: format.gameType,
		generation: dex.gen,
		spLimit: ruleTable.evLimit ?? 0,
		spPerStatMax: 32,
		fixedIv: 31,
		level: ruleTable.adjustLevel ?? ruleTable.maxLevel,
		usesLevelClauseMod: ruleTable.has('levelclausemod'),
		teamSize: ruleTable.maxTeamSize,
		pickedTeamSize: ruleTable.pickedTeamSize ?? ruleTable.maxTeamSize,
		activePerSide,
		openTeamSheets: ruleTable.has('openteamsheets') || ruleTable.has('forceopenteamsheets'),
		teamSheetRevealsNature: format.mod.startsWith('champions'),
		itemClause: ruleTable.valueRules.has('itemclause')
			? Number(ruleTable.valueRules.get('itemclause'))
			: null,
		maxMoveCount: ruleTable.maxMoveCount,
		teraEnabled: false,
		megaPerSide: 1,
		showdownCommit,
		generatedAt: new Date().toISOString(),
	};

	// ---- species ---------------------------------------------------------------
	// teamLegal = can be put on a team sheet. Mega formes are not team-legal but are
	// reachable in battle, so they are included with teamLegal: false.
	const teamLegalIds = new Set<string>();
	const reachable = new Map<string, Species>();

	for (const s of dex.species.all()) {
		if (!s.exists || s.num <= 0 || s.isNonstandard) continue;
		if (!s.isMega && !s.battleOnly && !ruleTable.isBannedSpecies(s)) teamLegalIds.add(s.id);
		reachable.set(s.id, s);
	}

	// Pull in every mega forme reachable from a legal stone.
	const megaMap: Record<string, Record<string, string>> = {};
	for (const item of dex.items.all()) {
		if (!item.exists || item.isNonstandard || !item.megaStone) continue;
		const entry: Record<string, string> = {};
		for (const [fromName, toName] of Object.entries(item.megaStone)) {
			const from = dex.species.get(fromName);
			const to = dex.species.get(toName);
			if (!from.exists || !to.exists) continue;
			entry[from.id] = to.id;
			reachable.set(to.id, to);
		}
		if (Object.keys(entry).length) megaMap[item.id] = entry;
	}

	const species: SpeciesEntry[] = [];
	for (const s of [...reachable.values()].sort((a, b) => a.id.localeCompare(b.id))) {
		const battleOnly = s.battleOnly
			? (Array.isArray(s.battleOnly) ? s.battleOnly : [s.battleOnly]).map(n => dex.species.get(n).id)
			: undefined;
		const changesFromName = Array.isArray(s.changesFrom) ? s.changesFrom[0] : s.changesFrom;
		species.push({
			id: s.id,
			name: s.name,
			num: s.num,
			baseSpecies: s.baseSpecies,
			forme: s.forme,
			types: [...s.types],
			baseStats: { ...s.baseStats },
			abilities: Object.values(s.abilities).filter(Boolean) as string[],
			weightkg: s.weightkg,
			genderRatio: { ...s.genderRatio },
			...(s.gender ? { gender: s.gender } : {}),
			isMega: !!s.isMega,
			...(s.requiredItem ? { requiredItem: dex.items.get(s.requiredItem).id } : {}),
			...(battleOnly ? { battleOnly } : {}),
			...(changesFromName ? { changesFrom: dex.species.get(changesFromName).id } : {}),
			teamLegal: teamLegalIds.has(s.id),
		});
	}

	// ---- moves -----------------------------------------------------------------
	const moves: MoveEntry[] = [];
	for (const m of dex.moves.all()) {
		if (!m.exists || m.isNonstandard || m.isZ || m.isMax) continue;
		const hooks = codeHooks(m as unknown as object);
		const rawSecondaries = m.secondaries ?? (m.secondary ? [m.secondary] : []);
		const secondaries: MoveSecondary[] = rawSecondaries.filter(Boolean).map(sec => ({
			...(sec.chance !== undefined ? { chance: sec.chance } : {}),
			...(sec.status ? { status: sec.status } : {}),
			...(sec.volatileStatus ? { volatileStatus: sec.volatileStatus } : {}),
			...(sec.boosts ? { boosts: { ...sec.boosts } as Record<string, number> } : {}),
			...(sec.self?.boosts ? { self: { boosts: { ...sec.self.boosts } as Record<string, number> } } : {}),
		}));
		const entry: MoveEntry = {
			id: m.id,
			name: m.name,
			num: m.num ?? 0,
			type: m.type,
			category: m.category,
			basePower: m.basePower,
			accuracy: m.accuracy,
			pp: m.pp,
			priority: m.priority,
			target: m.target,
			critRatio: m.critRatio ?? 1,
			flags: { ...m.flags } as Record<string, number>,
			hasCustomCode: hooks.length > 0,
			customHooks: hooks,
		};
		Object.assign(entry, pickDeclarative(m as unknown as Record<string, unknown>, DECLARATIVE_MOVE_KEYS));
		if (m.self) {
			entry.self = {
				...(m.self.boosts ? { boosts: { ...m.self.boosts } as Record<string, number> } : {}),
				...(m.self.volatileStatus ? { volatileStatus: m.self.volatileStatus } : {}),
				...(m.self.sideCondition ? { sideCondition: m.self.sideCondition } : {}),
			};
		}
		if (secondaries.length) entry.secondaries = secondaries;
		moves.push(entry);
	}

	// ---- items -----------------------------------------------------------------
	const items: ItemEntry[] = [];
	for (const it of dex.items.all()) {
		if (!it.exists || it.isNonstandard) continue;
		const hooks = codeHooks(it as unknown as object);
		items.push({
			id: it.id,
			name: it.name,
			num: it.num,
			isBerry: !!it.isBerry,
			...(it.naturalGift ? { naturalGift: { ...it.naturalGift } } : {}),
			...(it.fling ? { fling: { ...it.fling } } : {}),
			...(it.megaStone ? { megaStone: { ...it.megaStone } } : {}),
			...(it.itemUser ? { itemUser: [...it.itemUser] } : {}),
			...(it.onPlate ? { onPlate: it.onPlate } : {}),
			...(it.zMove ? { zMove: true } : {}),
			hasCustomCode: hooks.length > 0,
			customHooks: hooks,
		});
	}

	// ---- abilities -------------------------------------------------------------
	const abilities: AbilityEntry[] = [];
	for (const ab of dex.abilities.all()) {
		if (!ab.exists || ab.isNonstandard) continue;
		const hooks = codeHooks(ab as unknown as object);
		abilities.push({
			id: ab.id,
			name: ab.name,
			num: ab.num,
			rating: ab.rating,
			flags: { ...ab.flags } as Record<string, number>,
			...(ab.suppressWeather ? { suppressWeather: true } : {}),
			hasCustomCode: hooks.length > 0,
			customHooks: hooks,
		});
	}

	// ---- typechart -------------------------------------------------------------
	const types = dex.types.all().map(t => t.name).sort();
	const typechart: Record<string, Record<string, number>> = {};
	for (const atk of types) {
		typechart[atk] = {};
		for (const def of types) {
			const taken = dex.types.get(def).damageTaken[atk];
			typechart[atk][def] = TYPE_MOD[taken ?? 0];
		}
	}

	const natures: NatureEntry[] = dex.natures.all()
		.map(n => ({
			name: n.name,
			...(n.plus ? { plus: n.plus as string } : {}),
			...(n.minus ? { minus: n.minus as string } : {}),
			existsInRealGame: !(NATURES_NOT_IN_REAL_GAME as readonly string[]).includes(n.id),
		}))
		.sort((a, b) => a.name.localeCompare(b.name));

	return {
		meta,
		natures,
		statIds: [...Dex.stats.ids()],
		types,
		typechart,
		species,
		moves,
		items,
		abilities,
		megaMap,
		boostNames: ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'],
	};
}

/** Which effects carry Showdown code that the declarative dump does not capture. */
export function coverageSummary(cfg: RegulationConfig) {
	const count = <T extends { hasCustomCode: boolean }>(xs: T[]) => ({
		total: xs.length,
		withCode: xs.filter(x => x.hasCustomCode).length,
	});
	return {
		moves: count(cfg.moves),
		items: count(cfg.items),
		abilities: count(cfg.abilities),
		species: { total: cfg.species.length, teamLegal: cfg.species.filter(s => s.teamLegal).length },
	};
}

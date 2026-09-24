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
	/** Each hit after the first rolls its own accuracy (Triple Axel, Population Bomb). */
	multiaccuracy?: boolean;
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
	/**
	 * How long the effect this move applies lasts, as Showdown declares it. Emitted so the
	 * resolver reads durations from the dump instead of from a hand-written list -- four
	 * effects were silently permanent because they were missing from one.
	 *
	 * `durationCallback` says the fixed number is not the whole answer: `partiallytrapped`
	 * declares 5 and its callback returns `random(5, 7)`, i.e. 5 or 6. The consumer is
	 * expected to report the approximation rather than pretend the fixed value is exact.
	 */
	durations?: Record<string, DurationEntry>;
	customHooks: string[];
}

export interface ItemEntry {
	id: string;
	name: string;
	num: number;
	isBerry: boolean;
	/**
	 * A Choice item: it locks its holder into the first move it uses. Not emitted before,
	 * so the consumer had no way to know which items lock and simply never enforced it.
	 */
	isChoice?: boolean;
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
	/**
	 * Types immune to a named non-type effect, e.g. `prankster: ['Dark']`.
	 *
	 * `damageTaken` mixes attacking types with a handful of effect names, and these are the
	 * effect names. Prankster is the one that matters in practice: since gen 7 a
	 * Prankster-boosted status move simply fails against a foe Dark type.
	 */
	effectImmunities: Record<string, string[]>;
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

/**
 * The move fields that name an effect whose duration is worth reading.
 *
 * `weather` was missing, so a weather move carried no duration at all and the consumer used
 * a hardcoded 5 -- which is also where Heat Rock, Damp Rock, Smooth Rock and Icy Rock live.
 */
const EFFECT_NAMING_KEYS = [
	'volatileStatus', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain',
] as const;

/** How long an effect lasts, and what changes that. */
export interface DurationEntry {
	/** The number declared on the condition, if any. */
	duration?: number;
	/** True when a `durationCallback` exists at all. */
	durationCallback?: boolean;
	/** What the callback returns with nothing held: the ordinary case. */
	base?: number;
	/** itemId -> duration, for items that change the answer (Light Clay, Grip Claw). */
	byItem?: Record<string, number>;
	/** abilityId -> duration, for abilities that change it (Persistent). */
	byAbility?: Record<string, number>;
	/** True when the callback's answer moves with `this.random`, i.e. it is really rolled. */
	rolled?: boolean;
}

/**
 * Durations for every effect one move can apply, keyed by effect id.
 *
 * Reads the condition on the move itself, then the named conditions its `volatileStatus`,
 * `sideCondition`, `slotCondition` and `pseudoWeather` refer to. A named condition is
 * looked up through the dex so a mod's override is honoured -- Champions changes several.
 *
 * A `durationCallback` is *called* rather than merely noted, because "there is a callback"
 * was being read as "Showdown rolls this" and for thirteen of the fourteen effects that have
 * one it actually means "an item or an ability extends it". See `probeDurationCallback`.
 */
function collectDurations(
	dex: ReturnType<typeof Dex.forFormat>,
	move: Record<string, unknown>,
	items: readonly string[],
	abilities: readonly string[]
): Record<string, DurationEntry> {
	const out: Record<string, DurationEntry> = {};

	const record = (rawId: unknown, condition: unknown) => {
		if (typeof rawId !== 'string' || !rawId || !condition || typeof condition !== 'object') return;
		// Showdown's `weather` field is inconsistently cased -- 'sunnyday' but 'RainDance' and
		// 'Sandstorm' -- and the consumer normalises before looking the duration up, so the
		// key has to be normalised here or rain and sand silently miss their entry.
		const id = rawId.toLowerCase().replace(/[^a-z0-9]/g, '');
		const c = condition as Record<string, unknown>;
		const entry: DurationEntry = {};
		if (typeof c.duration === 'number') entry.duration = c.duration;
		if (typeof c.durationCallback === 'function') {
			entry.durationCallback = true;
			Object.assign(entry, probeDurationCallback(c.durationCallback as CallbackFn, items, abilities));
		}
		if (Object.keys(entry).length) out[id] = entry;
	};

	// The move's own condition applies to whichever effect the move names...
	const own = move.condition;
	for (const key of EFFECT_NAMING_KEYS) {
		record(move[key], own);
	}
	// ...and to a volatile named after the move itself, which is how a move that adds its
	// own condition from a handler does it. Throat Chop's duration-2 sound lock is only
	// reachable this way: it names no effect and calls `addVolatile('throatchop')` from a
	// 100%-chance secondary, so without this the volatile would be added with no duration
	// and never expire -- the bug class that made four effects permanent.
	record(move.id, own);
	// ...and the named condition, which is where conditions.ts keeps most of them.
	for (const key of EFFECT_NAMING_KEYS) {
		const id = move[key];
		if (typeof id !== 'string' || !id) continue;
		try {
			record(id, dex.conditions.get(id));
		} catch {
			// A condition the dex does not know is not a duration we can report.
		}
	}
	return out;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type CallbackFn = (this: any, target?: any, source?: any, effect?: any) => unknown;

/**
 * Calls a `durationCallback` under controlled conditions and reports what moves the answer.
 *
 * The callbacks in play read only three things: `source.hasItem(id)`, `source.hasAbility(id)`
 * and `this.random(a, b)`. So a stub source that admits to exactly one item or ability, and
 * a stub `this` whose `random` returns a chosen end of its range, is enough to read the whole
 * function off without hardcoding any of it. `this.add` is a no-op because two of them log.
 *
 * Anything that throws is skipped rather than guessed at: a callback needing more context
 * than this reports only its declared `duration`, which is the state before this existed.
 */
function probeDurationCallback(
	callback: CallbackFn,
	items: readonly string[],
	abilities: readonly string[]
): Partial<DurationEntry> {
	const source = (item?: string, ability?: string) => ({
		hasItem: (id: unknown) => id === item,
		hasAbility: (id: unknown) => id === ability,
		getItem: () => ({ id: item ?? '' }),
		volatiles: {},
		side: { sideConditions: {} },
	});
	const battle = (low: boolean) => ({
		random: (from?: number, to?: number) => {
			if (from === undefined) return low ? 0 : 0.999999;
			if (to === undefined) return low ? 0 : from - 1;
			return low ? from : to - 1;
		},
		add: () => {},
		debug: () => {},
		effectState: {},
		hint: () => {},
	});

	// The signatures differ by effect kind: a side condition's callback is
	// `(target, source, effect)` while a terrain's or a room's is `(source, effect)`. Both
	// orders are tried and the one that reacts to the stub is the one that matters --
	// guessing wrong is how Terrain Extender and Persistent went missing on the first pass.
	const call = (item?: string, ability?: string, low = true): number[] => {
		const held = source(item, ability);
		const out: number[] = [];
		const orders: any[][] = [[source(), held, {}], [held, {}]];
		for (const args of orders) {
			try {
				const value = callback.apply(battle(low), args as [any, any, any]);
				if (typeof value === 'number') out.push(value);
			} catch {
				// A callback needing more context than this reports only its declared
				// duration, which is the state before this existed.
			}
		}
		return out;
	};

	const baseline = call();
	if (!baseline.length) return {};
	const base = baseline[0];
	const out: Partial<DurationEntry> = { base };
	const bases = new Set(baseline);

	// Rolled, not extended: the answer moves when `random` returns the other end.
	if (call(undefined, undefined, false).some(v => !bases.has(v))) out.rolled = true;

	const differing = (values: number[]): number | undefined =>
		values.find(v => !bases.has(v));

	const byItem: Record<string, number> = {};
	for (const id of items) {
		const value = differing(call(id));
		if (value !== undefined) byItem[id] = value;
	}
	if (Object.keys(byItem).length) out.byItem = byItem;

	const byAbility: Record<string, number> = {};
	for (const id of abilities) {
		const value = differing(call(undefined, id));
		if (value !== undefined) byAbility[id] = value;
	}
	if (Object.keys(byAbility).length) out.byAbility = byAbility;

	return out;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

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
	// Triple Axel and Population Bomb roll accuracy again before each later hit
	// (`hitStepMoveHitLoop`). Left out until IKA-235, which hid the port's gate for it: the
	// port read the first roll as all three hits.
	'multiaccuracy',
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
	// The candidate ids the duration callbacks are probed against. Every item and ability
	// the dex knows, so a new Light Clay in a later regulation is found without editing this.
	const itemIds = dex.items.all().map(i => i.id);
	const abilityIds = dex.abilities.all().map(a => a.id);

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
		const durations = collectDurations(
			dex, m as unknown as Record<string, unknown>, itemIds, abilityIds
		);
		if (Object.keys(durations).length) entry.durations = durations;
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
			...(it.isChoice ? { isChoice: true } : {}),
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

	// The keys of `damageTaken` that are not attacking types: named effects a type can be
	// immune to. Only value 3 (immune) is meaningful for these -- there is no "resists
	// Prankster".
	const effectImmunities: Record<string, string[]> = {};
	const typeNames = new Set(types);
	for (const def of types) {
		for (const [key, taken] of Object.entries(dex.types.get(def).damageTaken)) {
			if (typeNames.has(key) || taken !== 3) continue;
			(effectImmunities[key] ??= []).push(def);
		}
	}
	for (const list of Object.values(effectImmunities)) list.sort();

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
		effectImmunities,
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

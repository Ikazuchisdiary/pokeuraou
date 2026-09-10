/**
 * Canonical position representation.
 *
 * This is THE interface between the Showdown-backed TypeScript layer and the Python
 * solver. Both sides must read and write exactly this shape. The Python resolver works
 * from it directly and never touches Showdown; the oracle produces it so trajectories
 * can be compared turn by turn.
 *
 * Design notes:
 * - Stat spreads are always in SP space (0..32 per stat, <= 66 total). Showdown's
 *   champions mod stores SP in `set.evs` natively, so there is no conversion anywhere.
 * - `sp` and `nature`: Open Team Sheets in Champions reveals nature but not the SP
 *   spread. `sp: null` means "hidden" and is what the belief layer fills in.
 * - Volatiles are split into `volatiles` (fields the resolver models) and
 *   `unmodelledVolatiles` (everything else, present so divergence is detectable rather
 *   than silently ignored).
 */

export interface EffectSnapshot {
	id: string;
	duration?: number;
	/** Layer/stack count for Spikes, Toxic Spikes, Stockpile, ... */
	layers?: number;
	/** Generic counter used by Fury Cutter, Rollout, Protect stall, ... */
	counter?: number;
	/** Slot of the Pokemon that applied this, when the resolver needs it. */
	sourceSlot?: string;
	/** Move id, for Encore / Disable / Choice lock / Taunt sources. */
	move?: string;
	/** Any remaining EffectState fields, JSON-safe, for divergence reporting. */
	extra?: Record<string, string | number | boolean>;
}

export interface MoveSlotSnapshot {
	id: string;
	pp: number;
	maxpp: number;
	disabled: boolean;
	/**
	 * Whether this move has been used since the Pokemon came in.
	 *
	 * Last Resort fails until every *other* move has been used, and the flag resets on
	 * switch-in -- so spent PP is not a substitute for it.
	 */
	used: boolean;
}

export interface PokemonSnapshot {
	/** Position in the side's party (stable across the battle). */
	slot: number;
	/** Current species id (mega forme after evolving). */
	species: string;
	/** Species id the set was built with. */
	baseSpecies: string;
	/**
	 * Types as they are right now, which are not always the species' types: Protean and
	 * Libero retype their user once per switch-in, and Mimicry, Soak, Reflect Type and
	 * Forest's Curse all change them mid-battle. STAB depends on these, so they are
	 * recorded rather than derived.
	 */
	types: string[];
	level: number;
	gender: string;
	ability: string;
	/**
	 * The ability's own state, JSON-safe fields only.
	 *
	 * Some abilities keep their "has this fired yet" flag or their counter here rather
	 * than in a volatile -- Protean and Libero guard on `effectState.protean`, so without
	 * this a Pokemon that has already been retyped looks as though it is about to be.
	 */
	abilityState: Record<string, string | number | boolean>;
	/** Held item id, or null once consumed/knocked off. */
	item: string | null;
	/** Item the Pokemon started with, so consumption is visible. */
	baseItem: string | null;
	/** Nature name. Public information under Champions Open Team Sheets. */
	nature: string;
	/** SP spread, or null when hidden (opponent). */
	sp: Record<string, number> | null;
	moves: MoveSlotSnapshot[];
	hp: number;
	maxhp: number;
	fainted: boolean;
	status: string | null;
	statusDuration?: number;
	/** Toxic counter / Sleep turns, when applicable. */
	statusCounter?: number;
	boosts: Record<string, number>;
	volatiles: EffectSnapshot[];
	unmodelledVolatiles: string[];
	isMega: boolean;
	/** Index into the side's `active` array, or null when on the bench. */
	activeIndex: number | null;
	/** Trapped by the opponent (cannot switch). */
	trapped: boolean;
	/** Switched in this turn (relevant to Fake Out, First Impression). */
	newlySwitched: boolean;
	/** How many times this Pokemon has been hit, which is Rage Fist's base power. */
	timesAttacked: number;
	/** Last move used, needed for Torment / Encore / Choice lock reasoning. */
	lastMove: string | null;
	/**
	 * Whether last turn's move succeeded.
	 *
	 * Stomping Tantrum and Temper Flare double their base power when it is false, so this
	 * is not derivable from anything else in the position -- the turn it refers to is
	 * already over.
	 */
	moveLastTurnFailed: boolean;
	/** Move the Pokemon is locked into by a choice item or Encore, if any. */
	lockedMove: string | null;
	/**
	 * Final stats as the simulator holds them.
	 *
	 * Normally redundant -- they follow from `sp` and the species -- but not always:
	 * Transform copies the target's stats except HP, so a transformed Pokemon's stats do
	 * not follow from its own spread. Present whenever the producer knows them; the solver
	 * uses them for its own side and for a transformed Pokemon, and falls back to the
	 * belief layer's candidate spreads for a hidden opponent.
	 */
	statsOverride?: Record<string, number>;
	/** Set while Transform is active, so the override is known to be authoritative. */
	transformed: boolean;
}

export interface SideSnapshot {
	id: string;
	name: string;
	/** Party slots currently on the field, in slot order; null for an empty slot. */
	active: (number | null)[];
	pokemon: PokemonSnapshot[];
	sideConditions: EffectSnapshot[];
	/** Per active slot (Healing Wish, Wish, Future Sight). */
	slotConditions: EffectSnapshot[][];
	/** Mega Evolution is a once-per-battle side resource. */
	megaUsed: boolean;
	/** Which party slots hold a mega stone that still matches their species. */
	megaCapableSlots: number[];
}

export interface FieldSnapshot {
	weather: string | null;
	weatherDuration?: number;
	terrain: string | null;
	terrainDuration?: number;
	pseudoWeather: EffectSnapshot[];
}

export interface Position {
	format: string;
	turn: number;
	field: FieldSnapshot;
	sides: SideSnapshot[];
	/** '' while the battle is running, 'move' | 'switch' | 'teampreview' when a choice is due. */
	requestState: string;
	ended: boolean;
	winner: string | null;
}

/** EffectState keys that are structural rather than mechanically meaningful. */
const EFFECT_STATE_IGNORED = new Set([
	'id', 'effectOrder', 'target', 'source', 'sourceEffect', 'isBreakable', 'duration',
	'sourceSlot', 'layers', 'counter', 'move', 'sourceMove',
]);

/**
 * Volatiles the Python resolver models. Anything else on a Pokemon lands in
 * `unmodelledVolatiles` so the diff test can attribute divergence to it.
 */
export const MODELLED_VOLATILES = new Set([
	'protect', 'endure', 'banefulbunker', 'burningbulwark', 'spikyshield', 'kingsshield',
	'obstruct', 'silktrap', 'maxguard', 'craftyshield', 'matblock', 'quickguard', 'wideguard',
	'substitute', 'flinch', 'confusion', 'leechseed', 'taunt', 'encore', 'disable', 'torment',
	'attract', 'yawn', 'partiallytrapped', 'trapped', 'followme', 'ragepowder', 'spotlight',
	'helpinghand', 'choicelock', 'mustrecharge', 'lockedmove', 'twoturnmove', 'destinybond',
	'focusenergy', 'laserfocus', 'charge', 'aquaring', 'ingrain', 'magnetrise', 'telekinesis',
	'roost', 'stall', 'sparklingaria', 'saltcure', 'curse', 'nightmare', 'perishsong',
	'imprison', 'miracleeye', 'foresight', 'gastroacid', 'healblock', 'embargo', 'powertrick',
	'defensecurl', 'minimize', 'smackdown', 'octolock', 'syrupbomb', 'glaiverush', 'dragoncheer',
	// Marks that Protean/Libero has already fired this switch-in, so it will not fire again.
	'protean',
	// Unburden's doubled Speed lives on a volatile added when the item is lost, so the
	// ability alone does not say whether it is active.
	'unburden',
]);

function snapshotEffect(state: Record<string, unknown>): EffectSnapshot {
	const out: EffectSnapshot = { id: String(state.id ?? '') };
	if (typeof state.duration === 'number') out.duration = state.duration;
	if (typeof state.layers === 'number') out.layers = state.layers;
	if (typeof state.counter === 'number') out.counter = state.counter;
	if (typeof state.sourceSlot === 'string') out.sourceSlot = state.sourceSlot;
	const move = state.move ?? state.sourceMove;
	if (typeof move === 'string') out.move = move;
	else if (move && typeof move === 'object' && 'id' in move) out.move = String((move as { id: unknown }).id);

	const extra: Record<string, string | number | boolean> = {};
	for (const [k, v] of Object.entries(state)) {
		if (EFFECT_STATE_IGNORED.has(k)) continue;
		if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') extra[k] = v;
	}
	if (Object.keys(extra).length) out.extra = extra;
	return out;
}

/** Keeps only the JSON-safe scalars from an effect state, dropping object references. */
function jsonSafe(state: Record<string, unknown> | undefined): Record<string, string | number | boolean> {
	const out: Record<string, string | number | boolean> = {};
	for (const [k, v] of Object.entries(state ?? {})) {
		if (k === 'id' || k === 'effectOrder' || k === 'target' || k === 'source') continue;
		if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') out[k] = v;
	}
	return out;
}

function effectList(table: { [id: string]: Record<string, unknown> }): EffectSnapshot[] {
	return Object.values(table)
		.map(snapshotEffect)
		.filter(e => e.id)
		.sort((a, b) => a.id.localeCompare(b.id));
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyBattle = any;

/**
 * Reads a Position out of a live Showdown Battle.
 *
 * `megaUsedBySide` is passed in because Showdown has no single flag for it: after
 * `runMegaEvo` every ally's `canMegaEvo` is cleared, and champions keeps mega formes
 * through fainting. The oracle session tracks it from the `-mega` protocol message,
 * which is exactly what a client sees.
 */
export function positionFromBattle(battle: AnyBattle, megaUsedBySide: boolean[]): Position {
	const field = battle.field;
	const fieldSnap: FieldSnapshot = {
		weather: field.weather || null,
		terrain: field.terrain || null,
		pseudoWeather: effectList(field.pseudoWeather),
	};
	if (field.weather && typeof field.weatherState?.duration === 'number') {
		fieldSnap.weatherDuration = field.weatherState.duration;
	}
	if (field.terrain && typeof field.terrainState?.duration === 'number') {
		fieldSnap.terrainDuration = field.terrainState.duration;
	}

	const sides: SideSnapshot[] = battle.sides.map((side: AnyBattle, sideIndex: number) => {
		const slotOf = new Map<AnyBattle, number>();
		side.pokemon.forEach((p: AnyBattle, i: number) => slotOf.set(p, i));

		const pokemon: PokemonSnapshot[] = side.pokemon.map((p: AnyBattle, slot: number) => {
			const modelled: EffectSnapshot[] = [];
			const unmodelled: string[] = [];
			for (const [id, state] of Object.entries(p.volatiles as Record<string, Record<string, unknown>>)) {
				if (MODELLED_VOLATILES.has(id)) modelled.push(snapshotEffect({ ...state, id }));
				else unmodelled.push(id);
			}
			modelled.sort((a, b) => a.id.localeCompare(b.id));
			unmodelled.sort();

			const activeIndex = side.active.indexOf(p);
			const boosts: Record<string, number> = {};
			for (const [k, v] of Object.entries(p.boosts as Record<string, number>)) {
				if (v) boosts[k] = v;
			}

			const snap: PokemonSnapshot = {
				slot,
				species: p.species.id,
				baseSpecies: battle.dex.species.get(p.set.species).id,
				types: [...p.getTypes()],
				level: p.level,
				gender: p.gender || 'N',
				ability: p.ability,
				abilityState: jsonSafe(p.abilityState),
				item: p.item || null,
				baseItem: battle.dex.items.get(p.set.item).id || null,
				nature: p.set.nature || 'Serious',
				sp: { ...p.set.evs },
				moves: p.moveSlots.map((m: AnyBattle) => ({
					id: m.id,
					pp: m.pp,
					maxpp: m.maxpp,
					disabled: !!m.disabled,
					used: !!m.used,
				})),
				hp: p.hp,
				maxhp: p.maxhp,
				fainted: !!p.fainted,
				status: p.status || null,
				boosts,
				volatiles: modelled,
				unmodelledVolatiles: unmodelled,
				isMega: !!p.species.isMega,
				activeIndex: activeIndex >= 0 ? activeIndex : null,
				trapped: !!p.trapped,
				newlySwitched: !!p.newlySwitched,
				timesAttacked: p.timesAttacked ?? 0,
				lastMove: p.lastMove?.id ?? null,
				moveLastTurnFailed: p.moveLastTurnResult === false,
				// A Pokemon part-way through a charging move has no choice: Showdown fires the
				// stored move with `[from] lockedmove`. The two-turn lock therefore takes
				// precedence over a choice item's or Encore's.
				lockedMove: (p.volatiles.twoturnmove?.move
					?? p.volatiles.choicelock?.move
					?? p.volatiles.encore?.move
					?? null) as string | null,
				// storedStats holds atk..spe only; HP lives in maxhp, which Transform leaves alone.
				statsOverride: { hp: p.maxhp, ...p.storedStats },
				transformed: !!p.transformed,
			};
			if (typeof p.statusState?.duration === 'number') snap.statusDuration = p.statusState.duration;
			if (typeof p.statusState?.time === 'number') snap.statusCounter = p.statusState.time;
			else if (typeof p.statusState?.stage === 'number') snap.statusCounter = p.statusState.stage;
			return snap;
		});

		const megaCapableSlots: number[] = [];
		side.pokemon.forEach((p: AnyBattle, slot: number) => {
			const item = p.getItem();
			if (item?.megaStone?.[p.baseSpecies.name]) megaCapableSlots.push(slot);
		});

		return {
			id: side.id,
			name: side.name,
			active: side.active.map((p: AnyBattle) => (p ? (slotOf.get(p) ?? null) : null)),
			pokemon,
			sideConditions: effectList(side.sideConditions),
			slotConditions: side.slotConditions.map((t: AnyBattle) => effectList(t)),
			megaUsed: !!megaUsedBySide[sideIndex],
			megaCapableSlots,
		};
	});

	return {
		format: battle.format.id,
		turn: battle.turn,
		field: fieldSnap,
		sides,
		requestState: battle.requestState || '',
		ended: !!battle.ended,
		winner: battle.winner ?? null,
	};
}

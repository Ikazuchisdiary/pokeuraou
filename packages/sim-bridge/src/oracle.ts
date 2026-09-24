/**
 * Deterministic Showdown oracle.
 *
 * The Python resolver is checked against this. Randomness is not merely seeded but
 * *forced*: every source of chance in a turn is pinned to a chosen outcome, so a given
 * (position, action pair, randomness policy) has exactly one correct successor state and
 * the comparison is an equality test rather than a distribution test.
 *
 * Showdown's chance sources, all of which route through Battle methods we can override:
 * - damage roll:  Battle#randomizer  -> trunc(trunc(d * (100 - random(16))) / 100)
 * - accuracy:     Battle#randomChance(accuracy, 100)
 * - crit:         Battle#randomChance(1, critMult[ratio]) with critMult in {24,8,2,1}
 * - secondaries:  Battle#random(100)
 * - multi-hit:    Battle#random(2) / random(a, b) / random(7)
 * - speed ties:   Battle#prng.shuffle(list, start, end)
 *
 * `randomChance(1, 100)` would be ambiguous between a 1%-accuracy move and a crit, but
 * crit denominators are never 100, so keying on (numerator, denominator) is exact.
 */
import { Battle, Dex, Teams, TeamValidator } from 'pokemon-showdown';
import { positionFromBattle, type Position } from './position';
import { ZERO_SP_NATURE_SUBSTITUTE } from './regulation';

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyBattle = any;

export interface RandomnessPolicy {
	/** 0 = maximum damage (100%), 15 = minimum (85%). */
	damageRoll: number;
	accuracy: 'hit' | 'miss';
	crit: boolean;
	/** Whether secondary effects trigger (moves with chance < 100). */
	secondary: boolean;
	multihit: 'min' | 'max';
	/** 'keep' leaves the tied group in queue order; 'reverse' flips it. */
	speedTie: 'keep' | 'reverse';
	/**
	 * What `sample(values)` answers: `values[0]` or the last one. 'last' reaches the other
	 * foe of a `randomNormal` move (`side.randomFoe`), which 'first' never does (IKA-178).
	 */
	sample: 'first' | 'last';
	/**
	 * Answers for the accuracy rolls (`randomChance(n, 100)`) of each step, in the order
	 * they are asked; past its end, `accuracy` answers. Counted from the start of every
	 * step, so a later hit of Triple Axel can miss after the first hit landed (IKA-235).
	 */
	accuracyScript?: ('hit' | 'miss')[];
}

export const DEFAULT_POLICY: RandomnessPolicy = {
	damageRoll: 0,
	accuracy: 'hit',
	crit: false,
	secondary: false,
	multihit: 'min',
	speedTie: 'keep',
	sample: 'first',
};

/** Crit denominators used by Showdown, i.e. the values that identify a crit roll. */
const CRIT_DENOMINATORS = new Set([24, 8, 2, 1]);

/**
 * One roll Showdown asked the policy for.
 *
 * `chance` is `randomChance(numerator, denominator)`, `random` is `random(from, to)`, and
 * `sample` is `sample(values)`. The policy pins the answer, so these arguments are the only
 * evidence of what odds the simulator would have used -- which is what makes a hardcoded
 * probability on our side checkable rather than asserted.
 */
export type RollRecord =
	| { kind: 'chance'; numerator: number; denominator: number }
	| { kind: 'random'; from?: number; to?: number }
	| { kind: 'sample'; values: string[] };

function installPolicy(battle: AnyBattle, policy: RandomnessPolicy) {
	const trunc: (n: number, bits?: number) => number = battle.trunc.bind(battle);
	const roll = Math.max(0, Math.min(15, policy.damageRoll | 0));

	battle.randomizer = (baseDamage: number) => trunc(trunc(baseDamage * (100 - roll)) / 100);

	// Every roll Showdown asks for, in order, so the consumer can check a pinned
	// probability against the odds the simulator actually used. The policy answers these
	// with a fixed value, so the outcome says nothing -- the arguments say everything. Read
	// and cleared per step; see `rolls` in the step result.
	const rolls: RollRecord[] = [];
	battle._pokeuraouRolls = rolls;

	battle.randomChance = (numerator: number, denominator: number): boolean => {
		rolls.push({ kind: 'chance', numerator, denominator });
		if (numerator === 1 && CRIT_DENOMINATORS.has(denominator)) {
			// A guaranteed crit (denominator 1) stays guaranteed.
			return denominator === 1 ? true : policy.crit;
		}
		if (denominator === 100) {
			// `rolls` is cleared at every step: this is the step's n-th accuracy roll.
			const asked = rolls.filter(r => r.kind === 'chance' && r.denominator === 100).length;
			return (policy.accuracyScript?.[asked - 1] ?? policy.accuracy) === 'hit';
		}
		// Anything else (Gen 2/3 Quick Claw, ability procs) follows the secondary policy.
		return policy.secondary;
	};

	battle.random = (from?: number, to?: number): number => {
		rolls.push({ kind: 'random', from, to });
		if (from === undefined) return policy.secondary ? 0 : 0.999999;
		if (to === undefined) {
			// random(n) -> [0, n)
			if (from === 100) return policy.secondary ? 0 : 99; // secondary roll
			return policy.multihit === 'min' ? 0 : from - 1;
		}
		return policy.multihit === 'min' ? from : to - 1;
	};

	const shuffle = (list: unknown[], start = 0, end = list.length) => {
		if (policy.speedTie === 'reverse') {
			const seg = list.slice(start, end).reverse();
			for (let i = 0; i < seg.length; i++) list[start + i] = seg[i];
		}
	};
	battle.prng.shuffle = shuffle;
	// Recorded because the champions mod expresses the Champions sleep rule as
	// `sample([2, 3, 3])` -- one turn of sleep a third of the time and two the rest -- and
	// no denominator would reveal that distribution.
	battle.sample = <T>(items: readonly T[]): T => {
		rolls.push({ kind: 'sample', values: items.map(v => String(v)) });
		return policy.sample === 'last' ? items[items.length - 1] : items[0];
	};
}

export interface TeamSet {
	species: string;
	item?: string;
	ability: string;
	nature: string;
	/** SP spread, 0..32 per stat, <= 66 total. */
	sp: Partial<Record<string, number>>;
	moves: string[];
	level?: number;
	gender?: string;
	name?: string;
}

const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'] as const;

/**
 * Converts our SP-space set to a Showdown PokemonSet.
 *
 * No numeric conversion happens: the champions mod reads `set.evs` as stat points
 * directly. The one adjustment is the 0-SP case, which Showdown rejects when the nature
 * is Serious and suggests fixing by picking another neutral nature; real Champions only
 * has Serious, so we substitute the stat-identical Hardy at the boundary.
 */
export function toShowdownSet(set: TeamSet, level: number): Record<string, unknown> {
	const evs: Record<string, number> = {};
	let total = 0;
	for (const stat of STAT_IDS) {
		const v = set.sp[stat] ?? 0;
		evs[stat] = v;
		total += v;
	}
	const ivs: Record<string, number> = {};
	for (const stat of STAT_IDS) ivs[stat] = 31;
	const nature = total === 0 && Dex.natures.get(set.nature).id === 'serious'
		? ZERO_SP_NATURE_SUBSTITUTE
		: set.nature;
	return {
		name: set.name ?? set.species,
		species: set.species,
		item: set.item ?? '',
		ability: set.ability,
		moves: [...set.moves],
		nature,
		evs,
		ivs,
		level: set.level ?? level,
		...(set.gender ? { gender: set.gender } : {}),
	};
}

export interface CreateRequest {
	format: string;
	teams: [TeamSet[], TeamSet[]];
	seed?: [number, number, number, number];
	policy?: Partial<RandomnessPolicy>;
}

export interface StepResult {
	position: Position;
	/** Per-side choice request, as Showdown would send it to a client. */
	requests: (Record<string, unknown> | null)[];
	/** Protocol lines produced since the previous step. */
	log: string[];
	choiceErrors: string[];
	/**
	 * Rolls Showdown asked the policy for since the previous step.
	 *
	 * The policy pins the answers, so the outcome of a roll says nothing -- but the odds it
	 * was asked with are the simulator's own, mod overrides included. That is what makes a
	 * probability hardcoded on the consumer's side checkable instead of asserted.
	 */
	rolls: RollRecord[];
}

export class OracleSession {
	readonly battle: AnyBattle;
	readonly policy: RandomnessPolicy;
	private megaUsed = [false, false];
	private logPos = 0;
	private choiceErrors: string[] = [];

	constructor(req: CreateRequest) {
		const format = Dex.formats.get(req.format);
		if (!format.exists) throw new Error(`Unknown format: ${req.format}`);
		this.policy = { ...DEFAULT_POLICY, ...(req.policy ?? {}) };
		const level = Dex.formats.getRuleTable(format).adjustLevel ?? 50;
		const teams = req.teams.map(t => t.map(s => toShowdownSet(s, level)));

		this.battle = new Battle({
			formatid: format.id as never,
			seed: (req.seed ? `${req.seed.join(',')}` : '1,2,3,4') as never,
			strictChoices: false,
			// Keep the log from growing unbounded: Showdown throws "Infinite loop" once
			// log.length - sentLogPos exceeds 1000.
			send: () => {},
			p1: { name: 'p1', team: teams[0] as never },
			p2: { name: 'p2', team: teams[1] as never },
		});
		installPolicy(this.battle, this.policy);
	}

	private drainLog(): string[] {
		const lines = this.battle.log.slice(this.logPos) as string[];
		this.logPos = this.battle.log.length;
		for (const line of lines) {
			// `|-mega|p1a: Charizard|Charizard|Charizardite Y`
			if (!line.startsWith('|-mega|') && !line.startsWith('|-burst|')) continue;
			const who = line.split('|')[2] ?? '';
			const sideIndex = who.startsWith('p2') ? 1 : 0;
			this.megaUsed[sideIndex] = true;
		}
		return lines;
	}

	result(): StepResult {
		const log = this.drainLog();
		return {
			position: positionFromBattle(this.battle, this.megaUsed),
			requests: this.battle.sides.map((s: AnyBattle) => s.activeRequest ?? null),
			log,
			choiceErrors: this.choiceErrors.splice(0),
			rolls: (this.battle._pokeuraouRolls as RollRecord[]).splice(0),
		};
	}

	/** Applies one choice per side and advances the battle. */
	step(choices: (string | null)[]): StepResult {
		this.battle.sides.forEach((side: AnyBattle, i: number) => {
			const choice = choices[i];
			if (!choice) return;
			if (!side.activeRequest || side.activeRequest.wait) return;
			side.choice.error = '';
			if (!this.battle.choose(side.id, choice)) {
				this.choiceErrors.push(`${side.id}: ${side.choice.error || `rejected "${choice}"`}`);
			}
		});
		// Showdown only flushes when the log grows past 500 lines; keep our own bound so
		// the "Infinite loop" guard never fires in long analysis sessions.
		if (this.battle.log.length - this.battle.sentLogPos > 400) {
			this.battle.sentLogPos = this.battle.log.length;
		}
		return this.result();
	}

	/**
	 * Reports which of `candidates` Showdown accepts for one side at the current
	 * position, without disturbing this session. Used to test our action enumerator.
	 *
	 * Each candidate runs on a fresh clone. Rolling back in place is not reliable:
	 * `Side#clearChoice` leaves request state behind, and `undoChoice` refuses once a
	 * choice is marked `cantUndo` (trapping/disabling effects, deliberately, so that
	 * undo cannot leak information).
	 */
	probeChoices(sideIndex: number, candidates: string[]): { choice: string; ok: boolean; error?: string }[] {
		const snapshot = this.snapshot();
		return candidates.map(choice => {
			const clone = OracleSession.load(snapshot, this.policy);
			const side = clone.sides[sideIndex];
			side.choice.error = '';
			const ok = clone.choose(side.id, choice);
			return ok ? { choice, ok: true } : { choice, ok: false, error: side.choice.error || 'rejected' };
		});
	}

	/**
	 * Showdown's own action speed for each active Pokemon, plus the pieces it is built
	 * from. Lets the Python Speed calculation be compared directly instead of inferred
	 * from a turn order, which would conflate it with the ordering logic.
	 */
	actionSpeeds(): {
		side: number;
		slot: number;
		species: string;
		ability: string;
		item: string | null;
		status: string | null;
		storedSpe: number;
		boostSpe: number;
		/** getStat('spe'), i.e. after boosts and every ModifySpe handler. */
		effectiveSpe: number;
		/** getActionSpeed(), i.e. effectiveSpe negated under Trick Room. */
		actionSpeed: number;
	}[] {
		const out = [];
		for (const [sideIndex, side] of this.battle.sides.entries()) {
			for (const [slot, mon] of side.active.entries()) {
				if (!mon || mon.fainted) continue;
				out.push({
					side: sideIndex,
					slot,
					species: mon.species.id,
					ability: mon.ability,
					item: mon.item || null,
					status: mon.status || null,
					storedSpe: mon.storedStats.spe,
					boostSpe: mon.boosts.spe,
					effectiveSpe: mon.getStat('spe', false, false),
					actionSpeed: mon.getActionSpeed(),
				});
			}
		}
		return out;
	}

	/**
	 * Serializes the battle to a string.
	 *
	 * A string, not an object: `State.deserializeBattle` mutates the object it is given
	 * (filling in live Battle/Side/Pokemon references), and `serializeBattle` assigns
	 * `state.log = battle.log` by reference, so an object snapshot would alias the
	 * original battle's log. The log is dropped because clones never need it and it is
	 * most of the payload.
	 */
	snapshot(): string {
		const state = this.battle.toJSON();
		state.log = [];
		return JSON.stringify(state);
	}

	/** Restores a battle from `snapshot()` and re-installs the randomness policy. */
	static load(snapshot: string, policy: RandomnessPolicy): AnyBattle {
		const battle = Battle.fromJSON(snapshot) as AnyBattle;
		battle.restart(() => {});
		battle.sentLogPos = battle.log.length;
		installPolicy(battle, policy);
		return battle;
	}
}

/** Stat spread evaluation: the authority our Python stat calculator is tested against. */
export function spreadStats(
	formatId: string,
	rows: { species: string; nature: string; sp: Partial<Record<string, number>> }[]
): Record<string, number>[] {
	const format = Dex.formats.get(formatId);
	const dex = Dex.forFormat(format);
	const level = Dex.formats.getRuleTable(format).adjustLevel ?? 50;
	// A Battle is needed because statModify lives on the format's battle scripts.
	const probe = new Battle({
		formatid: format.id as never,
		seed: '1,2,3,4' as never,
		send: () => {},
	});
	return rows.map(row => {
		const species = dex.species.get(row.species);
		if (!species.exists) throw new Error(`Unknown species: ${row.species}`);
		const evs: Record<string, number> = {};
		for (const stat of STAT_IDS) evs[stat] = row.sp[stat] ?? 0;
		const set = { evs, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, nature: row.nature, level };
		return (probe as AnyBattle).spreadModify(species.baseStats, set) as Record<string, number>;
	});
}

export function validateTeam(formatId: string, team: TeamSet[]): string[] | null {
	const format = Dex.formats.get(formatId);
	const level = Dex.formats.getRuleTable(format).adjustLevel ?? 50;
	const validator = new TeamValidator(format.id as never);
	return validator.validateTeam(team.map(s => toShowdownSet(s, level)) as never);
}

export function importTeam(text: string): Record<string, unknown>[] {
	return (Teams.import(text) ?? []) as unknown as Record<string, unknown>[];
}

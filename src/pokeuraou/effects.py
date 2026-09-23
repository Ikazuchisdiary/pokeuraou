"""Ability and item damage modifiers, transcribed from Showdown.

Each entry records the modifier *and* the event slot it belongs to, because Showdown
chains modifiers within an event and applies the accumulated result once -- an effect
placed in the wrong slot rounds differently and can shift damage by 1.

Every modifier here was read off ``vendor/pokemon-showdown/data/abilities.ts`` and
``items.ts`` at the pinned commit; the ``priority`` field is Showdown's own handler
priority, which fixes the chaining order. Effects that are *not* listed are not silently
ignored: :func:`unmodelled_effects` reports them, ``tools/coverage.py`` weights them by
metagame usage, and the differential test attributes every divergence to a named effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

#: The events in a damage calculation, in the order Showdown runs them.
Slot = Literal[
    "base_power",  # runEvent('BasePower')
    "atk",  # runEvent('ModifyAtk' / 'ModifySpA')
    "def",  # runEvent('ModifyDef' / 'ModifySpD')
    "weather_damage",  # priorityEvent('WeatherModifyDamage')
    "stab",  # runEvent('ModifySTAB')
    "damage",  # runEvent('ModifyDamage')
    "crit_ratio",  # runEvent('ModifyCritRatio'), an additive stage rather than a modifier
    "accuracy",  # runEvent('ModifyAccuracy')
    "speed",  # runEvent('ModifySpe')
]


@dataclass(frozen=True, slots=True)
class Ctx:
    """Everything a modifier predicate may consult about one attack.

    Scalar, not batched: ability, item, status, types and boosts are all public
    information under Champions Open Team Sheets, so only the stat spread varies across
    belief particles and none of these predicates depend on it.
    """

    move_id: str
    move_type: str
    move_category: str  # 'Physical' | 'Special' | 'Status'
    move_base_power: int
    move_flags: frozenset[str]
    move_priority: int
    #: Type effectiveness multiplier of the move against the defender.
    effectiveness: float
    #: log2 of effectiveness, clamped to -6..6, as Showdown's `typeMod`.
    type_mod: int
    is_crit: bool
    is_spread: bool
    has_secondary: bool
    #: Whether the move deals recoil or crash damage. Showdown keeps these as move fields
    #: (``recoil``, ``hasCrashDamage``) rather than flags, so Reckless cannot read
    #: ``move_flags`` for them.
    has_recoil: bool

    attacker_species: str
    attacker_types: tuple[str, ...]
    attacker_ability: str
    attacker_item: str | None
    attacker_status: str | None
    attacker_volatiles: frozenset[str]
    attacker_gender: str

    defender_species: str
    defender_types: tuple[str, ...]
    defender_ability: str
    defender_item: str | None
    defender_status: str | None
    defender_volatiles: frozenset[str]
    defender_gender: str
    #: True when the defender is at full HP for every particle in the batch. Multiscale
    #: is all-or-nothing per particle, so a mixed batch is handled by the caller.
    defender_at_full_hp: bool
    #: True when the attacker is at or below a third of its HP, which is what the pinch
    #: abilities (Overgrow, Blaze, Torrent, Swarm) key on.
    attacker_in_pinch: bool
    #: True when the attacker is below half HP, for Defeatist.
    attacker_below_half: bool

    weather: str | None
    terrain: str | None
    #: Side conditions on the defender's side.
    defender_side_conditions: frozenset[str]
    #: Abilities on the defender's *ally* only, for Friend Guard. It must not include the
    #: attacker's side: Friend Guard protects its partner, not the Pokemon it faces.
    defender_ally_abilities: tuple[str, ...]
    #: Abilities of every active Pokemon on both sides, for the field-wide Aura abilities.
    field_abilities: tuple[str, ...]
    #: Number of Pokemon on each half of the field (2 in doubles), for screen strength.
    active_per_half: int


@dataclass(frozen=True, slots=True)
class Modifier:
    """A single chainable modifier."""

    id: str
    slot: Slot
    numerator: float
    denominator: float = 1.0
    #: Showdown's handler priority; higher runs first, fixing the chaining order.
    priority: int = 0
    #: Whether the modifier applies to this attack.
    when: Callable[[Ctx], bool] = field(default=lambda _c: True)
    #: Set for defender-side handlers (onSourceModify*, onAnyModify*), for reporting only.
    from_defender: bool = False
    note: str = ""


def _has(flag: str) -> Callable[[Ctx], bool]:
    return lambda c: flag in c.move_flags


def _is_physical(c: Ctx) -> bool:
    return c.move_category == "Physical"


def _is_special(c: Ctx) -> bool:
    return c.move_category == "Special"


def _move_type(t: str) -> Callable[[Ctx], bool]:
    return lambda c: c.move_type == t


def _sun(c: Ctx) -> bool:
    return c.weather in ("sunnyday", "desolateland")


def _rain(c: Ctx) -> bool:
    return c.weather in ("raindance", "primordialsea")


# ---------------------------------------------------------------------------
# Abilities
# ---------------------------------------------------------------------------

#: Abilities that boost a move type from anywhere on the field. Showdown lets only one
#: holder apply (`move.auraBooster`), so they are collected separately from the attacker's
#: and defender's own abilities and applied once.
AURA_ABILITIES: dict[str, str] = {"fairyaura": "Fairy", "darkaura": "Dark"}

#: `chainModify([move.hasAuraBreak ? 3072 : 5448, 4096])`. Aura Break does not cancel an
#: aura, it inverts it into a reduction.
AURA_FP = 5448
AURA_BROKEN_FP = 3072
AURA_BREAK_ABILITY = "aurabreak"

ABILITY_MODIFIERS: dict[str, tuple[Modifier, ...]] = {
    # -- BasePower ---------------------------------------------------------
    "technician": (
        Modifier(
            "technician", "base_power", 1.5, priority=30,
            # Showdown compares the base power *after* modifiers chained so far; with
            # priority 30 it runs first, so this is the raw base power.
            when=lambda c: 0 < c.move_base_power <= 60,
        ),
    ),
    "toughclaws": (
        Modifier("toughclaws", "base_power", 5325, 4096, priority=21, when=_has("contact")),
    ),
    "sheerforce": (
        Modifier(
            "sheerforce", "base_power", 5325, 4096, priority=21,
            when=lambda c: c.has_secondary,
            note="also suppresses the secondary effect",
        ),
    ),
    "sharpness": (
        Modifier("sharpness", "base_power", 1.5, priority=19, when=_has("slicing")),
    ),
    "strongjaw": (Modifier("strongjaw", "base_power", 1.5, priority=19, when=_has("bite")),),
    "megalauncher": (
        Modifier("megalauncher", "base_power", 1.5, priority=19, when=_has("pulse")),
    ),
    "ironfist": (Modifier("ironfist", "base_power", 1.2, priority=23, when=_has("punch")),),
    "punkrock": (
        Modifier("punkrock", "base_power", 1.3, priority=7, when=_has("sound")),
        Modifier(
            "punkrock", "damage", 0.5, priority=0, from_defender=True, when=_has("sound"),
        ),
    ),
    "steelworker": (
        Modifier("steelworker", "atk", 1.5, priority=5, when=_move_type("Steel")),
    ),
    "dragonsmaw": (Modifier("dragonsmaw", "atk", 1.5, priority=5, when=_move_type("Dragon")),),
    "transistor": (
        Modifier("transistor", "atk", 5325, 4096, priority=5, when=_move_type("Electric")),
    ),
    "rockypayload": (
        Modifier("rockypayload", "atk", 1.5, priority=5, when=_move_type("Rock")),
    ),
    "waterbubble": (
        Modifier("waterbubble", "atk", 2.0, priority=5, when=_move_type("Water")),
        Modifier(
            "waterbubble", "atk", 0.5, priority=6, from_defender=True,
            when=_move_type("Fire"),
        ),
    ),
    "rivalry": (
        Modifier(
            "rivalry", "base_power", 1.25, priority=8,
            when=lambda c: (
                c.attacker_gender in ("M", "F")
                and c.attacker_gender == c.defender_gender
            ),
        ),
        Modifier(
            "rivalry", "base_power", 0.75, priority=8,
            when=lambda c: (
                c.attacker_gender in ("M", "F")
                and c.defender_gender in ("M", "F")
                and c.attacker_gender != c.defender_gender
            ),
        ),
    ),
    "reckless": (
        Modifier(
            "reckless", "base_power", 4915, 4096, priority=23,
            when=lambda c: c.has_recoil,
        ),
    ),
    "sandforce": (
        Modifier(
            "sandforce", "base_power", 5325, 4096, priority=21,
            when=lambda c: c.weather == "sandstorm"
            and c.move_type in ("Rock", "Ground", "Steel"),
        ),
    ),
    # Aura abilities boost the type for *everyone* on the field.
    "fairyaura": (
        Modifier("fairyaura", "base_power", 5448, 4096, priority=20, when=_move_type("Fairy")),
    ),
    "darkaura": (
        Modifier("darkaura", "base_power", 5448, 4096, priority=20, when=_move_type("Dark")),
    ),
    # -- Atk / SpA ---------------------------------------------------------
    "hugepower": (Modifier("hugepower", "atk", 2.0, priority=5, when=_is_physical),),
    "purepower": (Modifier("purepower", "atk", 2.0, priority=5, when=_is_physical),),
    "hustle": (
        Modifier("hustle", "atk", 1.5, priority=5, when=_is_physical),
        Modifier("hustle", "accuracy", 0.8, when=_is_physical),
    ),
    "guts": (
        Modifier(
            "guts", "atk", 1.5, priority=5,
            when=lambda c: _is_physical(c) and c.attacker_status is not None,
        ),
    ),
    "toxicboost": (
        Modifier(
            "toxicboost", "atk", 1.5, priority=5,
            when=lambda c: _is_physical(c) and c.attacker_status in ("psn", "tox"),
        ),
    ),
    "flareboost": (
        Modifier(
            "flareboost", "atk", 1.5, priority=5,
            when=lambda c: _is_special(c) and c.attacker_status == "brn",
        ),
    ),
    "solarpower": (
        Modifier(
            "solarpower", "atk", 1.5, priority=5, when=lambda c: _is_special(c) and _sun(c)
        ),
    ),
    "defeatist": (
        Modifier(
            "defeatist", "atk", 0.5, priority=5, when=lambda c: c.attacker_below_half
        ),
    ),
    # The pinch abilities boost their type at or below a third of maximum HP.
    "overgrow": (
        Modifier(
            "overgrow", "atk", 1.5, priority=5,
            when=lambda c: c.attacker_in_pinch and c.move_type == "Grass",
        ),
    ),
    "blaze": (
        Modifier(
            "blaze", "atk", 1.5, priority=5,
            when=lambda c: c.attacker_in_pinch and c.move_type == "Fire",
        ),
    ),
    "torrent": (
        Modifier(
            "torrent", "atk", 1.5, priority=5,
            when=lambda c: c.attacker_in_pinch and c.move_type == "Water",
        ),
    ),
    "swarm": (
        Modifier(
            "swarm", "atk", 1.5, priority=5,
            when=lambda c: c.attacker_in_pinch and c.move_type == "Bug",
        ),
    ),
    "gorillatactics": (
        Modifier("gorillatactics", "atk", 1.5, priority=5, when=_is_physical),
    ),
    "flowergift": (
        Modifier("flowergift", "atk", 1.5, priority=5, when=lambda c: _is_physical(c) and _sun(c)),
        Modifier(
            "flowergift", "def", 1.5, priority=5,
            when=lambda c: _is_special(c) and _sun(c),
            note="raises SpD",
        ),
    ),
    # Defender-side Atk/SpA reducers.
    "thickfat": (
        Modifier(
            "thickfat", "atk", 0.5, priority=6, from_defender=True,
            when=lambda c: c.move_type in ("Fire", "Ice"),
        ),
    ),
    "heatproof": (
        Modifier(
            "heatproof", "atk", 0.5, priority=6, from_defender=True, when=_move_type("Fire")
        ),
    ),
    "purifyingsalt": (
        Modifier(
            "purifyingsalt", "atk", 0.5, priority=6, from_defender=True,
            when=_move_type("Ghost"),
        ),
    ),
    # -- Def / SpD ---------------------------------------------------------
    "marvelscale": (
        Modifier(
            "marvelscale", "def", 1.5, priority=6,
            when=lambda c: _is_physical(c) and c.defender_status is not None,
        ),
    ),
    "furcoat": (Modifier("furcoat", "def", 2.0, priority=6, when=_is_physical),),
    "grasspelt": (
        Modifier(
            "grasspelt", "def", 1.5, priority=6,
            when=lambda c: _is_physical(c) and c.terrain == "grassyterrain",
        ),
    ),
    # -- ModifyDamage ------------------------------------------------------
    "tintedlens": (
        Modifier("tintedlens", "damage", 2.0, when=lambda c: c.type_mod < 0),
    ),
    "neuroforce": (
        Modifier("neuroforce", "damage", 5120, 4096, when=lambda c: c.type_mod > 0),
    ),
    "solidrock": (
        Modifier(
            "solidrock", "damage", 0.75, from_defender=True, when=lambda c: c.type_mod > 0
        ),
    ),
    "filter": (
        Modifier("filter", "damage", 0.75, from_defender=True, when=lambda c: c.type_mod > 0),
    ),
    "prismarmor": (
        Modifier(
            "prismarmor", "damage", 0.75, from_defender=True, when=lambda c: c.type_mod > 0
        ),
    ),
    "multiscale": (
        Modifier(
            "multiscale", "damage", 0.5, from_defender=True,
            when=lambda c: c.defender_at_full_hp,
        ),
    ),
    "shadowshield": (
        Modifier(
            "shadowshield", "damage", 0.5, from_defender=True,
            when=lambda c: c.defender_at_full_hp,
        ),
    ),
    "icescales": (
        Modifier("icescales", "damage", 0.5, from_defender=True, when=_is_special),
    ),
    "fluffy": (
        Modifier("fluffy", "damage", 0.5, from_defender=True, when=_has("contact")),
        Modifier("fluffy", "damage", 2.0, from_defender=True, when=_move_type("Fire")),
    ),
    # -- STAB --------------------------------------------------------------
    "adaptability": (
        Modifier(
            "adaptability", "stab", 2.0,
            when=lambda c: c.move_type in c.attacker_types,
            note="replaces the 1.5x STAB rather than chaining with it",
        ),
    ),
}

#: Abilities that change a move's type and add a 1.2x boost (``-ate`` abilities).
TYPE_CHANGING_ABILITIES: dict[str, str] = {
    "aerilate": "Flying",
    "pixilate": "Fairy",
    "galvanize": "Electric",
    "refrigerate": "Ice",
    "normalize": "Normal",
    "liquidvoice": "Water",
}
#: Showdown chains [4915, 4096] for the -ate abilities' boost.
TYPE_CHANGE_BOOST_FP = 4915

#: Abilities granting outright immunity to a type, with the side effect they trigger.
TYPE_IMMUNITY_ABILITIES: dict[str, str] = {
    "levitate": "Ground",
    "flashfire": "Fire",
    "waterabsorb": "Water",
    "dryskin": "Water",
    "voltabsorb": "Electric",
    "lightningrod": "Electric",
    "motordrive": "Electric",
    "stormdrain": "Water",
    "sapsipper": "Grass",
    "eartheater": "Ground",
    "wellbakedbody": "Fire",
    "windrider": "Flying",
}

#: Abilities that ignore the defender's ability-based damage reduction.
MOLD_BREAKER_ABILITIES = frozenset({"moldbreaker", "teravolt", "turboblaze", "myceliummight"})

#: Abilities that retype the user to the move's type before it lands, which makes every
#: move a STAB move. In gen 9 this fires once per switch-in and Showdown records it with a
#: volatile of the same name.
RETYPING_ABILITIES = frozenset({"protean", "libero"})

#: Abilities that let Normal and Fighting moves hit Ghost types.
GHOST_PIERCING_ABILITIES = frozenset({"scrappy", "mindseye"})

#: Abilities that absorb one hit entirely while their forme is intact.
HIT_ABSORBING_ABILITIES = frozenset({"disguise", "iceface"})

#: Weather-setting abilities, needed so a switch-in can change the field.
WEATHER_ABILITIES: dict[str, str] = {
    "drought": "sunnyday",
    "drizzle": "raindance",
    "sandstream": "sandstorm",
    "snowwarning": "snowscape",
    "desolateland": "desolateland",
    "primordialsea": "primordialsea",
    "deltastream": "deltastream",
}

#: Abilities that suppress weather effects entirely.
WEATHER_SUPPRESSING_ABILITIES = frozenset({"cloudnine", "airlock"})


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

#: Type-boosting held items: 1.2x on the matching type. Restricted to the M-C pool.
TYPE_BOOST_ITEMS: dict[str, str] = {
    "blackbelt": "Fighting",
    "blackglasses": "Dark",
    "charcoal": "Fire",
    "dragonfang": "Dragon",
    "fairyfeather": "Fairy",
    "hardstone": "Rock",
    "magnet": "Electric",
    "metalcoat": "Steel",
    "miracleseed": "Grass",
    "mysticwater": "Water",
    "nevermeltice": "Ice",
    "poisonbarb": "Poison",
    "sharpbeak": "Flying",
    "silkscarf": "Normal",
    "silverpowder": "Bug",
    "softsand": "Ground",
    "spelltag": "Ghost",
    "twistedspoon": "Psychic",
}

#: Single-use berries that halve a super-effective hit of their type. Chilan is the
#: exception: it halves Normal-type damage regardless of effectiveness.
RESIST_BERRIES: dict[str, str] = {
    "babiriberry": "Steel",
    "chartiberry": "Rock",
    "chilanberry": "Normal",
    "chopleberry": "Fighting",
    "cobaberry": "Flying",
    "colburberry": "Dark",
    "habanberry": "Dragon",
    "kasibberry": "Ghost",
    "kebiaberry": "Poison",
    "occaberry": "Fire",
    "passhoberry": "Water",
    "payapaberry": "Psychic",
    "rindoberry": "Grass",
    "roseliberry": "Fairy",
    "shucaberry": "Ground",
    "tangaberry": "Bug",
    "wacanberry": "Electric",
    "yacheberry": "Ice",
}

ITEM_MODIFIERS: dict[str, tuple[Modifier, ...]] = {
    "lifeorb": (Modifier("lifeorb", "damage", 5324, 4096, note="also 10% recoil"),),
    "expertbelt": (
        Modifier("expertbelt", "damage", 4915, 4096, when=lambda c: c.type_mod > 0),
    ),
    # Choice Band and Choice Specs are deliberately absent: neither is in this
    # regulation's item pool, so a modifier for them could never fire and could never be
    # tested. `tools/coverage.py` reports an unmodelled item that does exist; untestable
    # code that guards against a hypothetical one is worse than that report.
    "muscleband": (Modifier("muscleband", "base_power", 4505, 4096, when=_is_physical),),
    "wiseglasses": (Modifier("wiseglasses", "base_power", 4505, 4096, when=_is_special),),
    "normalgem": (
        Modifier(
            "normalgem", "base_power", 5325, 4096, when=_move_type("Normal"),
            note="consumed on use",
        ),
    ),
    "lightball": (
        Modifier(
            "lightball", "atk", 2.0, priority=5,
            when=lambda c: c.attacker_species.startswith("pikachu"),
        ),
    ),
    "choicescarf": (Modifier("choicescarf", "speed", 1.5),),
    "ironball": (Modifier("ironball", "speed", 0.5),),
    "widelens": (Modifier("widelens", "accuracy", 4505, 4096),),
    "zoomlens": (Modifier("zoomlens", "accuracy", 1.2, note="only when moving second"),),
    "brightpowder": (
        Modifier("brightpowder", "accuracy", 0.9, from_defender=True),
    ),
    "scopelens": (Modifier("scopelens", "crit_ratio", 1, note="+1 crit stage"),),
    "leek": (
        Modifier(
            "leek", "crit_ratio", 2,
            when=lambda c: c.attacker_species.startswith(("farfetchd", "sirfetchd")),
            note="+2 crit stages",
        ),
    ),
}

#: Items and abilities that let a full-HP Pokemon survive an otherwise lethal hit at 1 HP.
#: They do not modify the damage roll; they cap the damage actually dealt, which is what
#: Showdown's protocol reports. All of them test `effect.effectType === 'Move'`, so they
#: do not save from residual damage, recoil or Life Orb.
SURVIVE_AT_ONE_ITEMS = frozenset({"focussash"})
SURVIVE_AT_ONE_ABILITIES = frozenset({"sturdy"})

#: Items that give a *chance* to survive a lethal hit at 1 HP, as (numerator, denominator).
#: Focus Band is deliberately not in SURVIVE_AT_ONE_ITEMS: it is `randomChance(1, 10)`
#: from any HP and is not consumed, so treating it as a certain full-HP save is wrong in
#: both directions.
SURVIVE_CHANCE_ITEMS: dict[str, tuple[int, int]] = {"focusband": (1, 10)}

#: Side conditions that reduce damage, with their doubles strength. Showdown uses
#: chainModify([2732, 4096]) when activePerHalf > 1 and 0.5 in singles.
SCREEN_CONDITIONS: dict[str, str] = {
    "reflect": "Physical",
    "lightscreen": "Special",
    "auroraveil": "both",
}
SCREEN_FP_DOUBLES = 2732
SCREEN_FP_SINGLES = 2048


def item_is_removable(reg: object, species_id: str, item_id: str | None) -> bool:
    """Whether an item can be taken off this Pokemon.

    A Mega Stone refuses (`onTakeItem` returns false for the species it belongs to), which
    matters three times over: Knock Off gets no boost, Knock Off takes nothing, and Trick
    cannot swap it away. Read from the regulation's own mega map rather than a list here.
    """
    if not item_id:
        return False
    target = getattr(reg, "mega_target", None)
    if target is None:
        return True
    return target(species_id, item_id) is None


def type_boost_item_fp() -> int:
    """Type-boosting items chain [4915, 4096], i.e. 1.2x."""
    return 4915


@cache
def all_modelled_abilities() -> frozenset[str]:
    """Every ability the calculator accounts for, so it can report the ones it does not.

    Cached because `calculate` consults it on every hit and the answer is a constant of
    the module.
    """
    return frozenset(
        set(ABILITY_MODIFIERS)
        | set(TYPE_CHANGING_ABILITIES)
        | set(TYPE_IMMUNITY_ABILITIES)
        | set(WEATHER_ABILITIES)
        | MOLD_BREAKER_ABILITIES
        | WEATHER_SUPPRESSING_ABILITIES
        | RETYPING_ABILITIES
        | GHOST_PIERCING_ABILITIES
        | HIT_ABSORBING_ABILITIES
        # Abilities with no effect on a damage calculation.
        | {
            # Implemented in the resolver rather than the calculator, and missing from
            # this set -- so the calculator reported `attacker.ability:stancechange` on
            # every hit, 158 times in a 13,000-game run, for an ability that works. The
            # forme change it drives does not touch a damage modifier; it changes which
            # Pokemon the stats are read from, which happens before `calculate` is called.
            "stancechange",
            "defiant", "competitive", "intimidate", "prankster", "roughskin", "ironbarbs",
            "hospitality", "unburden", "armortail", "queenlymajesty", "dazzling", "contrary",
            "stamina", "poisontouch", "chlorophyll", "swiftswim", "sandrush", "slushrush",
            "goodasgold", "shadowtag", "arenatrap", "magnetpull", "rockhead", "regenerator",
            "aurabreak", "innerfocus", "toxicdebris", "friendguard", "soundproof", "compoundeyes",
            "unnerve", "mirrorarmor", "galewings",
            "static", "flamebody", "effectspore", "cutecharm", "synchronize", "trace",
            "clearbody", "whitesmoke", "fullmetalbody", "owntempo", "oblivious", "limber",
            "immunity", "insomnia", "vitalspirit", "waterveil", "magicguard", "sturdy",
            "wonderguard", "pressure", "moxie", "justified", "weakarmor", "sandveil",
            "snowcloak", "runaway", "keeneye", "hypercutter", "bigpecks", "telepathy",
            "healer", "symbiosis", "sweetveil", "flowerveil", "aromaveil", "bulletproof",
            # (overgrow/blaze/torrent/swarm are modelled above, not here)
            "overcoat", "magician", "pickpocket", "gluttony", "harvest", "cheekpouch",
            "naturalcure", "shedskin", "hydration", "raindish", "icebody", "leafguard",
            "quickfeet", "steadfast", "rattled", "anticipation", "forewarn", "frisk",
            "superluck", "sniper", "battlearmor", "shellarmor", "damp", "lightmetal",
            "heavymetal", "wanderingspirit", "perishbody", "seedsower", "windpower",
            "protosynthesis", "quarkdrive", "beadsofruin", "swordofruin", "tabletsofruin",
            "vesselofruin", "supremeoverlord", "costar", "guarddog", "cudchew",
            "wellbakedbody", "eartheater", "myceliummight", "toxicchain", "supersweetsyrup",
            "embodyaspect", "spicyspray",
            # The hit count, in `resolve.multihit_counts` (IKA-160).
            "skilllink",
            # A switch after the hit, in `resolve._emergency_exit` (IKA-191).
            "emergencyexit", "wimpout",
        }
    )


@cache
def all_modelled_items(mega_stones: frozenset[str] = frozenset()) -> frozenset[str]:
    """Items whose effect on a damage roll is accounted for.

    ``mega_stones`` lets a caller declare the regulation's stones as handled: they are not
    damage modifiers at all -- their whole effect is the forme change, which the mega
    machinery owns -- so counting them as gaps would misreport coverage.
    """
    return frozenset(
        set(ITEM_MODIFIERS)
        | set(TYPE_BOOST_ITEMS)
        | set(RESIST_BERRIES)
        | set(mega_stones)
        # Items with no direct effect on a damage roll.
        | {
            "focussash", "sitrusberry", "leftovers", "lumberry", "mentalherb", "whiteherb",
            "rockyhelmet", "redcard", "ejectbutton", "shedshell", "airballoon", "focusband",
            "lightclay", "terrainextender", "damprock", "heatrock", "icyrock",
            "smoothrock", "electricseed", "grassyseed", "mistyseed", "psychicseed",
            "quickclaw", "kingsrock", "shellbell", "bigroot", "bindingband", "metronome",
            "leppaberry", "oranberry", "aspearberry", "cheriberry", "chestoberry",
            "pechaberry", "persimberry", "rawstberry", "custapberry",
        }
    )


def unmodelled_effects(
    ability_ids: frozenset[str], item_ids: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Which of the given abilities/items this module does not model.

    Reported rather than assumed away: the coverage tool weights the result by metagame
    usage so the next thing to implement is chosen by impact, not by guesswork.
    """
    return (
        ability_ids - all_modelled_abilities(),
        item_ids - all_modelled_items(),
    )

__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Fault schedules for dataset episodes.

One schedule is sampled per episode, up front, from the episode seed. The episode runner then feeds
it to FaultInjector at the scheduled simulation times through the same public inject_* methods the
Yamcs commander calls - so a fault in a dataset run is the same fault an operator would trigger.

Three sampling rules, all of which come from what the dataset is for rather than from what is easy:

MAJORITY NOMINAL. A detector needs negatives, so most episodes carry no fault at all. The default
class weights put 60% of episodes clean.

ONSET IN THE MIDDLE. A fault that starts at t=0 teaches a classifier to recognise a regime, not to
detect a transition. Every onset is sampled inside a central window so each faulted episode holds
three phases: a clean baseline, the transition, and the settled faulted regime.

OVERLAP IS RARE. Simultaneous faults happen and have to be represented, but they are the unusual
case - about 9% of episodes. Sequential pairs (one recovers, another begins) are rarer still.

Windows are expressed as fractions of the episode, so changing the episode length rescales
everything and the three phases stay proportional.

No omni/pxr imports - this module is testable with plain python3.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.mission_specific.perseverance.dataset import crater as crater_module
from src.mission_specific.perseverance.faults.fault_injector import ALL, CORNERS, WHEELS

# Fault kinds, in the shape apply_spec uses: (targets or None, how many magnitudes, recoverable).
# steer_stuck is not recoverable: it has no severity, so there is no magnitude that means "healthy"
# and only clear_faults releases it. That fact propagates into the sampling rules below.
KIND_SHAPES = {
    "wheel_torque": (WHEELS, 1, True),
    "wheel_stuck": (WHEELS, 1, True),
    "wheel_slip": (WHEELS, 1, True),
    "wheel_sink": (WHEELS, 1, True),
    "steer_torque": (CORNERS, 1, True),
    "steer_stuck": (CORNERS, 1, False),
    "imu": (None, 2, True),
    "camera": (None, 2, True),
    "battery": (None, 1, True),
    "comms": (None, 2, True),
}

# Kinds whose two magnitudes are independent failure modes rather than one magnitude split in two.
# The README calls these out as distinct signatures - an imu bias is a shifted trace, imu noise is a
# rougher one - so a schedule picks a mode rather than always driving both at once.
TWO_MODE_KINDS = {
    "imu": ("bias", "noise"),
    "camera": ("loss", "noise"),
    "comms": ("tm_loss", "tc_loss"),
}

# "crater" is last on purpose: with its weight at zero, rng.choices draws exactly what it drew before
# the class existed, so seeds from older runs still give the same schedules.
EPISODE_CLASSES = ("nominal", "single", "concurrent", "sequential", "crater")

DEFAULTS = {
    # Episode-class mix. nominal is deliberately the majority.
    # crater only counts when generate() is given a crater config (the --crater flag); otherwise it
    # is forced to zero whatever the config says.
    "class_weights": {"nominal": 0.60, "single": 0.28, "concurrent": 0.09, "sequential": 0.03, "crater": 0.0},
    # Relative likelihood of each fault kind, given that a fault happens at all.
    "kind_weights": {
        "wheel_torque": 1.0,
        "wheel_stuck": 1.0,
        "wheel_slip": 1.0,
        "wheel_sink": 1.0,
        "steer_torque": 1.0,
        "steer_stuck": 0.7,
        "imu": 1.0,
        "camera": 1.0,
        "battery": 1.0,
        "comms": 1.0,
    },
    # Fractions of the episode. A single fault onsets in the middle; the phases either side are what
    # make the transition learnable.
    "onset_window": [0.40, 0.60],
    # Concurrent pairs: the second fault follows this long after the first, and neither recovers, so
    # the overlap is guaranteed to be real rather than incidental.
    "concurrent_gap": [0.03, 0.12],
    # Sequential pairs need room for four events, so they start earlier and use a wider span.
    "sequential_first_onset": [0.20, 0.30],
    "sequential_hold": [0.15, 0.25],
    "sequential_gap": [0.05, 0.15],
    "sequential_last_onset_max": 0.80,
    # A single fault sometimes clears on its own; a recovery transition is worth having in the data.
    "recovery_probability": 0.25,
    "recovery_hold": [0.15, 0.30],
    "recovery_onset_max": 0.90,
    # Severities start well above zero: a fault too weak to have an effect is a mislabelled nominal
    # episode, which is worse for training than no episode at all.
    "severity_range": [0.30, 1.00],
    # steer_stuck is an angle, not a severity. Sampled away from zero in both directions, since a
    # corner stuck at 0 deg is nearly indistinguishable from a healthy one driving straight.
    "steer_stuck_angle_deg": [10.0, 45.0],
    # How often an actuator fault hits every wheel/corner at once instead of just one.
    "all_targets_probability": 0.15,
}


@dataclass(frozen=True)
class FaultEvent:
    """One scheduled change of fault state: an onset, or a recovery back to healthy."""

    time_s: float
    kind: str
    target: Optional[str]
    magnitudes: Tuple[float, ...]
    action: str  # "onset" or "recovery"

    def as_dict(self) -> Dict:
        return {
            "time_s": round(self.time_s, 3),
            "kind": self.kind,
            "target": self.target,
            "magnitudes": [round(m, 6) for m in self.magnitudes],
            "action": self.action,
        }


@dataclass
class ScheduledFault:
    """One fault's whole life: when it starts, how hard, and when (if ever) it lets go."""

    kind: str
    target: Optional[str]
    magnitudes: Tuple[float, ...]
    onset_s: float
    recovery_s: Optional[float] = None
    mode: str = ""  # which of a two-mode kind's magnitudes is driven; "" for single-magnitude kinds

    def events(self) -> List[FaultEvent]:
        events = [FaultEvent(self.onset_s, self.kind, self.target, self.magnitudes, "onset")]
        if self.recovery_s is not None:
            # Recovery re-injects the same family at zero, which is how a torque fault is lifted.
            # Never scheduled for steer_stuck; see KIND_SHAPES.
            healthy = tuple(0.0 for _ in self.magnitudes)
            events.append(FaultEvent(self.recovery_s, self.kind, self.target, healthy, "recovery"))

        return events

    def as_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "target": self.target,
            "magnitudes": [round(m, 6) for m in self.magnitudes],
            "mode": self.mode,
            "onset_s": round(self.onset_s, 3),
            "recovery_s": None if self.recovery_s is None else round(self.recovery_s, 3),
        }


@dataclass
class FaultSchedule:
    """A whole episode's worth of fault activity, plus the label bookkeeping that goes with it."""

    episode_class: str
    duration_s: float
    faults: List[ScheduledFault] = field(default_factory=list)
    # A crater episode's terrain and operator plan; None for every other class.
    crater: Optional["crater_module.CraterScenario"] = None
    _events: List[FaultEvent] = field(default_factory=list, repr=False)
    _next: int = field(default=0, repr=False)

    def __post_init__(self):
        if not self._events:
            self._events = sorted(
                (event for fault in self.faults for event in fault.events()),
                key=lambda event: event.time_s,
            )

    # ── playback ─────────────────────────────────────────────────────────────────
    def pop_due(self, time_s: float) -> List[FaultEvent]:
        """Every event whose time has arrived since the last call. Monotonic in time_s."""
        due = []
        while self._next < len(self._events) and self._events[self._next].time_s <= time_s:
            due.append(self._events[self._next])
            self._next += 1

        return due

    def reset_playback(self) -> None:
        self._next = 0

    # ── labels ───────────────────────────────────────────────────────────────────
    @property
    def is_nominal(self) -> bool:
        """No fault and no planned trap. A crater the rover only avoids or skirts is still nominal."""
        return not self.faults and not self.plans_trap

    @property
    def plans_trap(self) -> bool:
        return self.crater is not None and self.crater.outcome == "trapped"

    def crater_planned_at(self, time_s: float) -> bool:
        """True from the dash onward in a trapped episode. The plan, not the truth - see the recorder."""
        return self.plans_trap and self.crater.dash_s is not None and time_s >= self.crater.dash_s

    def active_at(self, time_s: float) -> List[ScheduledFault]:
        """Which faults are live at this instant. The per-sample label."""
        return [
            fault for fault in self.faults
            if fault.onset_s <= time_s and (fault.recovery_s is None or time_s < fault.recovery_s)
        ]

    def faulted_fraction(self) -> float:
        """
        Fraction of the episode during which at least one fault is active.

        Episode-level class balance is not what a per-timestep detector actually trains on: with a
        clean lead-in inside every faulted episode, the sample-level positive rate is far below the
        episode-level one. Reporting both keeps that from being a surprise later.
        """
        spans = [
            (fault.onset_s, self.duration_s if fault.recovery_s is None else fault.recovery_s)
            for fault in self.faults
        ]
        # A trapped rover stays trapped, so the planned span runs from the dash to the end. It is an
        # upper bound: the rover takes a few seconds to reach the rim after the dash begins.
        if self.plans_trap and self.crater.dash_s is not None:
            spans.append((self.crater.dash_s, self.duration_s))
        if not spans or self.duration_s <= 0:
            return 0.0

        # Union of the active intervals, which may overlap by construction.
        spans.sort()
        covered, current_start, current_end = 0.0, spans[0][0], spans[0][1]
        for start, end in spans[1:]:
            if start > current_end:
                covered += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        covered += current_end - current_start

        return max(0.0, min(1.0, covered / self.duration_s))

    def as_dict(self) -> Dict:
        return {
            "episode_class": self.episode_class,
            "duration_s": round(self.duration_s, 3),
            "is_nominal": self.is_nominal,
            "faulted_fraction": round(self.faulted_fraction(), 6),
            "faults": [fault.as_dict() for fault in self.faults],
            "events": [event.as_dict() for event in self._events],
            "crater": None if self.crater is None else self.crater.as_dict(),
        }


# ── sampling ─────────────────────────────────────────────────────────────────────
def merged_config(config: Optional[Dict] = None) -> Dict:
    """DEFAULTS overlaid with the robot config's dataset block, one level deep."""
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    for key, value in dict(config or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = value

    return merged


def _uniform(rng: random.Random, bounds: Sequence[float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return rng.uniform(low, high) if high > low else low


def _weighted_choice(rng: random.Random, weights: Dict[str, float], exclude: Sequence[str] = ()) -> str:
    names = [name for name, weight in weights.items() if weight > 0 and name not in exclude]
    if not names:
        raise ValueError(f"no candidates left after excluding {sorted(exclude)}")

    return rng.choices(names, weights=[float(weights[name]) for name in names], k=1)[0]


def _sample_magnitudes(rng: random.Random, kind: str, config: Dict) -> Tuple[Tuple[float, ...], str]:
    """Magnitudes for one fault, plus which mode of a two-mode kind was chosen."""
    if kind == "steer_stuck":
        angle = _uniform(rng, config["steer_stuck_angle_deg"]) * rng.choice((-1.0, 1.0))
        return (angle,), ""

    severity = _uniform(rng, config["severity_range"])
    modes = TWO_MODE_KINDS.get(kind)
    if modes is None:
        return (severity,), ""

    # Either failure mode alone, or both together - three equally likely shapes.
    mode = rng.choice((modes[0], modes[1], "both"))
    if mode == modes[0]:
        return (severity, 0.0), mode
    if mode == modes[1]:
        return (0.0, severity), mode

    return (severity, _uniform(rng, config["severity_range"])), "both"


def _sample_target(rng: random.Random, kind: str, config: Dict) -> Optional[str]:
    targets = KIND_SHAPES[kind][0]
    if targets is None:
        return None
    if rng.random() < float(config["all_targets_probability"]):
        return ALL

    return rng.choice(targets)


def _make_fault(rng: random.Random, kind: str, onset_s: float, duration_s: float,
                config: Dict, recovery_s: Optional[float] = None) -> ScheduledFault:
    magnitudes, mode = _sample_magnitudes(rng, kind, config)

    return ScheduledFault(
        kind=kind,
        target=_sample_target(rng, kind, config),
        magnitudes=magnitudes,
        onset_s=onset_s,
        recovery_s=recovery_s,
        mode=mode,
    )


def _recoverable_kinds(config: Dict) -> Dict[str, float]:
    return {
        kind: weight for kind, weight in config["kind_weights"].items()
        if weight > 0 and KIND_SHAPES.get(kind, (None, 0, False))[2]
    }


def _single_fault(rng: random.Random, duration_s: float, config: Dict) -> ScheduledFault:
    kind = _weighted_choice(rng, config["kind_weights"])
    onset = _uniform(rng, config["onset_window"]) * duration_s

    recovery = None
    if KIND_SHAPES[kind][2] and rng.random() < float(config["recovery_probability"]):
        hold = _uniform(rng, config["recovery_hold"]) * duration_s
        recovery = min(onset + hold, float(config["recovery_onset_max"]) * duration_s)
        # A recovery that lands on top of the onset is not a transition worth recording.
        if recovery <= onset:
            recovery = None

    return _make_fault(rng, kind, onset, duration_s, config, recovery)


def generate(seed: int, duration_s: float, config: Optional[Dict] = None,
             crater_config: Optional[Dict] = None) -> FaultSchedule:
    """
    Sample one episode's fault schedule.

    Deterministic in seed: the same seed and duration always give the same schedule, which is what
    makes a recorded episode reproducible from its metadata alone.

    crater_config enables the crater class (see crater.py); without it that class never comes up.
    """
    config = merged_config(config)
    rng = random.Random(seed)
    duration_s = float(duration_s)

    weights = dict(config["class_weights"])
    if crater_config is None:
        weights["crater"] = 0.0
    if sum(float(weights.get(name, 0.0)) for name in EPISODE_CLASSES) <= 0.0:
        raise ValueError("every episode class has zero weight; a crater-only class_weights needs --crater")
    episode_class = rng.choices(
        list(EPISODE_CLASSES),
        weights=[float(weights.get(name, 0.0)) for name in EPISODE_CLASSES],
        k=1,
    )[0]

    if episode_class == "nominal":
        return FaultSchedule(episode_class, duration_s, [])

    if episode_class == "crater":
        crater_config = crater_module.merged_config(crater_config)
        scenario = crater_module.sample_scenario(rng, duration_s, crater_config)
        # Overlap is allowed: a crater episode can also carry one fault, drawn like a single episode's.
        faults = []
        if rng.random() < float(crater_config["fault_probability"]):
            faults.append(_single_fault(rng, duration_s, config))
        return FaultSchedule(episode_class, duration_s, faults, crater=scenario)

    if episode_class == "single":
        return FaultSchedule(episode_class, duration_s, [_single_fault(rng, duration_s, config)])

    if episode_class == "concurrent":
        # Different kinds, so both are simultaneously visible in different health flags rather than
        # the second silently overwriting the first. Neither recovers, which is what guarantees the
        # overlap is real.
        first_kind = _weighted_choice(rng, config["kind_weights"])
        second_kind = _weighted_choice(rng, config["kind_weights"], exclude=(first_kind,))

        first_onset = _uniform(rng, config["onset_window"]) * duration_s
        gap = _uniform(rng, config["concurrent_gap"]) * duration_s
        second_onset = min(first_onset + gap, float(config["recovery_onset_max"]) * duration_s)

        return FaultSchedule(episode_class, duration_s, [
            _make_fault(rng, first_kind, first_onset, duration_s, config),
            _make_fault(rng, second_kind, second_onset, duration_s, config),
        ])

    # sequential: the first fault has to be able to let go, so it is drawn from the recoverable
    # kinds only - steer_stuck can never be the first half of a sequential pair.
    first_kind = _weighted_choice(rng, _recoverable_kinds(config))
    second_kind = _weighted_choice(rng, config["kind_weights"], exclude=(first_kind,))

    first_onset = _uniform(rng, config["sequential_first_onset"]) * duration_s
    recovery = first_onset + _uniform(rng, config["sequential_hold"]) * duration_s
    second_onset = min(
        recovery + _uniform(rng, config["sequential_gap"]) * duration_s,
        float(config["sequential_last_onset_max"]) * duration_s,
    )
    # Clamp rather than resample, but never let the two touch: overlapping would make this a
    # concurrent episode wearing the wrong label.
    if second_onset <= recovery:
        recovery = max(first_onset, second_onset - 1.0)

    return FaultSchedule(episode_class, duration_s, [
        _make_fault(rng, first_kind, first_onset, duration_s, config, recovery),
        _make_fault(rng, second_kind, second_onset, duration_s, config),
    ])


# ── fixed schedules (--fault with --dataset-out) ──────────────────────────────────
def parse_spec(spec: str) -> Tuple[str, Optional[str], Tuple[float, ...], Optional[float]]:
    """
    Parse one --fault spec into (kind, target, magnitudes, onset_s).

    Same shapes as FaultInjector.apply_spec, plus an optional "@<seconds>" suffix for the onset:

        wheel_slip:front_left:0.8          onset sampled in onset_window
        wheel_slip:front_left:0.8@300      onset at 300 s of simulation time

    Raises ValueError rather than printing: a dataset run that silently drops its fault would record
    a whole episode of nominal data labelled as the fault it was asked for.
    """
    body, _, onset_text = spec.partition("@")
    parts = [part.strip() for part in body.split(":")]
    shape = KIND_SHAPES.get(parts[0])
    if shape is None:
        raise ValueError(f"unknown fault kind in {spec!r}; expected one of {sorted(KIND_SHAPES)}")

    targets, magnitude_count, _ = shape
    takes_target = targets is not None
    expected = 1 + int(takes_target) + magnitude_count
    if len(parts) != expected:
        raise ValueError(f"bad fault spec {spec!r}: expected {expected} colon-separated fields, got {len(parts)}")

    target = None
    if takes_target:
        target = parts[1]
        if target != ALL and target not in targets:
            raise ValueError(f"bad target {target!r} in {spec!r}; expected {ALL} or one of {list(targets)}")

    try:
        magnitudes = tuple(float(part) for part in parts[1 + int(takes_target):])
        onset_s = float(onset_text) if onset_text.strip() else None
    except ValueError:
        raise ValueError(f"non-numeric magnitude or onset in {spec!r}") from None
    if onset_s is not None and onset_s < 0.0:
        raise ValueError(f"negative onset in {spec!r}")

    return parts[0], target, magnitudes, onset_s


def _mode_of(kind: str, magnitudes: Tuple[float, ...]) -> str:
    modes = TWO_MODE_KINDS.get(kind)
    if modes is None:
        return ""
    if magnitudes[0] and magnitudes[1]:
        return "both"

    return modes[0] if magnitudes[0] else modes[1]


def from_specs(specs: Sequence[str], seed: int, duration_s: float,
               config: Optional[Dict] = None) -> FaultSchedule:
    """
    A schedule holding exactly the given faults, for recording a chosen scenario.

    Kind, target and magnitudes come from the specs. An onset left out is drawn from onset_window
    with the episode seed, so the episode still opens with a clean baseline. Nothing recovers.
    """
    config = merged_config(config)
    rng = random.Random(seed)
    duration_s = float(duration_s)

    faults = []
    for spec in specs:
        kind, target, magnitudes, onset_s = parse_spec(spec)
        if onset_s is None:
            onset_s = _uniform(rng, config["onset_window"]) * duration_s
        elif onset_s >= duration_s:
            raise ValueError(f"onset {onset_s} s in {spec!r} is past the {duration_s:.1f} s episode")
        faults.append(ScheduledFault(kind, target, magnitudes, onset_s, mode=_mode_of(kind, magnitudes)))

    if not faults:
        return FaultSchedule("nominal", duration_s, [])

    return FaultSchedule("single" if len(faults) == 1 else "concurrent", duration_s, faults)

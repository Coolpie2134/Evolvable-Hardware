"""
nv_evo/pulse.py — deterministic event-level pulse abstraction.

The nervous net is "a continuous-time, asynchronous system with binary-valued
output. It propagates a pulse arriving at its input with a fixed delay" (§3).
This module preserves those digital interface semantics, but it is not a
transistor- or charge-level reproduction of Fig. 1:

  * a WIRE (each cell's output net) is idle or carries a pulse [start, end);
  * all action is precipitated by a pulse EDGE (the pulse's leading edge —
    the paper's 1->0 transition; polarity is abstracted away);
  * a buffer node (E1==E2) triggers on a single edge of its one input wire;
    a coincidence node triggers only when edges arrive on BOTH excitatory
    wires within the coincidence window COINC ("neither input alone can
    trigger a response");
  * an active inhibitory input (its wire is pulsing at trigger time) prevents
    the response;
  * a triggered node emits a pulse of width WIDTH after a fixed delay DELAY,
    and is refractory until its own pulse ends (the comparator hysteresis —
    no chattering);
  * external inputs are INJECTED onto the input cell's net (wired-OR, like
    driving the physical perimeter wire). A held-high input is one long pulse
    — one edge — not a train of edges.

DELAY, WIDTH, and COINC are independent behavioural parameters in this
abstraction. In the physical circuit those behaviours are coupled through the
charge/leak/comparator dynamics implemented separately by ``paper_analog``.
Every leading edge is retained in ``rise_times`` for
continuous-time fitness; wires are also sampled once per TICK for playback and
legacy persistence-window targets.

With the default constants (DELAY = WIDTH = TICK, COINC < TICK) behaviour on
the integer tick lattice matches the earlier synchronous engine — that engine
was the quantization of this one. The constants can now be varied to study
sub-tick timing (unequal delays, edge alignment, incommensurate loop periods).

NODE-TIMING MODELS (PulseConfig.model). The baseline abstraction regenerates ONE
fixed pulse width after ONE fixed delay, so an input pulse's LENGTH is discarded the
moment the first node fires (every internal wire has a single driver, so no
merging ever restores it). That throws away a degree of freedom, so alongside
the baseline this module offers two variants that give width a role:

  * 'uniform'       — fixed WIDTH and DELAY at every node (legacy default)
  * 'evolved_width' — each node's emitted pulse width is under genetic control
                      (an evolvable per-node-type multiplier of WIDTH), passed
                      in as a {cell: width} map. Delay stays DELAY. Pulse width
                      becomes a heritable trait selection can tune.
  * 'pulse_delay'   — retained as the checkpoint/API identifier for the
                      width-preserving model: both edges are transported after
                      the firing node's evolvable delay d, so [t, t+w) becomes
                      [t+d, t+d+w). A neutral genome uses DELAY everywhere.

All three coincide when every pulse is one WIDTH wide and evolved widths are
neutral. Legacy runs remain byte-identical under the default 'uniform' model.

REFRACTORY DIVERGENCE ON WIDE PULSES. A node's refractory period is its own
output-pulse duration in every model — but under width preservation that
duration follows the INPUT: a 'pulse_delay' node transporting a pulse of width
w is busy for delay + w, while a 'uniform' node regenerating the same edge is
refractory for only DELAY + WIDTH. So on stimulus wider than WIDTH (e.g. the
mixed 0.5-2.25s oracle event banks) a NEUTRAL width-preserving genome can drop
a following edge that uniform passes (delay 1.0, width 2.25, next edge 3 later:
busy until t+3.25). This is deliberate physics, not a bug — evolution recovers
the edge by shortening the node's delay (any d <= gap - w) — but it means
neutral 'pulse_delay' starts measurably harder than 'uniform' on wide-pulse
banks, which matters when reading cross-model learning curves.
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass

from .hexgrid import hex_dirs

# Backward-compatible defaults. New runs carry an immutable PulseConfig instead
# of mutating these module values or relying on worker-process environments.
DELAY = 1.0   # fixed node propagation delay
WIDTH = 1.0   # output pulse width (leak rate)
COINC = 0.5   # excitatory coincidence window
TICK  = 1.0     # sampling period of the scoring layer (one target tick)


# The digital node-timing MODELS ('uniform' is the legacy abstraction; the other two
# are this project's variants that give pulse WIDTH a role instead of letting
# every node regenerate one fixed width — see the module docstring above and
# tests/test_pulse_models.py):
#   'uniform'       — every node emits width WIDTH after delay DELAY.
#   'evolved_width' — each node's emitted pulse width is genetic: an evolvable
#                     per-node-type (routing state) multiplier of WIDTH, supplied
#                     as a {cell: width} map. Delay stays DELAY.
#   'pulse_delay'   — historical identifier for evolved-delay, width-preserving
#                     edge transport: [t, t+w) -> [t+d_node, t+d_node+w).
# 'paper_analog' selects the analog Fig. 1 node engine (nv_evo/analog.py):
# charge/leak/comparator/hysteresis, from which coincidence width, output width
# and refractory EMERGE instead of being fixed constants. It is a distinct
# ENGINE, not a timing tweak of this one, but shares the model slot so the run
# pipeline selects it uniformly. Routing-only evolution (no width/delay vectors).
NODE_MODELS = ('uniform', 'evolved_width', 'pulse_delay', 'paper_analog')


# ── model families (reference plan's point 5) ───────────────────────────────────
# Readable family names for the four clearly-separated substrates, mapped to the
# internal node_model that drives them. The legacy identifiers are KEPT as the
# canonical values (every checkpoint and test still uses them), so these are
# additive aliases for UI/description, not a rename:
#   paper_event        — the current deterministic digital abstraction, retained
#                        unchanged for checkpoints and comparisons ('uniform').
#   paper_analog       — charge / leak / comparator / hysteresis analog node
#                        (nv_evo/analog.py); with tri3 tiles, the closest paper
#                        model available here, still using normalised constants.
#   waveform_transport — width-preserving evolved-delay transport ('pulse_delay'),
#                        explicitly a DIFFERENT substrate (not the Fig. 1 node).
#   extended_analog    — RESERVED: programmable capacitor weights, per-direction
#                        leak bias, bounded thresholds, each carrying an
#                        area/energy cost so evolution cannot buy unlimited analog
#                        precision for free. Not yet implemented.
MODEL_FAMILIES = {
    'paper_event':        'uniform',
    'paper_analog':       'paper_analog',
    'waveform_transport': 'pulse_delay',
    'extended_analog':    None,          # reserved — see above
}


def resolve_model_family(name):
    """Map a family name (or a raw node_model) to its node_model. Raises for the
    reserved-but-unimplemented families so a run can't silently select nothing."""
    if name in NODE_MODELS:
        return name
    if name in MODEL_FAMILIES:
        model = MODEL_FAMILIES[name]
        if model is None:
            raise NotImplementedError(
                "model family '%s' is reserved but not yet implemented" % name)
        return model
    raise ValueError('unknown model family or node_model: %r' % (name,))


@dataclass(frozen=True)
class PulseConfig:
    """Immutable paper-model timing parameters carried with one run."""
    delay: float = DELAY
    width: float = WIDTH
    coincidence: float = COINC
    event_cap: int = 2048
    # Node-timing variant (see NODE_MODELS). 'uniform' remains the legacy digital
    # behavior; old development configs are migrated by RunConfig.from_dict.
    model: str = 'uniform'
    # Fixed physical constants for the paper_analog model.  They live on the
    # run configuration (rather than being hidden AnalogConfig defaults) so a
    # checkpoint reproduces the same circuit.  Digital models ignore them.
    analog_threshold: float = 0.5
    analog_step: float = 0.34
    analog_tau_leak: float = 1.10
    analog_hysteresis: float = 0.08

    def __post_init__(self):
        analog_values = (self.analog_threshold, self.analog_step,
                         self.analog_tau_leak, self.analog_hysteresis)
        if (not all(math.isfinite(v) for v in
                    (self.delay, self.width, self.coincidence) + analog_values)
                or self.delay <= 0 or self.width <= 0 or self.coincidence < 0
                or self.event_cap < 1):
            raise ValueError('delay/width/event_cap must be positive and coincidence non-negative')
        if self.model not in NODE_MODELS:
            raise ValueError('model must be one of %s' % (NODE_MODELS,))
        if not (0 < self.analog_threshold < 1.0):
            raise ValueError('analog_threshold must lie strictly between 0 and 1')
        gap = 1.0 - self.analog_threshold
        if not (gap / 2.0 < self.analog_step < gap):
            raise ValueError('analog_step must let two terminal edges fire but not one')
        if self.analog_tau_leak <= 0 or self.analog_hysteresis < 0:
            raise ValueError('analog_tau_leak must be positive and analog_hysteresis non-negative')
        if self.analog_threshold + self.analog_hysteresis >= 1.0:
            raise ValueError('analog threshold plus hysteresis must remain below rest (1.0)')

_NEG = float('-inf')


class PulseSim:
    """Event-driven pulse simulation of one grown nervous net.

    Usage: sim = PulseSim(grid, routing); then once per tick
    ``state = sim.step({input_cell: bit, ...})`` which injects/extends input
    pulses, advances the event queue, and returns {cell: 0/1} — each wire
    sampled at the middle of the tick. ``sim.ever`` maps cells to whether
    their wire has pulsed at all (the combinational "did it fire" read-out).
    """

    def __init__(self, grid, routing, max_events=None, config=None, widths=None,
                 delays=None, sources=None):
        self.grid    = grid
        self.routing = routing
        self.config  = config or PulseConfig()
        # node-timing model, resolved once. widths {cell: pulse_width} is used
        # only by the 'evolved_width' model; absent/None cells fall back to
        # config.width, so an empty map reproduces 'uniform' exactly. delays is
        # the analogous per-cell map for evolved-delay width preservation.
        self._model  = self.config.model
        self._widths = widths or {}
        self._delays = delays or {}
        # src[v] = (s1, s2, si): the cells feeding v's E1 / E2 / I1.
        # watch[u] = cells that read u on an excitatory input.
        #
        # ``sources`` PRE-RESOLVES those feeder cells for every node, bypassing
        # the single-tile hex_dirs decode. The tri-tile substrate (tritile.py)
        # uses it: each hex tile becomes THREE sub-nodes whose inputs cross tile
        # boundaries, so a node key is no longer a bare (x,y) with hex_dirs
        # neighbours. When ``sources`` is None the single-circuit path below is
        # byte-identical to before, so every existing run is unaffected.
        self.src   = {}
        self.op    = {}                               # v -> 'and' | 'or'
        self.watch = {c: [] for c in grid}
        for v, entry in routing.items():
            if v not in grid:
                continue
            e1, e2, i1 = entry[0], entry[1], entry[2]
            self.op[v] = entry[3] if len(entry) > 3 else 'and'
            if sources is not None:
                s1, s2, si = sources.get(v, (None, None, None))
            else:
                nb = hex_dirs(*v)
                s1 = nb[e1] if e1 is not None else None
                s2 = nb[e2] if e2 is not None else None
                si = nb[i1] if i1 is not None else None
            self.src[v] = (s1, s2, si)
            for s in {s1, s2}:
                if s is not None and s in self.watch:
                    self.watch[s].append(v)
        self.pulse_start = {c: _NEG for c in grid}
        self.pulse_until = {c: _NEG for c in grid}
        self.refr_until  = {c: _NEG for c in grid}
        self.last_edge   = {c: {} for c in grid}      # v -> {source: edge time}
        self.ever        = {c: 0 for c in grid}
        # Raw leading-edge timestamps are the substrate's behavioural output.
        # Tick samples remain a playback convenience, not the source of truth
        # for event-based fitness.
        self.rise_times  = {c: [] for c in grid}
        # The COMPLETE output waveform per wire: [start, end] per pulse, in
        # time order. ``end`` is float('inf') while a width-preserving
        # aggregate is still high (closed when it falls). Rise-only scoring
        # keeps using rise_times; interval-aware scoring (real falls, real
        # widths) reads this instead of reconstructing falls from samples.
        self.pulse_intervals = {c: [] for c in grid}
        self.max_events  = None if max_events is None else max(1, int(max_events))
        self.event_count = 0
        self.overflow    = False
        self._heap = []                               # (time, seq, cell, pulse_end)
        self._seq  = 0
        # The width-preserving model transports drive rises and falls
        # independently. ``uniform`` and ``evolved_width`` never touch this
        # state and continue through the legacy interval heap above.
        self._edge_heap = []                          # (time, seq, kind, cell, drive)
        self._drive_seq = 0
        self._aggregate_seq = 0
        self._active_drives = {c: set() for c in grid}
        self._aggregate_id = {c: None for c in grid}
        self._drive_owner = {}                        # generated drive -> node
        self._busy_drive = {c: None for c in grid}
        self._followers = {}                          # source pulse -> group ids
        self._transport_groups = {}
        self._group_seq = 0
        self._or_group = {}                           # OR node -> open union group
        self._tick = 0                                # next tick index
        self._prev = {}                               # input cell -> previous bit

    # ── wires and events ─────────────────────────────────────────────────────

    def _high(self, c, t):
        # a routed direction may point at a dead / off-grid cell: no wire,
        # never high (its excitatory edge never comes; its veto never fires)
        return self.pulse_start.get(c, _NEG) <= t < self.pulse_until.get(c, _NEG)

    def _push(self, t, cell, end):
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, cell, end))

    def _run_until(self, t_end):
        if self._model == 'pulse_delay':
            self._run_width_preserving_until(t_end)
            return
        self._run_legacy_until(t_end)

    def _run_legacy_until(self, t_end):
        """Process events chronologically. Events sharing a timestamp are
        applied in two phases — first every wire rise, then every edge
        notification — so e.g. an inhibitory pulse arriving simultaneously
        with the excitatory edges reliably vetoes them."""
        while self._heap and self._heap[0][0] <= t_end:
            t = self._heap[0][0]
            batch = []
            while self._heap and self._heap[0][0] == t:
                batch.append(heapq.heappop(self._heap))
            new_edges = []
            for (_, _, cell, end) in batch:
                if self._high(cell, t):                       # wired-OR merge:
                    if end > self.pulse_until[cell]:          # already high ->
                        self.pulse_until[cell] = end          # extend, no edge
                        self.pulse_intervals[cell][-1][1] = float(end)
                else:
                    if (self.max_events is not None
                            and self.event_count >= self.max_events):
                        # A pathological oscillator must not make one genome
                        # monopolise evaluation.  Mark the run invalid and stop
                        # deterministically; scorers turn overflow into zero.
                        self.overflow = True
                        self._heap.clear()
                        return
                    self.pulse_start[cell] = t
                    self.pulse_until[cell] = end
                    self.ever[cell] = 1
                    self.rise_times[cell].append(float(t))
                    self.pulse_intervals[cell].append([float(t), float(end)])
                    self.event_count += 1
                    new_edges.append(cell)
            for u in new_edges:
                for v in self.watch[u]:
                    self._consider(v, u, t)

    def _new_drive(self, owner=None):
        self._drive_seq += 1
        drive = self._drive_seq
        if owner is not None:
            self._drive_owner[drive] = owner
        return drive

    def _edge_push(self, t, kind, cell, drive):
        self._seq += 1
        heapq.heappush(self._edge_heap,
                       (float(t), self._seq, kind, cell, drive))

    def _source_key(self, cell):
        pulse_id = self._aggregate_id.get(cell)
        return None if pulse_id is None else (cell, pulse_id)

    def _add_group_source(self, group_id, source_key):
        group = self._transport_groups.get(group_id)
        if group is None or source_key is None or source_key in group['pending']:
            return
        group['pending'].add(source_key)
        self._followers.setdefault(source_key, set()).add(group_id)

    def _accept_transport(self, v, source_keys, t, policy='all', or_group=False):
        """Start one generated drive and bind its fall to source pulse(s).

        ``policy='all'`` preserves a single pulse or an OR union (last carrier
        fall). ``policy='first'`` preserves simultaneous coincidence overlap
        (first carrier fall). Both edges receive the same selected node delay.
        """
        keys = {key for key in source_keys if key is not None}
        if not keys or self._busy_drive[v] is not None:
            return None
        drive = self._new_drive(owner=v)
        self._busy_drive[v] = drive
        self.refr_until[v] = float('inf')
        delay = self._delays.get(v, self.config.delay)
        self._edge_push(t + delay, 'rise', v, drive)
        self._group_seq += 1
        group_id = self._group_seq
        self._transport_groups[group_id] = {
            'dest': v, 'drive': drive, 'policy': policy, 'pending': set(),
            'delay': delay,
            'or_node': v if or_group else None,
        }
        for key in keys:
            self._add_group_source(group_id, key)
        if or_group:
            self._or_group[v] = group_id
        return group_id

    def _resolve_transport(self, group_id, source_fall):
        group = self._transport_groups.pop(group_id, None)
        if group is None:
            return
        end = float(source_fall) + group['delay']
        self._edge_push(end, 'fall', group['dest'], group['drive'])
        self.refr_until[group['dest']] = end
        or_node = group['or_node']
        if or_node is not None and self._or_group.get(or_node) == group_id:
            self._or_group.pop(or_node, None)

    def _source_fell(self, source_key, t):
        for group_id in tuple(self._followers.pop(source_key, ())):
            group = self._transport_groups.get(group_id)
            if group is None:
                continue
            group['pending'].discard(source_key)
            if group['policy'] == 'first' or not group['pending']:
                self._resolve_transport(group_id, t)

    def _record_width_preserving_rise(self, cell, t):
        if (self.max_events is not None
                and self.event_count >= self.max_events):
            self.overflow = True
            self._edge_heap.clear()
            self._heap.clear()
            return False
        self._aggregate_seq += 1
        self._aggregate_id[cell] = self._aggregate_seq
        self.pulse_start[cell] = t
        self.pulse_until[cell] = float('inf')
        self.ever[cell] = 1
        self.rise_times[cell].append(float(t))
        self.pulse_intervals[cell].append([float(t), float('inf')])
        self.event_count += 1
        return True

    def _notify_width_preserving_rises(self, new_rises, t):
        """Evaluate a complete simultaneous rise batch deterministically."""
        affected = {}
        for u in new_rises:
            for v in self.watch[u]:
                affected.setdefault(v, set()).add(u)

        for v in sorted(affected):
            if self._busy_drive[v] is not None:
                continue
            s1, s2, si = self.src[v]
            if si is not None and self._high(si, t):
                continue
            rising = affected[v]
            if s1 == s2:
                if s1 in rising:
                    self._accept_transport(v, [self._source_key(s1)], t)
                continue
            if self.op[v] == 'or':
                # If an accepted OR waveform is already open, this edge was
                # attached to it before source-fall resolution. Otherwise begin
                # one deterministic union of every excitatory source high now.
                if v not in self._or_group:
                    carriers = [self._source_key(s) for s in (s1, s2)
                                if s is not None and self._high(s, t)]
                    self._accept_transport(v, carriers, t, policy='all',
                                           or_group=True)
                continue

            # Coincidence: update all simultaneous edges before testing. A tie
            # transports their overlap (first fall); a unique later edge carries
            # its own width, exactly as the legacy leading-edge rule selects it.
            for source in rising:
                self.last_edge[v][source] = t
            tied = s1 in rising and s2 in rising
            if tied:
                self._accept_transport(
                    v, [self._source_key(s1), self._source_key(s2)], t,
                    policy='first')
                continue
            source = s1 if s1 in rising else s2
            other = s2 if source == s1 else s1
            if (source is not None and
                    t - self.last_edge[v].get(other, _NEG)
                    <= self.config.coincidence):
                self._accept_transport(v, [self._source_key(source)], t)

    def _run_width_preserving_until(self, t_end):
        """Transport aggregate waveform edges by one evolved delay per node.

        Each scheduled interval owns a drive identity. A wire rises only when
        its active-drive set changes from empty to non-empty and falls only when
        the set becomes empty. Applying a complete timestamp batch atomically
        makes overlapping/touching intervals a single physical waveform and
        makes simultaneous inhibition independent of submission order.
        """
        while self._edge_heap and self._edge_heap[0][0] <= t_end:
            t = self._edge_heap[0][0]
            batch = []
            while self._edge_heap and self._edge_heap[0][0] == t:
                batch.append(heapq.heappop(self._edge_heap))

            touched = {item[3] for item in batch}
            was_high = {cell: bool(self._active_drives[cell]) for cell in touched}
            for _, _, kind, cell, drive in batch:
                if kind == 'rise':
                    self._active_drives[cell].add(drive)
            for _, _, kind, cell, drive in batch:
                if kind != 'fall':
                    continue
                self._active_drives[cell].discard(drive)
                owner = self._drive_owner.pop(drive, None)
                if owner is not None and self._busy_drive.get(owner) == drive:
                    self._busy_drive[owner] = None

            new_rises = []
            fallen = []
            for cell in sorted(touched):
                is_high = bool(self._active_drives[cell])
                if not was_high[cell] and is_high:
                    if not self._record_width_preserving_rise(cell, t):
                        return
                    new_rises.append(cell)
                elif was_high[cell] and not is_high:
                    source_key = self._source_key(cell)
                    self.pulse_until[cell] = t
                    self.pulse_intervals[cell][-1][1] = float(t)
                    fallen.append(source_key)

            # Attach a newly-rising OR carrier before same-time source falls
            # resolve a group. Thus A ending exactly as B starts keeps the
            # transported OR waveform continuous, with no boundary glitch.
            for u in new_rises:
                source_key = self._source_key(u)
                for v in self.watch[u]:
                    if self.op.get(v) == 'or' and v in self._or_group:
                        self._add_group_source(self._or_group[v], source_key)
            for source_key in fallen:
                if source_key is not None:
                    self._source_fell(source_key, t)
                    self._aggregate_id[source_key[0]] = None

            self._notify_width_preserving_rises(new_rises, t)

    def advance_to(self, when):
        """Process queued events through an absolute physical time.

        ``step`` intentionally stops at the mid-tick display sample.  Temporal
        evaluation calls this once at the end of a run so late half-tick edges
        are retained in ``rise_times`` instead of being silently discarded.
        """
        self._run_until(float(when))

    def activity_at(self, when):
        """Return the wire levels at an already-processed physical time."""
        when = float(when)
        return {cell: (1 if self._high(cell, when) else 0)
                for cell in self.grid}

    def inject_pulse(self, cell, t, width=None):
        """Inject one input pulse of ``width`` (default WIDTH) onto ``cell``'s net
        at absolute time ``t``, wired-OR with any existing pulse. This is the
        continuous-time analogue of ``step``'s per-tick input injection, used by
        the async Interactive/Designer playback to place edges at real (possibly
        sub-tick) times. Events are queued; call ``advance_to`` to process them."""
        if cell in self.grid:
            w = self.config.width if width is None else float(width)
            start = float(t)
            if not math.isfinite(start) or not math.isfinite(w) or w <= 0:
                raise ValueError('pulse start must be finite and width must be positive')
            if self._model == 'pulse_delay':
                drive = self._new_drive()
                self._edge_push(start, 'rise', cell, drive)
                self._edge_push(start + w, 'fall', cell, drive)
            else:
                self._push(start, cell, start + w)

    def _consider(self, v, u, t):
        """Cell v sees a pulse edge from neighbour u at time t."""
        if t < self.refr_until[v]:
            return
        s1, s2, si = self.src[v]
        if si is not None and self._high(si, t):              # inhibitory veto
            return
        if s1 == s2:                                          # buffer: one edge
            trig = (u == s1)
        elif self.op[v] == 'or':                              # OR: either edge
            trig = (u == s1 or u == s2)                       # activates the node
        else:                                                 # coincidence: both
            self.last_edge[v][u] = t                          # edges within COINC
            other = s2 if u == s1 else s1
            trig = (t - self.last_edge[v].get(other, _NEG)) <= self.config.coincidence
        if trig:
            # This is deliberately the unchanged legacy path used only by
            # uniform and evolved_width. Width-preserving edge transport is
            # handled by _run_width_preserving_until instead.
            if self._model == 'evolved_width':
                width_v = self._widths.get(v, self.config.width)
            else:
                width_v = self.config.width
            delay_v = self.config.delay
            self.refr_until[v] = t + delay_v + width_v
            self._push(t + delay_v, v, t + delay_v + width_v)

    # ── the per-tick interface used by scoring / playback ────────────────────

    def step(self, input_vals):
        """Advance one tick. input_vals {cell: 0/1} drives the input nets:
        a 0->1 stream transition injects a pulse edge; consecutive 1s extend
        the same pulse (a held level is ONE long pulse, one edge). Returns
        {cell: 0/1} — every wire sampled mid-tick."""
        t0 = self._tick * TICK
        for c, b in input_vals.items():
            b = 1 if b else 0
            if b:
                if self._model == 'pulse_delay':
                    # Later asserted ticks add one touching drive interval. The
                    # active-drive union extends the same aggregate waveform and
                    # exactly matches one pre-injected held interval.
                    width = (TICK if self._prev.get(c, 0)
                             else max(self.config.width, TICK))
                    self.inject_pulse(c, t0, width)
                elif self._prev.get(c, 0) and self.pulse_until[c] >= t0:
                    extended = max(self.pulse_until[c], t0 + TICK)
                    if extended > self.pulse_until[c]:
                        self.pulse_until[c] = extended
                        if self.pulse_intervals[c]:
                            self.pulse_intervals[c][-1][1] = float(extended)
                    else:
                        self.pulse_until[c] = extended
                else:
                    self._push(t0, c, t0 + max(self.config.width, TICK))
            self._prev[c] = b
        sample_t = t0 + 0.5 * TICK
        self._run_until(sample_t)
        self._tick += 1
        return {c: (1 if self._high(c, sample_t) else 0) for c in self.grid}

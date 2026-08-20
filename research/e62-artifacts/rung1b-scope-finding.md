# E62 rung 1b — the wiring rung cannot ship a change, by construction

Recorded before spending timed legs. Source read at `ea683aae`, host measured
directly.

## The shippable knob is already at its maximum

The assignment permits exactly one shippable line from this rung:
`Qwen36MTPBlockSession.swift:213`, `wiredZHDefaultFraction = 1.0`.

That value is consumed at `:243`:

```swift
var target = Int(Double(active) * min(max(fraction, 0.0), 1.0))
```

The clamp is `min(max(fraction, 0.0), 1.0)`. The reachable set of
`wiredZHDefaultFraction` is therefore `[0.0, 1.0]`, and the shipped value
`1.0` is the **upper end of that set**. No value of `:213` can wire more than
the current build already wires. Every alternative value wires strictly less.

The clamp is not an accident. The doc comment at `:208-212` states the design
intent: capacity is the live post-warm footprint plus a small page-rounding
allowance, chosen so that "later scratch fails the fit test and stays on the
commit-free unwired path". Wiring past `active` would pull scratch into the
resident set, which the design deliberately avoids. Removing the clamp is an
edit to `:243`, which is not a permitted shippable line.

## Consequence: every outcome of an ON/OFF test leads to "nothing ships"

| Outcome on this host | Reading | Shippable change to `:213` |
|---|---|---|
| wiring ON clearly faster than OFF | mechanism is valuable, and `1.0` is already the maximum | none |
| ON indistinguishable from OFF | mechanism is inert, so scaling it down cannot help either | none |
| wiring ON clearly slower than OFF | reducing `fraction` could help | **not on this evidence** — see below |

Only the third row points at a ship, and that row is exactly the result a
48 GiB host is biased to produce for reasons that do not transfer.

## Why a "wiring hurts" result here would not transfer

Wiring pins roughly the live post-warm footprint. The measured worker peak RSS
on this host is 15.101 GB, so the wired target is about 31 % of this machine's
48 GiB. On the ranked 96 GiB M5 the same target is about 16 % of physical
memory. Pinning a third of a machine's RAM is a materially harsher regime than
pinning a sixth of it, and it is the regime most likely to manufacture a
spurious "wiring hurts" reading through host-level memory pressure that the
ranked runner never sees.

So the one branch that could motivate lowering `:213` is also the one branch
this host cannot measure credibly. I will not propose a fraction reduction on
48 GiB evidence.

## What I am doing instead

I am not spending eight 512-token legs (~35 min) on a rung whose every branch
ends in "nothing ships". Replaced with:

1. **One cheap 64-token leg on the `wired` throwaway arm.** It proves the file
   sink reads, and it records `active`, `applied` and `maxrec` on this host so
   the mechanism's size is documented rather than assumed.
2. **`wired_residency_active` is still reported on every timed leg**, as
   required. On the stock arm on this 48 GiB host the gate at `:225` is false,
   so the honest value is `false` on every stock leg, and the one-leg probe
   above is what substantiates that rather than merely asserting it.
3. If rung 1 finds a geometry winner, I re-check that winner once with wiring
   forced ON, so a shippable geometry claim is not resting only on the
   wiring-off regime. See the caveat below.

## Caveat this creates for rungs 1-3, stated up front

`hw.memsize` on this host is 51539607552 bytes, exactly 48.0 GiB. The gate at
`:225` needs 96 GiB, so **wiring is off for every stock timed leg here, while
it is on for the ranked M5**. The buffer-geometry ladder is therefore measured
in a different residency regime than the ranked runner uses. Command-buffer
geometry and resident-set behaviour are not obviously independent, so a
geometry effect measured with wiring off is directional for the ranked host,
not a transfer guarantee. That is the reason for step 3 above.

## Report-only observation about the gate constant

The gate is `physicalMemory >= (UInt64(96) << 30)` = 103079215104 bytes. A
machine advertised as "96GB" passes only if the vendor figure is binary
(96 x 2^30). If the ranked host ever reports decimal 96 x 10^9 = 89.4 GiB, or
reports any reserved-memory deduction, the gate fails silently and the wiring
optimisation is dormant on the ranked runner with no error and no log line.
The margin is zero.

This is recorded as an observation only. The assignment forbids shipping any
change to the 96 GiB gates, so I am not proposing one, and gate lowering stays
throwaway-local.

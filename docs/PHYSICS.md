# Drilling generator: physics, signatures and assumptions

Code: `generators/drilling.py`. Profile: `profiles/drilling.py`. All numbers below are the values
used in the code at full event size (`scale = 1.0`). Events are drawn with scale 0.15 to 1.0
(log-uniform), so most events are smaller.


## About the sources

I did not check any of these against the web in this session: the references are cited **from memory**
and were not re-opened. I give author, title and year only, no chapter or page numbers, because I
am not sure of them. Verify each one before citing it elsewhere. Where I could not name a source for
an item I say "no source".

- **[Bourgoyne]** Bourgoyne, Millheim, Chenevert, Young, *Applied Drilling Engineering*, SPE Textbook Series Vol. 2, 1986.
- **[Grace]** R. D. Grace, *Blowout and Well Control Handbook*, Gulf Professional Publishing, 2003.
- **[Mitchell]** Mitchell and Miska, *Fundamentals of Drilling Engineering*, SPE Textbook Series Vol. 12, 2011.
- **[SLB]** Schlumberger Oilfield Glossary (glossary.slb.com), entries such as "kick", "lost circulation", "stuck pipe", "washout". Not checked that every term I attribute to it has an entry (in particular "pack-off").
- **[Teale]** R. Teale, "The concept of specific energy in rock drilling", *Int. J. Rock Mech. Min. Sci.*, 1965 (mechanical specific energy).

## Sampling and noise

- One row per minute per well. NEEDS REVIEW: real rig data is usually 1 s to 10 s; 1 minute hides fast transients (e.g. pressure spikes). No source; chosen to keep files small.
- Sensor noise is Gaussian and independent per channel (WOB 0.8 kN, RPM 0.7, torque 0.12 kN·m, ROP 0.4 m/h, SPP 0.4 bar, flow-in 0.4 %, flow-out 1.2 %, pit volume 0.03 m³, mud weight 0.002 sg, gas 0.3 units). WOB, torque, ROP, gas also carry AR(1) process noise. NEEDS REVIEW: all values; no source. Real sensors have spikes, dropouts, and non-Gaussian errors that are not modelled.
- Flow-out has a per-well calibration offset of up to ±1 %. NEEDS REVIEW (paddle-type flow-out sensors are generally considered less accurate than pump-stroke flow-in; I believe this but have no source to point to).

## Normal behaviour

| Item | Model | Status |
|---|---|---|
| Depth progress | Depth integrates ROP while drilling. Start depth 800 to 3000 m, bit never moves up (no tripping, no reaming) | Simplification |
| Stands and connections | Every 28.5 m (three ~9.5 m joints) drilling stops for a connection of 4 to 9 min: pumps drop to 40 % then 0 then 50 %, rotation and WOB go to 0, ROP 0 | Stand length is common practice [Mitchell]; durations NEEDS REVIEW (no source) |
| Formation changes | Layers 80 to 400 m thick, each with its own drillability (0.5 to 1.8 multiplier on ROP), torque factor (0.9 to 1.3), and background gas (3 to 40 units) | NEEDS REVIEW: invented values; real formations also change pore pressure and lithology-related signals not modelled |
| ROP | `rop = k * drillability * (WOB/100)^0.8 * (RPM/100)^0.4 * noise`, k 18 to 32 m/h | Power-law form is in the spirit of the bit-weight / rotary-speed terms in [Bourgoyne]; the exponents 0.8 and 0.4 are my choice. NEEDS REVIEW |
| WOB / RPM | WOB set point 60 to 160 kN with AR(1) variation; RPM set point 80 to 140, re-drawn each stand; both ramp up over 3 min after a connection | NEEDS REVIEW (no source) |
| Torque | `(1 + 0.0022*depth + 0.06*WOB*formation_factor) * noise` kN·m (friction grows with depth, bit torque grows with WOB) | Qualitative shape only. NEEDS REVIEW (no source for coefficients) |
| Standpipe pressure | `(flow/2000)^1.8 * (mud_weight/1.5) * (40 + 0.03*depth)` bar, first-order lag 1.5 min | Exponent ~1.8 for turbulent pipe flow is textbook hydraulics [Bourgoyne, Mitchell]; coefficients invented, ignore bit nozzle and the real pipe/annulus split. NEEDS REVIEW |
| Flow in | Pump rate 1800 to 3000 L/min per well, from pump strokes, so very small noise | NEEDS REVIEW |
| Flow out | First-order lag of flow in (time constant 1.5 to 3 min) | NEEDS REVIEW: real lag is the annulus transit time (minutes, depends on depth and rate) and is not a single exponential |
| Pit volume | Mass balance: `d(pit) = (flow_out - flow_in)/1000` m³ per minute. This gives a small pit **gain at each connection** (return flow keeps coming after the pumps stop) | Mass balance is exact in the code (tested). Connection flow-back being a normal signature is standard practice [Grace]. Magnitude NEEDS REVIEW |
| Pit transfers | About 2 per day, ±0.5 to 3 m³ over 3 min, labelled Normal. They move pit volume with **no** flow imbalance | Deliberate false-alarm source. NEEDS REVIEW (no source for frequency or size) |
| Pit management | Pit volume relaxes to its start value with a 6 h time constant (crews top up or dump mud) | Invented so that losses and gains do not stay forever. NEEDS REVIEW |
| Mud weight | 1.05 + 0.00018 * depth (+ per-well offset), re-set each stand, rounded to 0.01 | Mud weight rising with depth is normal practice; the numbers are invented. NEEDS REVIEW |
| Gas | Formation background gas plus a connection-gas bump (+5 to 25 units) after each connection, both delayed by a surface lag of `0.017*depth` min (5 to 60 min) | Background and connection gas are standard mud-logging concepts; I have no source to name for the lag formula. NEEDS REVIEW |

Not modelled: tripping, reaming, circulating off bottom, pump changes, drilling breaks that are not
kicks, mud-weight changes mid-stand, hole-size effects, directional drilling (no hook load, no
inclination, so no drag-based stuck-pipe signature), temperature, pore/fracture pressure.

## Anomaly signatures

Each event has a linear ramp up (gradual onset), a hold, and a linear ramp down. Ground truth labels
a row as anomalous from the first row of the ramp (when the effect is still tiny) until the end of
the ramp down. Events may overlap; effects then add and the label with the largest current
`scale * envelope` is the primary label (`anomaly_type`), all active ones are in `active_anomalies`.

Ramp / hold / release ranges (minutes): Kick 10 to 45 / 10 to 40 / 5 to 20; Lost Circulation 5 to 40 / 20 to 120 / 10 to 40; Stuck Pipe 10 to 60 / 20 to 90 / 3 to 10; Washout 60 to 240 / 30 to 180 / 1 to 3; Pack-off 5 to 30 / 10 to 60 / 5 to 20; Sensor Drift 120 to 480 / 0 to 60 / 1 to 3. NEEDS REVIEW: all durations (no source).

After an event ends the effect is simply removed. No operator response (shut-in, weighting up,
pulling out) is modelled.

### Kick (formation fluid influx)

| Channel | Signature at scale 1.0 | Source / status |
|---|---|---|
| flow_out | Up to +250 L/min above flow in, **also with the pumps off** (so it shows during connections) | Flow-out increase with constant pump rate, and flow while the pumps are off ("flow check"), are the classic primary indicators [Grace]; [SLB] "kick". Magnitude NEEDS REVIEW |
| pit_volume | Rises through mass balance (a few m³ over an hour) | Pit gain is the other classic primary indicator [Grace]. Size NEEDS REVIEW |
| spp | −6 % | Lighter fluid in the annulus lowering pump pressure is commonly listed as a secondary indicator [Grace]; size and even sign are situation-dependent. NEEDS REVIEW |
| rop | +20 % ("drilling break") | A drilling break is commonly listed as a warning sign [Grace]; but it is caused by the formation, not by the kick, so coupling it to the event is a simplification. NEEDS REVIEW |
| gas | +120 units, delayed by the surface lag | Gas-cut mud as a late indicator [Grace]; amount and shape NEEDS REVIEW |
| others | unchanged | |

Assumption: the influx is instant liquid-like displacement at the surface (no gas expansion, so no
accelerating flow-out near the surface).

### Lost circulation

| Channel | Signature at scale 1.0 | Source / status |
|---|---|---|
| flow_out | Reduced by up to 600 L/min, proportional to the current pump rate (no loss with pumps off) | Reduced returns are the defining sign [SLB] "lost circulation", [Bourgoyne]. Proportionality to pump rate is my assumption. NEEDS REVIEW |
| pit_volume | Falls through mass balance | Standard [Bourgoyne]. |
| spp | −10 % | Pressure often decreases with losses, but not always. NEEDS REVIEW |
| others | unchanged | |

Not modelled: loss of hydrostatic head leading to a kick (ballooning, "kick and loss"), except by
overlapping random events.

### Stuck pipe

| Channel | Signature at scale 1.0 | Source / status |
|---|---|---|
| rop | −95 % (depth progress stops) | Obvious consequence; size NEEDS REVIEW |
| torque | +80 % plus erratic spikes (extra noise scaled by severity) | Rising and erratic torque as a pre-sticking or sticking sign is standard [Mitchell] (stated from memory; no chapter). Size NEEDS REVIEW |
| rpm | −40 % (rotary slows against the load) | NEEDS REVIEW; no source |
| spp, flows, pit | unchanged | Consistent with differential sticking, where circulation continues. Mechanical sticking with restricted circulation is the pack-off case below. NEEDS REVIEW |

Not modelled: hook load / overpull and drag, which are the primary stuck-pipe indicators in practice
and are not in the channel list. This is the biggest gap.

### Washout (hole in the drill string)

| Channel | Signature at scale 1.0 | Source / status |
|---|---|---|
| spp | −25 %, growing slowly over 1 to 4 h (gradual by design: erosion widens the hole) | A pressure drop at constant pump rate is the standard sign [SLB] "washout" (stated from memory). Rate and size NEEDS REVIEW |
| rop | −15 % (less bit hydraulic energy) | NEEDS REVIEW; no source |
| flow in / flow out / pit | unchanged | Fluid leaves the string and returns through the annulus. NEEDS REVIEW |

Not modelled: pump-stroke increase when the driller tries to compensate; the string failing
(twist-off). The word "washout" is also used for hole enlargement (wellbore washout); here it means
a string washout only.

### Pack-off

| Channel | Signature at scale 1.0 | Source / status |
|---|---|---|
| spp | +35 % | Rising pump pressure with a blocked annulus is the usual description [SLB] (I am not sure "pack-off" has its own entry; verify). Size NEEDS REVIEW |
| flow_out | −20 % of the lagged flow (restricted returns). Pit volume falls through mass balance | NEEDS REVIEW: where the missing fluid goes (formation, compression) is not modelled; in the data it looks like a loss, and what separates it from Lost Circulation is that pump pressure rises instead of falling |
| torque | +30 % with erratic spikes | NEEDS REVIEW; no source |
| rop | −40 % 

### Sensor drift

| Channel | Signature | Source / status |
|---|---|---|
| One random channel of flow_out, flow_in, pit_volume, spp, wob, mud_weight | A bias that grows **linearly** over 2 to 8 h (plus up to 1 h hold) to `scale * span`, sign random, then resets to zero ("recalibrated"). Spans: flow 150 L/min, pit 3 m³, SPP 15 bar, WOB 15 kN, mud weight 0.04 sg | Drift as slow linear bias is a common modelling assumption; no source for these spans. NEEDS REVIEW |
| all others | unchanged, including the true physics | Tested: only the drifting channel differs from the clean run |

This is deliberately hard: drift on flow_out or pit_volume looks like a kick or a loss in the raw
channel. What distinguishes it is that no other channel agrees (no SPP, ROP or gas change).
Large drifts can push readings out of physical range (e.g. negative WOB).

## Derived physics features (`profiles/drilling.py`)

| Feature | Definition | Status |
|---|---|---|
| `flow_delta_lpm` | flow_out − flow_in, 5-min trailing mean | Standard surveillance signal [Grace] |
| `pit_rate_m3ph` | pit volume change over 10 min, per hour | Standard surveillance signal [Grace] |
| `depth_progress_mph` | depth change over 10 min, per hour | Simple; stalls when stuck |
| `spp_norm` | spp / (flow_in/2000)^1.8, undefined below 500 L/min | Same hydraulic assumption as the generator, so it will look better on this data than on real data. NEEDS REVIEW |
| `mse_mpa` | `WOB/A + 2π·RPM·T/(A·ROP)`, A = 0.0366 m² (8.5 in bit assumed) | Mechanical specific energy [Teale]. Bit size is an assumption. NEEDS REVIEW |

All are past-only (diff or trailing window), so a feature at time t never uses later rows (tested).

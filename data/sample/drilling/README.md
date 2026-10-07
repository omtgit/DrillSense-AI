# Drilling sample (synthetic)

`drilling_data.csv`: 3 wells x 1 day at 1-minute spacing (4,320 rows), from
`python scripts/generate_drilling_data.py --sample` (seed 11, 4 events/day). It contains all six
anomaly types and at least one overlapping pair. `drilling_events.csv` lists the injected events.

Channels: `depth_m, wob_kn, rpm, torque_knm, rop_mph, spp_bar, flow_in_lpm, flow_out_lpm,
pit_volume_m3, mud_weight_sg, gas_units`, plus `rig_state` (drilling/connection).

Caveats:
- Fully synthetic and physically motivated, not field data. The signatures are my assumptions;
  see `docs/PHYSICS.md` (many items are marked NEEDS REVIEW).
- `anomaly_type`, `anomaly_flag`, `active_anomalies`, `event_scale`, `severity`, `priority_score` and
  `recommended_response` are generator ground truth (the last three are a lookup by type), not model output.
- The Streamlit app does not read this folder yet: it still expects the production-style columns.

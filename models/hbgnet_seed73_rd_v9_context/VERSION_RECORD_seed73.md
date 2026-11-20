# Seed73 rd_v9_context deployment record

Package: `/root/wyk5/RGVNet/models/hbgnet_seed73_rd_v9_context`
Run: `hbgnet_protocol_b_seed73_rd_v9_context`
Seed: 73
Physical GPU: 2

Derived from the validated v8 seed17/v8 seed73 static implementations. The original
packages are untouched. Only experiment identity, seed guards, package/pretraining
paths and the corresponding verification identities were adapted, plus the additive
v9 context residual and its diagnostics.

- v9 scoring: `tools/region_discrimination_v9.py` subclasses
  `tools/region_discrimination_v8_grok_fullloss.RegionCandidateVerifier`, so the
  fill (78 channels) and relational cut (125 channels) paths, `candidate_masks`,
  the strict `logit > 0` gate, `hard_keep`, `gated_region` and the ordinary
  present-direction BCE are the identical v8 objects. `tools/region_discrimination_v8_grok_fullloss.py`
  is therefore a hashed external dependency of this package as well.
- v9 addition: 8-channel detached-F64 projection, 14-channel center block, 117-channel
  support block (3 masks x 3 scales x 13), 131 channels per direction, one 16-channel
  depthwise residual encoder per direction with a zero-initialised 1x1 head.
- Spec change vs the design draft: descriptor 126 -> 131 and context parameters
  4,994 -> 5,154, because the center block now carries the four center margins and
  the foreground edge probability so that the control and the trained variant share
  an identical center (plus 5 channels x 16 stem weights per direction).  Approved
  by the main agent; not reverted to 126.
- Aggregation guard: persistent integer buffer `aggregation_code`
  (multi_scale_support 0 / center_only 1) validated in `_load_from_state_dict`
  before any tensor copy, so a cross-variant state load raises instead of silently
  loading.  `evaluate.py`'s checkpoint finiteness scan skips `isfinite` on integer
  buffers and still rejects non-tensor entries.
- Calibration coefficients are validated after the cast to the logits dtype
  (half underflow to 0 and overflow to inf are rejected) and a non-finite
  calibrated score raises.
- Parameter counts: v8 verifier 6074, context 5154, v9 verifier 11228, Field 30468959.
- Center-only control: same skeleton and nominal parameter budget without dead zero
  inputs; its three scale slots duplicate the center statistics, so it is a
  1x1-aggregation ablation and not equal-effective-capacity evidence. Not trained here.
- Initialization: official PVT-v2-B2 only; no v8/SR/RD checkpoint is loaded or resumed.
- Recipe: 40 epochs, batch 8, accumulation 2, AMP off, Adam 1e-4, CosineAnnealingLR,
  validation every 2 epochs plus the final 5, selection 0.7 IoU + 0.3 BF1 unchanged.
- No automatic formal test; no threshold tuning; no positive-class reweighting.
- Manifest hashes are regenerated from the written bytes; engineering verification,
  the GPU smoke and the startup audit are executed anew for this seed.

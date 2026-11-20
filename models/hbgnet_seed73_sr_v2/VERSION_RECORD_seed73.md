# Seed73 deployment record

Created UTC: 2026-10-02T15:23:40.009907+00:00

Package: `/root/wyk5/RGVNet/models/hbgnet_seed73_sr_v2`
Run: `hbgnet_protocol_b_seed73_sr_v2`
Seed: 73
Physical GPU: 1

Derived from the validated `/root/wyk5/RGVNet/models/hbgnet_seed17_sr_v2` static implementation. The original package is untouched.
Only experiment identity, seed guards, package/pretraining paths and corresponding verification identities were adapted.
Model, losses, data loader, metrics, state recovery and shared SR/RD tools retain their source SHA-256.
Baseline control-path portability changes do not alter training code or the Protocol B recipe.
Manifest hashes are regenerated. Engineering verification is executed anew; no seed17 report is reused.
Training starts from official pretraining with fresh optimizer/RNG state. No automatic formal test.

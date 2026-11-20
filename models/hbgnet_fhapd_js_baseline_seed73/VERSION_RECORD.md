# FHAPD_TRAIN_SEED73_V1_20261005

# hbgnet_fhapd_js_baseline_seed73

Dataset FHAPD_JS; seed73; physical GPU2; model baseline.

Runtime package: `/root/wyk5/RGVNet/models/hbgnet_fhapd_js_baseline_seed73`. Local backup: `FHAPD_JS/01_baseline_hbgnet_seed73_gpu2`.

This is a new training-only Protocol B compatible run. Frozen FHAPD spatial split; 256 RGB, ImageNet normalization, H/V flips and rot90 each 0.5; batch8, accumulation2, AMP off, Adam1e-4/wd0, 40epochs, Cosine T_max1e10/eta_min1e-5. Validation at epoch1/every2/final5. Best score 0.7 IoU+0.3 BF1; no AI4Boundaries IoU floor or constrained checkpoint.

Official LossF([1,1,1]): raw Region BCEDice + two-class Boundary NLL + Distance MSE.

Model/loss/backbone sources are copied unchanged. Baseline and combined have distinct supervision; combined raw is not an independently trained baseline. Official PVT-v2-B2 checkpoint only, never resume an AI4 trained model. Per-run manifest and acceptance receipts are regenerated. GPU smoke uses a disposable model and 4 batch8 train batches/2 Adam updates, plus 64 real val samples; formal training constructs a fresh model and optimizer.

Strict sigmoid>0.5 for FHAPD validation; boundary tolerance3 pixels. GSD unknown, not verified3m; different parent geography unknown. Canny/Distance targets derived from 0/1 mask. Global micro metrics and predicted-object area weighting; object metrics undefined without prediction objects are null. Undefined diagnostic rates are null with counts; TensorBoard omits null fields.

Data adapter remains immutable. New config/data_contract fixes regional manifest/counts/digests. Engineering does not open test images/labels; test manifest text and DATA_MANIFEST metadata are read only for identity. `evaluate.py` is gated in this training-only version; formal test requires separate authorization and a separately frozen evaluation package using this run best.pt.

Startup audit requires 5 losses/2 optimizer updates/5 model forwards (plus5 SR and5 RD forwards for combined), live process and no exit receipt. Instrumentation is removed after startup; no continuous monitoring or automatic test. Version record and recoverable source backups precede formal launch.

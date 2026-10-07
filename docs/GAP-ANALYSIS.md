# Gap analysis: what separated us from the top 10

Written on the last night of Phase 1 (30 Sep 2026, CDT) and updated on 1 Oct with our final scores and the final
leaderboard snapshot (about 7:40 AM CDT). The inputs are our own measurements and the public
leaderboard. Nothing here uses or infers validation data.

## The exchange rate between compute and score

On 24 Sep we ran one training run for 2,288 s instead of 1,788 s. That is +28% compute, and the run was not
submittable. It improved val_bpb by 0.0155. A log fit gives:

```
delta_bpb ~= 0.063 * ln(effective compute ratio)
```

The fit's own slope near a ratio of 1 is about 0.00063 bpb per 1%. The per-step rate we measured directly on two
chips is a little lower, about 0.00057 bpb per 1% more optimizer steps, and the simulator's quality model uses that
(`d bpb / d ln steps` about -0.057 at 2,300 steps).

## What each target needed

Starting from our best official score, 0.96136 (K82s4), with `ratio = exp(gap / 0.063)`:

| target | gap | effective compute needed |
|---|---|---|
| #10 on the board, 0.9554 | 0.0060 | about +10% |
| 0.9540 | 0.0074 | about +12% |
| #1, 0.9328 | 0.0286 | about +57% |

Recipe changes on the final day were worth about 0.0005 each, and the late search over settings was exhausted (see
`FINDINGS.md`, section 5). +10% compute was not going to come from settings.

## Where the compute goes

- **Our utilisation.** About 285k tokens/s on a 124M-parameter model: roughly 30% MFU, with about 19% of wall
  time idle on host dispatch.
- **Our throughput wins all removed eager dispatch.** Scalar stack -9.1% step time, fused cross-entropy -2.8%,
  Muon sharding -2.6%, optimizer fusion about -0.7%.
- **What throughput would buy.** At an equal recipe, +15-23% tokens/s is worth roughly 0.009-0.013 bpb by the
  exchange rate above: more than our whole gap to #10. We believe kernel throughput (NKI) is the way to get it.
- **Architecture changes that save FLOPs lost for us.** Attention in only a subset of layers and U-Net skips
  both lost. With our kernels the FLOPs they save were cheap, while the capacity they gave up was expensive.
  Faster kernels change that trade.

## Conclusion for Phase 2

1. **Kernel throughput is the dominant lever.** Write NKI kernels for attention, the MLP (fused LeakyReLU^2 plus
   both projections) and the Muon/AdamW updates. +15-30% tokens/s is worth 0.009-0.017.
2. **Then grow the model into the freed time.** Depth 10 and 2 KV heads lost only because they cost steps.
3. **Keep the data-efficiency stack** on top: row pool (-0.0023), EMA blend (about -0.0005), EMA prewarm (about
   -0.0003), salt selection (sd about 0.0004).
4. **Price each kernel before building it.** A 60-step chip screen gives its step time, the exchange rate turns that
   into bpb, and a cold rehearsal confirms it (`EXACT-ORACLE.md`).

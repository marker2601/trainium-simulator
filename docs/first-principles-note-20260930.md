# Where K63 is still leaving score on the table: a first-principles note

> Written for the team on 30 Sep 2026, edited for a public reader (private file references replaced,
> times in CDT). What happened to each idea is in "Outcome" at the end.

30 Sep 2026. Skeptic + author pass over 12 candidate ideas for GPU round 2 (strict L40S, 2314 equal steps, 3 seeds).

## Summary

Our 30-minute run is already close to its best settings for learning rate, cooldown and time allocation. The logs say about 0.0002-0.0005 bpb is left there, not 0.002. The biggest effect in 430 harvested runs is not a setting. It is a coin flip in warm-up steps 3-10. About 40% of seeds take a "spike" path that costs +0.003 to +0.0045 bpb for good. That coin flip has also contaminated some of tonight's GPU verdicts. The chip reservations ended at 6:30 AM CDT, so nothing could be priced on Trainium before the next upload. That changes the rules. We should favour changes that add no big-tensor ops, or whose speed the virtual clock can already price. Four arms are worth GPU time. (1) XSHIFT: attention that skips a token's own position in blocks 6-8, at almost no op cost. (2) A late switch to a 2x batch inside the cooldown. (3) A gentler AdamW start that should remove the warm-up coin flip. (4) A bold XSHIFT arm across all 9 layers. None of them is likely to reach -0.002 alone. XSHIFT is the only one with a real chance of -0.001 or better.

## What the logs say (with sources)

1. **Warm-up coin flip is the biggest effect.** Good-path seeds (73, 67, 58, 421 on chip C) have raw loss about 13.95 at step 4. The loss then falls steadily. Spike-path seeds (401, 409, 419 on C; 97, 613, 619, 631, 643, 647 on D) drop faster to step 4, at 13.3-13.6. The loss then rises about 0.7 nats over steps 5-10. Final cost: +0.0032 (K60, chip C, n=4 vs 3) and +0.0044 (chip D, n=8 vs 6). A paired curve (s419 minus s58) stays at about +0.007 nats from step 1100 to the end. The cooldown does not repair it. The AdamW groups drive the fast drop: AdamW LR x1.2 moves step-4 loss 13.94 to 13.59, while Muon x1.15 only moves it to 13.86. Source: the harvested chip C/D train logs (not published; three reference runs are in `research/sim-data/chipC/`).
2. **Hardware can flip the path.** Seed 58 is on the good path on the chip but looks like a spike seed on the GPU. Its level gap is +0.0023 to +0.0032, against +0.0000 to +0.0006 for s73 and s67. So any change that alters steps 0-19 (all architecture changes do) re-rolls the coin for that seed.
3. **Some closures are probably coin flips, not effects.** Three c_proj 0.7 pairs on spike-path seeds were all about +0.004. On good-path seeds it averaged -0.0002. The CFC_INIT 0.7 (+0.0043) and wu40d (+0.0033, final train loss +0.0066 nats) GPU losses are also spike-sized. Source: the campaign ledger, 30 Sep 5:30 and 7:45 AM CDT entries.
4. **Official runs follow the chip path.** 23 official offsets run from +0.0053 to +0.0077. Since K50 the range is +0.0061 to +0.0072. None has a spike-sized outlier. So the upload seed s73 is safe as long as steps 0-19 do not change. But no change to steps 0-19 can be checked on a chip before the deadline. Source: `research/sim-data/official-uploads.csv`.
5. **LR, cooldown and time allocation are converged.** The lower early LR (0.35/0.6) loses training loss for most of the run and only wins in the last 15% of the cooldown (net -0.0001 to -0.0004). The strict-GPU cooldown fit peaks at 0.67, just -0.00002 better than 0.7. Scalar retunes have won 0 of 61 knob keys. Source: the campaign knob summary (not published).
6. **Every quality win of 0.0013 or more since 23 Sep was structural and cost at most about 1% step time.** Examples: MTP -0.0043, batch warm-up -0.0019, key offset -0.0013, LeakyReLU^2-fnm -0.0017, AS-a -0.0015. Several structures were closed on price, not on quality. XSA was about -0.0007 to -0.0018 per step but lost on its fp32 reductions. XSA on the late layers never had a full-length run.
7. **The late batch was set under old costs.** The last 1,562 of 2,385 updates run at 262k tokens. Today each update has a fixed cost of about 0.08 s (fit: 0.0805 + 0.2122 x k seconds, from the chip C run table / C43). Doubling to k=8 gives about +4.5% tokens per second in that phase.
8. **Correction to the regime brief.** K63 has 4 query heads of 256 and 1 KV head (C43 log `n_head: 4`, `n_kv_heads: 1`), not 3 q heads. Value embeddings are 1024 wide, on layers 0 and 8. They turn off the GQA fast path in those two layers (train.py L1565).

## Ranked interventions (net value = gain - 0.00057 x step cost %)

| # | Idea | Step cost | Expected gain | Net | Verdict |
|---|------|-----------|---------------|-----|---------|
| 1 | XSHIFT-AS (blocks 6-8) | ~0.3% (unpriced) | -0.0010 | -0.0008 | Run, 3 seeds |
| 2 | Late k=8 batch in the cooldown | 0 (priced by virtual clock) | -0.0006 | -0.0006 | Run s73, then 2 seeds |
| 3 | AdamW warm-up immunisation (prewarm / pow2) | 0 | ~0 on s73; -0.0014 per spike seed; about -0.0004 risk value for the next upload | -0.0004 | 60-step screens first |
| 4 | XSHIFT-ALL (all 9 layers) | ~0.8% (unpriced) | -0.0012 | -0.0007 | s73 only, high variance |
| 5 | Narrow VE (256 wide, U-shape) | maybe -0.5% (unpriced) | -0.0005 at equal steps | ? | Park: needs a chip price |
| 6 | KO2 head-restricted 2-gram | ~0.25% | -0.0003 | -0.0002 | Park |
| 7 | Split-band 2-gram KO (k only) | ~0.2% | -0.0003 | -0.0002 | Park |
| 8 | AS-batched attention (exact math) | maybe -1% | 0 quality | ? | Park: throughput only, no chip |
| 9 | AS-fused QKV (exact math) | maybe -0.8% | 0 quality | ? | Park: subsumed by #8 |
| 10 | Row-start sink key via KO pad | 0 | -0.0002 | -0.0002 | Drop; run the free diagnostic only |
| 11 | Cross-layer KV interleave | ~0.3% | ~0 | + | Drop |
| 12 | Split warm-up (quadratic AdamW) | 0 | - | - | Merged into #3 as its pow2 arm |

### 1. XSHIFT-AS: skip the token's own position in blocks 6-8

- **Hypothesis.** Blocks 6-8 read the block-5 snapshot. Their output is added to a residual that already holds the token itself. Attention to position t itself is redundant there. The literature's XSA gain comes from removing self-information. It lost here only on the price of its fp32 reductions.
- **Change.** After RoPE+norm and the existing KO `where`, shift all of k by one position with `cat(k[:, :1], k[:, :-1])`. Shift v the same way, after the VE add. Query t then sees only positions up to t-1, with correct RoPE angles. The KO band keeps its meaning (it now matches k_{s-1} and copies v_s for s up to t-1). Flag `FF_XSHIFT_LAYERS=6,7,8`. It is op-identical when off.
- **Skeptic notes.** This removes the diagonal from attention. XSA instead projects the self-value direction out of the output. They are related, not the same. With the diagonal gone, a query has no "attend to myself" no-op. Position 0 (a duplicate of token 0) becomes the only fixed sink. The extra ops are cats over 0.25C (1C for v in layer 8, which has VE). KO's own cat plus `where` measured step-free, but a shift has never been priced in this exact form, and no chip is left to price it.
- **Cost / gain.** About 0.3%. Expected -0.0006 to -0.0015 at equal steps.
- **Test.** Strict GPU, seeds 73 and 67, plus a GPU-verified good-path seed instead of 58. Controls are the existing sk63 runs. Apply a step-10 gate: if the arm's logged loss at step 10 is on a different path from its control (> 13.4 vs < 13.3), the pair is void. **Adopt only if the 3-seed mean is -0.0010 or better, 3/3 are negative, and the paired train diff has levelled off by steps 1100-1300.** That bar covers the unpriced step cost.

### 2. Late k=8 batch inside the cooldown

- **Hypothesis.** The per-update fixed cost is 8.7% of a k=4 step. Doubling the batch from 0.5 of charged time gives +4.5% tokens per second for the rest of the run. That is about +2.3% more tokens in total. Late in training, a bigger batch costs the least. It touches nothing before step 1100, so it cannot re-roll the warm-up coin. That makes it the safest idea to adopt without a chip.
- **Change.** Add a fourth accumulation phase, k=8, from 0.5 of charged time. Relax the parser assert (L1009), `accum_phase_of` and the seed tensor. Scale LR on the k=8 updates by x1.41 (square root) in one arm and x1.68 (the measured batch^0.75) in another. Prebuild the LR tensors before the clock.
- **Skeptic notes.** Static 393k and 524k batches lost (+0.0014, +0.0032). Those runs also enlarged the early batch, where it hurts most. The K41/K42 per-token numbers cap the upside near 0.001. The k=8 step time (1.778 s) is extrapolated from the k1/k2/k4 fit, not measured.
- **Test.** Strict GPU under the virtual clock with k8 = 1.778 s x the control's rescale. Let `FF_TIME_TARGET` end the run; do **not** pin it to 2314 steps. Start with s73, arms x1.41 and x1.68. Kill the family if neither is negative. If one is -0.0005 or better, add s67 and the good-path seed.

### 3. AdamW warm-up immunisation (prewarm arm; pow2 as a second arm)

- **Hypothesis.** Adam's first bias-corrected steps are nearly sign(g) x LR on every embedding row it touches, and its variance estimate is still noisy (the RAdam argument). That overshoot sends about 40% of seeds onto the spike path. Prewarm: steps 0-3 are not charged, so hold the AdamW LR at 0 there while m and v still accumulate. Then ramp linearly to 1.0 by step 19. Pow2: square the AdamW ramp for steps 0-19 only. Muon and all steps from 20 on are unchanged.
- **Why it matters.** For s73 alone it is worth about 0. Its value is that it takes the ±0.0035 coin flip out of every other test. It also takes the flip out of the next upload if #1, #2 or #4 is adopted with no chip check. And it lets c_proj 0.7 be re-tested fairly (good-path mean about -0.0002 to -0.0005).
- **Skeptic notes.** A gentler overall warm-up (wu40d) still looked like a spike on GPU s73. So "gentler" is not automatically "safer". Only the screen can tell.
- **Test.** 60-step GPU screens, which take minutes and are valid here because the spike is fully visible by step 20. Seeds: s58 plus 8 more (include 401, 409, 419, 613, 619, 643 if the GPU reproduces them), with and without each arm. **Pass only if 0 of 9 flagged runs spike:** logged loss < 13.3 at step 10 and < 11.85 at step 20. Then run full strict tests on s58 (expect -0.002 to -0.0035) and s73/s67 (expect within ±0.0003).

### 4. XSHIFT-ALL: the bold arm

- **Hypothesis.** On G7, XSA's per-step gain grew with coverage: about -0.0007 for blocks 0-3 and -0.0018 for all layers. XSHIFT has no reductions, so all 9 layers may be affordable.
- **Risk.** Early-layer heads that use themselves as a no-op lose that option. The price (about 0.8%) is also unpriced.
- **Test.** Uses the same code as #1. Run s73 only, next to #1. Kill it if the paired train diff is above +0.002 nats at step 1600. Add seeds only if s73 is -0.0015 or better. If it gains less than 1.5x XSHIFT-AS, prefer AS for its lower price.

### Parked (need a chip to judge) or dropped

- **Narrow VE (5).** Its real payoff is speed. VE layers currently skip the GQA fast path and expand k and v to 1C. The proxy cannot see that. Three gathers instead of one is also a Trainium risk (bigram hash +11.8%, SMEAR +3.8-4.5%). Park it for a chip.
- **AS-batched (8) and AS-fused QKV (9).** The math is identical, so the GPU can only confirm equivalence, which a CPU test already does. Without a chip S60 there is nothing to adopt. #9 is a subset of #8.
- **KO2 (6) and split-band KO (7).** The 2-gram signal is mostly already there. In blocks 6-8 the residual at t carries x_{t-1}, and the residual at s-1 carries x_{s-2}. So KO's q_t . k_{s-1} can already match 2-grams. The split version also halves the 1-gram band that earned KO's -0.0013. Expected about -0.0003, which is inside seed noise.
- **KO sink (10).** The prior is small (S3/BOS-mode was about equal on one seed). First run the free row-position diagnostic on an existing GPU checkpoint over all 2,097,152 tokens. Revisit only if loss at positions 512-1023 is not below loss at 128-255.
- **Cross-layer KV interleave (11).** It mixes value paths across layers, the same family that lost as S2A/S2B/M16 (+0.0015 to +0.0018). It also depends on #8. Drop it.

## Test rules for round 2 (apply to every arm)

- **Step-10 gate.** Compare the arm's logged loss at step 10 with its control on the same seed. A path mismatch (one above 13.4, the other below 13.3) voids the pair. Rerun on another seed. This saves about 1.8 h per contaminated run.
- **Replace GPU s58** with a seed verified good-path on the GPU, unless #3 passes.
- **Use the paired train diff.** Data order does not depend on the seed and the GPU is bit-repeatable, so same-step differences have 20-50x less noise than the raw loss. A diff that levels off by steps 1100-1300 is a lasting effect. One still shrinking at steps 1600-1900 is fading. A real -0.002 winner should sit at -0.005 nats or better by step 1600.
- **Log the next-token loss separately** (`FF_MTP_LOSS_NEXT`). Until 808 s the logged objective is mostly MTP, and its flat plateaus are easy to misread.

## What NOT to try

- **Scalar retunes:** LR scales, betas, WD, init std, momentum, softcap, QK gain, RoPE base, cooldown moved by 0.1. They have won 0 of 61 keys. Tonight's slate of about 20 of them produced no 3-seed winner.
- **Geometry, table size, sequence length, position loss weights.** All are bracketed on both sides (d8/d10, MLP x3/x5, hd128/512, RoPE bases, WD 0-0.10).
- **Any whole-run LR, momentum or cooldown reshape after step 20** as a -0.002 candidate. The cooldown erases these effects; about 0.0002-0.0005 is left.
- **Anything that changes steps 0-19 for the upload without #3.** It re-rolls the coin flip on a chip we can no longer check, at about 40% x +0.0035.
- **Throughput-only rewrites for the next upload.** The chip reservations had ended, so their price could not be measured.
- **Loader order / epoch-wrap tricks.** They are bounded at 0.0005 or less. Run one only if a strict slot is idle.

## Outcome (added for the release)

- **XSHIFT 6-8** was uploaded as an official test and scored 0.9681, +0.0034 vs K63. That is the warm-up
  spike size this note warned about: the change touched steps 0-19 and re-rolled the seed's path.
- **XSHIFT-all** lost on price: +7.5% step time on chip (0.961266 at only 2,219 steps).
- **Late k=8 in the cooldown** scored 0.9652 officially (+0.0005 vs K63). On top of the row pool it was
  +0.0009 and +0.0001 on two chips. Closed.
- **AdamW warm-up immunisation** lost on chip (+0.0010).
- **The prewarm that shipped** was a different one, and it was found later. The first EMA-on run on a fresh chip
  compiles the EMA kernels inside charged time, so it loses about 15-30 steps. `FF_EMA_PREWARM` moves that compile into
  the excluded startup. It is on in the final recipe.
- **The data-order lever this note did not rank** turned out to be the biggest late win: a 256-batch row pool
  gave -0.0023 (see `docs/FINDINGS.md`).

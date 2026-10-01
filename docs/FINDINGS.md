# Findings: AWS Trainium Frontier, Phase 1

FrontierForge, 15 Sep - 1 Oct 2026. Our best scored upload was **0.96136** (K82s4), outside the top 10. In the final
leaderboard snapshot (about 7:40 AM CDT on 1 Oct), #10 was 0.9554 and #1 was 0.9328. All numbers below come from our
own chip runs and the public leaderboard.
Times are US Central (CDT).

"Rehearsal" means a full 30-minute run of the exact upload file on our own Trainium chip, scored on the first 2M
tokens of the public validation shard. "Official" is the leaderboard score. One `k` is one 65,536-token
micro-batch per optimizer update.

## 1. Score history

| when (CDT) | file | official | what changed |
|---|---|---|---|
| 15 Sep | F2 | 1.1463 | first upload: 131k batch, 20% cooldown |
| 17-19 Sep | K15 -> K33 | 1.1360 -> 1.0427 | compile and static-loss work, depth 12, 262k batch, Muon/AdamW scaling, stream rows |
| 21 Sep | K40 | 0.9953 | depth 10 x width 1024, Muon shard; first score under 1.0 |
| 24 Sep | K44 | 0.9888 | cooldown 0.60 |
| 26 Sep | K50, K51 | 0.9760, 0.9742 | multi-token prediction, batch warm-up (131k -> 262k), key offset |
| 27-28 Sep | K53 -> K57 | 0.9712 -> 0.9686 | ZeRO Muon/AdamW, EMA prewarm, accumulation inside the graph |
| 29 Sep | K59, K60 | 0.9671, 0.9655 | LeakyReLU(0.35)^2; attention-source reuse |
| 30 Sep, 9:25 AM | K63 | 0.9647 | EMA off, cooldown 0.7, lower LR in the small-batch phases |
| 30 Sep, 4:37 PM | K70a | 0.9625 | **row-pool data sampling (256 batches)** |
| 30 Sep, 6:30 PM | K73s4 | 0.9620 | optimizer fusion + XU 48 + c_proj LR 0.8, shuffle salt 4 |
| 30 Sep, 8:20 PM | K77a | 0.9617 | + EMA blend 0.6 |
| 30 Sep, 10:55 PM | K82s4 | **0.96136** | + EMA prewarm (best official); rehearsal 0.954472, projected 0.9614-0.9615 |
| 1 Oct, 1:41 AM | K82s7 | 0.96140 | K82s4 with shuffle salt 7; best rehearsal (0.954379), projected about 0.9614 |

Every scored upload is in [`results/official-scores.csv`](../results/official-scores.csv). Two scores are left
out: an organiser-side rerun of K44 and a truncated duplicate run of F6.

Two things drove most of the descent:

- **Steps.** A 30-minute run did 912 optimizer steps on 18 Sep and about 2,390 on 30 Sep at the same batch
  size. The extra steps came from compiler-friendly code: a scalar stack, fused cross-entropy, Muon sharding,
  ZeRO, accumulation inside the graph and optimizer fusion.
- **A handful of structural changes**, each worth 0.0013 or more: multi-token prediction (-0.0043), batch warm-up
  (-0.0019), the row pool (-0.0023), LeakyReLU(0.35)^2 (-0.0017), attention-source reuse (-0.0015) and the key
  offset (-0.0013).

## 2. The best recipe we publish ([`recipes/K82s4/train.py`](../recipes/K82s4/train.py))

The base is the organiser's GPT baseline. The file is 252,013 bytes, under the 262,144-byte limit, and every
setting below is a default in the file:

- **Model.** Depth 9, width 1024. Four 256-wide query heads share one KV head. Value embeddings feed layers 0
  and 8, and the MLP uses LeakyReLU(0.35)^2. Blocks 6-8 reuse the block-5 attention source (`FF_ATTN_SRC=5:6,7,8`).
  There is a key offset and a 3-token multi-token-prediction loss (weights 1/0.5/0.25), faded to plain
  next-token prediction in 8 stages by 45% of the run. About 124M parameters.
- **Schedule.** The charged-time target is 1,795 s. The batch warms up from 65k to 131k to 262k tokens
  (`FF_ACCUM_SCHED=1:0.06,2:0.2,4`), and the two small-batch phases run at LR x0.35 and x0.6. The cooldown is
  70% of the run.
- **Optimizers.** Muon for the matrices, AdamW for the rest. The Newton-Schulz step is fused (`FF_OPT_FUSE=3`),
  the execution queue is 48 deep, and the MLP output projection runs at LR x0.8.
- **Data.** A 256-batch row pool from micro-batch 96 on (`FF_ROW_SHUFFLE=256`, `FF_ROW_SHUFFLE_FROM=96`). The
  shuffle RNG uses salt 4.
- **Weights at the end.** A 0.6 blend of the live weights with an EMA (every 32 steps). The EMA kernels are
  compiled during the excluded startup (`FF_EMA_PREWARM`), so the first EMA update costs no charged time.
- **Seed** 73.

## 3. What worked, late in the campaign

| lever | effect | evidence |
|---|---|---|
| **Row pool, 256 batches** | **-0.0023** on chip, -0.0022 officially | Chip E: 0.957259 at 2,318 steps vs 0.959523 at 2,329 on the same chip, so it cost about 0.5% of steps (about -0.0026 step-normalised). Official: 0.9625 vs 0.9647. A pool of 1,024 gave the same result as 256; a pool of 64 helped less. The pool starts at micro-batch 96, so the first 96 micro-batches (the warm-up) are unchanged. |
| Attention-source reuse | -0.0015 | Four same-seed pairs: -0.00181, -0.00136, -0.00106, -0.00173. Step cost under 0.1%. |
| Cooldown 0.7 + small-batch LR 0.35/0.6 | -0.0006 (step-normalised) | Three chip pairs, all three negative. Official: K63 0.9647 vs K60 0.9655 (K63 also turned the EMA off and added 2 s of budget). |
| EMA blend 0.6 | about -0.0005 | One chip pair, without prewarm. Official: K77a 0.9617 vs K73s4 0.9620. |
| EMA prewarm | recovers 15-30 steps; about -0.0003 officially | Without it, the first EMA run on a fresh chip compiles the EMA kernels inside charged time. Same chip G: 2,398 steps without EMA (K73s6), 2,375 with EMA (K77a), 2,405 with EMA and prewarm (K82s7; the salt differs, which does not change step time). Chip H: 2,390 without EMA (K73s4), 2,388 with both (K82s4). Official, one pair: K82s4 0.96136 vs K77a 0.9617. |
| Optimizer fusion + XU 48 | +0.8% steps | -0.0006 raw in a run that also had 2 s more budget; almost all of it is steps. On the row-pool base, fusion plus c_proj 0.8 added 0.0002 or less. |
| c_proj LR x0.8 | about -0.0004 (step-normalised) | Consistent on seed 73. LR x0.7 flipped sign on other seeds and was dropped. |
| Shuffle salt | sd about 0.0003-0.0004 | Four salts spanned 0.0008 on chip. Salt 4 kept its edge officially (0.9620 vs 0.9625); salt 6 did not (0.9625). Salt 7 had the best rehearsal on the final recipe but scored 0.96140, a tie with salt 4's 0.96136. A selected draw partly regresses. |

## 4. Seeds and the warm-up coin flip

The biggest single effect in about 430 harvested runs was not a setting. In warm-up steps 3-10, about 40% of seeds
take a "spike" path: the loss falls fast to step 4, then climbs about 0.7 nats. Those runs end **+0.003 to
+0.0045** worse, and the cooldown does not repair it.

- Any change to steps 0-19 re-rolls this coin for a given seed. Our one official test of such a change (XSHIFT in
  blocks 6-8) scored 0.9681, +0.0034. That is a re-rolled seed, not a measured effect.
- Seeds re-draw the init, the warm-up path and the data order (sd 0.0010-0.0012 including bad paths). Salts
  re-draw only the order after micro-batch 96 (sd about 0.0004). In a late seed lottery, 1 of about 6 new seeds
  landed on the good path, and even that one did not beat seed 73.
- The rule we followed: levers must start after step 19, and an effect under 0.0003 needs three seeds.

## 5. What lost

| idea | result |
|---|---|
| Bigger models: depth 10, 2 KV heads | +0.0038 / +0.0043 (depth 10), +0.0057 (kv 2). They ran 6-11% fewer steps. |
| Smaller and faster: depth 8 (+10.6% steps) | +0.0050 and about +0.003. Capacity beat steps here too. |
| Late k=8 batch in the cooldown | +0.0005 official; +0.0009 / +0.0001 on the row-pool base |
| Split AdamW cooldown | +0.0006 official |
| EMA-Nesterov lookahead 0.3 | +0.00004 official (a tie) |
| AdamW LR -16% | +0.0008 / +0.0009 (two seeds) |
| AdamW warm-up reshaping | +0.0010 (it touches steps 0-19) |
| Dropping the block-0 MLP; NoPE in layers 3-4; XSHIFT in all layers | +0.0019; +0.0033; +7.5% step time |
| Asymmetric softcap 16.5 | three pairs averaged -0.0002; a fourth on the K60 base was +0.0007; closed |
| Scalar retunes (LR scales, betas, WD, init, momentum, softcap, RoPE base, cooldown +-0.1) | won 0 of 61 knob keys |
| Bigger row pools, epoch reshuffles, pool-aware accumulation, document reordering | 0 +- 0.0003 |

## 6. The gap to the top 10 is kernel throughput

Near our operating point, score follows compute:

```
delta_bpb ~= 0.063 * ln(effective compute ratio)
```

We fitted this on a 24 Sep run trained for 2,288 s instead of 1,788 s: +28% compute bought -0.0155. The fit's own
slope near our operating point is about 0.00063 bpb per 1%; the per-step rate we measured directly on two chips is
about 0.00057 bpb per 1% more steps.

| target (final snapshot) | gap from our best official 0.96136 | compute needed |
|---|---|---|
| #10 (0.9554) | 0.0060 | about +10% |
| #1 (0.9328) | 0.0286 | about +57% |

We ran about 285k tokens/s on a 124M-parameter model: roughly 30% MFU, with about 19% of the time idle on host
dispatch. +10% effective compute was not going to come from settings, so we believe the way to close the gap is
kernel throughput (NKI). At an equal recipe, +15-23% tokens/s would be worth roughly 0.009-0.013.

Every throughput gain we had came from removing eager dispatch: scalar stack -9.1% step time, fused CE -2.8%,
Muon shard -2.6%, optimizer fusion about -0.7%. Late in the campaign, recipe changes were giving about 0.0005
each. More details: [`GAP-ANALYSIS.md`](GAP-ANALYSIS.md).

## 7. What we would do next

1. **Write NKI kernels** for attention, the fused MLP (LeakyReLU^2 plus both projections) and the Muon/AdamW
   updates. +15-30% tokens/s would be worth 0.009-0.017.
2. **Grow the model into the freed time.** Depth 10 and two KV heads lost only because they cost steps.
3. **Keep the data-efficiency stack.** Row pool, EMA blend + prewarm, attention-source reuse and salt selection
   all add on top of throughput.
4. **Price every kernel before building it.** A chip S60 screen gives the step time. The exchange rate turns it
   into bpb, and a deterministic rehearsal checks the result (see [`EXACT-ORACLE.md`](EXACT-ORACLE.md)).

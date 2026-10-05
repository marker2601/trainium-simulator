# The exact-oracle method: rehearse on a chip, add the offset

The most useful tool we had was not a model. It was a rule:

> Run the **exact upload file**, at the upload seed, on your own Trainium chip, for the full charged budget from a
> cold start. Score the trained weights on the first 2,097,152 tokens of the public validation shard. The official
> score will be that number **plus about +0.0066 to +0.0070**, with an sd of about 0.0004.

We used this rule to pick every rehearsed upload from 26 Sep on (all except K65a-K65d, four official tests of
single changes). Over the 15 rehearsed uploads since K50 the offset averaged +0.0067 with an sd of 0.0004 (range
+0.0061 to +0.0074). The largest miss against a written projection was 0.0006: K73s6, a salt-selected draw, was
projected at 0.9619 and scored 0.9625. The two final uploads landed within 0.0001 of their projections. Over all 30
uploads with a rehearsal, including the early chip-A/B era, the offset ranged from +0.0053 to +0.0077.

## Why it works

1. **Training is deterministic for a given file and seed.** A cold rehearsal and a warm rerun of the same file
   agreed to about 5e-6 (1.041525 vs 1.041520 on 19 Sep). The official run executes the same code on the same
   kind of chip, so it follows the same optimizer trajectory.
2. **The offset is mostly a property of the text, not of the recipe.** The first 2M public tokens are easier than
   typical text: the same model scores about +0.0070 worse on 20M public tokens, about the size of the offset. The
   offset therefore barely moves between very different recipes. It moved 0.0011 across nine submissions that
   changed the architecture, the optimizer and the data order. It may also depend on the instance that ran the
   rehearsal: on the final day it was larger for rehearsals on the faster chips G and H.

## The calibration data

| period | uploads | offset mean | sd | range |
|---|---|---|---|---|
| chip-C rehearsals K51-K60 (the simulator's offset model) | 7 | +0.00653 | 0.00042 | +0.0061 .. +0.0072 |
| final day, chips C/E/G/H (K63-K82s7) | 7 | +0.00687 | 0.00030 | +0.0065 .. +0.0074 |
| everything since K50 | 15 | +0.00671 | 0.00038 | +0.0061 .. +0.0074 |
| every scored upload with a rehearsal | 30 | +0.00648 | 0.00061 | +0.0053 .. +0.0077 |

As a held-out check, freeze the chip-C fit (+0.00653) and predict the seven final-day uploads from their rehearsals:
the errors (official - rehearsal - 0.00653) are K63 -0.00004, K70a +0.00002, K73s4 +0.00026, K73s6 +0.00083
(salt-selected), K77a +0.00044, K82s4 +0.00036 and K82s7 +0.00049. That is an MAE of 0.00035 and a bias of +0.00034:
the final-day chips ran a little hotter than chip C. K70a's rehearsal was normalised to chip-C speed.

Per-upload values are in [`results/official-scores.csv`](../results/official-scores.csv), and the table the
simulator fits on is `research/sim-data/official-uploads.csv`. Two organiser-side anomalies are excluded:

- a rerun of K44 on a slower host, which did 2.3% fewer steps and scored +0.0014 worse;
- a duplicate F6 run that was cut before its LR decay.

The K60 rehearsal log is in `research/rehearsal-logs/C41_cold_K60/`. It shows 2,361 steps and val_bpb 0.95910; the
official score was 0.9655, an offset of +0.0064.

## How we used it

- **Gate every upload.** Apart from K65a-K65d, a file was uploaded only after its cold rehearsal finished cleanly:
  exit 0, no graph breaks, charged time inside the budget, and the file under the 262,144-byte limit. Its projected
  score was written down before the upload.
- **Test on chips instead of on the leaderboard.** Because a rehearsal predicts the official score, a structural
  change can be judged on our own chips without spending an upload. On the final day we ran four chips as an
  "official test bench": each change was built as a complete upload file and rehearsed at seed 73.
- **Lotteries.** Shuffle salts and seeds were scored the same way, and the best rehearsal was uploaded. Expect a
  selected draw to give back part of its edge: salt 4 kept its local lead officially, salt 6 did not.

## The traps

- **Cold means cold.** One-time compiles inside charged time cost steps. The first EMA run on a fresh chip paid
  the EMA kernel compile inside the clock: 15-30 fewer steps than later, warm EMA runs on the same chip. On chip G,
  K77a (EMA, no prewarm) ran 2,375 steps and K82s7 (EMA with prewarm, another salt) ran 2,405. A warm
  rehearsal therefore looked 0.0002-0.0006 better than the cold official run would be. `FF_EMA_PREWARM` moves that
  compile into the excluded startup.
- **Chip speed.** Our chips differed by up to 3.5% in step time, and one on an older Neuron runtime was 7-12%
  slower in full runs. Judge a chip only against its own control, or normalise a same-recipe run to chip-C speed
  at 0.00057 bpb per 1% of steps. Never normalise across a change that alters step time.
- **The warm-up coin flip.** A change to steps 0-19 can move a seed onto a different warm-up path. The rehearsal
  still predicts the official score exactly: the official run takes the same path. But the difference you see is
  then a seed effect, not the change's effect.
- **The time target is a budget, not a promise.** A run's step clock stops a median 5.5 s short of the target. We
  left about 6-8 s of headroom under the 1,800 s cap.

## Reproducing it

The official harness, data loader and evaluation are the organiser's
([github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier)). On a trn2 instance,
put `recipes/K82s4/train.py` in place of the kit's `train.py`, run the kit's training command (8 ranks under
`torchrun`) and its public-shard evaluation, then add the offset:

```bash
python -c "from ffsim.offset import OffsetModel; m = OffsetModel().fit('research/sim-data/official-uploads.csv'); print(m.summary()['mean'], m.predictive_sd)"
```

For K82s4, the chip rehearsal was 0.954472 at 2,388 steps. The final-day mean offset (+0.0069 to +0.0070)
projected that to about 0.9614-0.9615, with a likely range of 0.9611-0.9619. It scored 0.96136 (offset +0.0069).
The command above prints the simulator's offset, which is fitted on the seven chip-C points K51-K60 (+0.00653).

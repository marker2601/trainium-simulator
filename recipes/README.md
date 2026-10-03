# Recipes

Both files are byte-identical to what we uploaded to the challenge. They are modified versions of the organiser's
Apache-2.0 baseline and stay under that licence (`LICENSE-APACHE-2.0`, attribution in `../NOTICE`). Each runs
inside the organiser's kit ([github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier)),
because it imports the kit's `prepare.py`.

| file | bytes | sha256 | chip rehearsal (seed 73) | official |
|---|---|---|---|---|
| `K82s4/train.py` | 252,013 | `8e21ce19411095cb498bb41bfa22c1327a10f8b524d6f7f43c026dbf831efae8` | 0.954472 @ 2,388 steps | 0.96136 (best official) |
| `K60/train.py` | 237,050 | `ae396432c31fe83122a91339916accffaa68aff10fdf47432cee46d3074953cc` | 0.95910 @ 2,361 steps | 0.9655 |

K82s4 is our best official score, 0.96136. K77a (0.9617) is K82s4 without `FF_EMA_PREWARM`, so it pays the EMA
compile inside charged time. Pass `FF_EMA_PREWARM=0` to K82s4 to reproduce it.

The best rehearsal of the campaign was K82s7: this K82s4 file with the shuffle salt changed from 4 to 7 (one
line, `7777 * 4` -> `7777 * 7` in the row-pool RNG seed). It rehearsed 0.954379 at 2,405 steps on a different
chip and was our final pre-close upload. It scored 0.96140, a tie with K82s4: salt gains partly regress officially
(see `../docs/FINDINGS.md`, section 3), so we publish K82s4 as the best file.

## K82s4: the best recipe we publish

Every setting is a default in the file, so run it with no `FF_*` overrides. Lineage:

```
K60  = K59 + attention-source reuse (FF_ATTN_SRC=5:6,7,8)                     official 0.9655
K62a = K60 + EMA off + time target 1795 s
K63  = K62a + cooldown 0.7 + small-batch LR 0.35/0.6 (FF_ACCUM_LR)            official 0.9647
K70a = K63 + row pool 256 from micro-batch 96 (FF_ROW_SHUFFLE)                official 0.9625
K72a = K70a + fused Newton-Schulz (FF_OPT_FUSE=3) + XU queue 48 + c_proj LR 0.8
K73s4 = K72a + shuffle salt 4                                                  official 0.9620
K77a = K73s4 + EMA on, blend 0.6 (FF_EMA_BLEND)                               official 0.9617
K82s4 = K77a + EMA prewarm (FF_EMA_PREWARM)                                   official 0.96136
```

These `FF_*` variables let you switch a lever off to measure it, for example `FF_ROW_SHUFFLE=0`,
`FF_EMA_BLEND=0` or `FF_ATTN_SRC=`. Read the asserts near each setting first: several levers depend on each other.
Any change that alters steps 0-19 re-rolls the seed's warm-up path; see `../docs/FINDINGS.md`, section 4.

## K60

K60 is the 29 Sep recipe. The GPU proxy (`../ffsim/gpu/train_gpu.py`) is generated from it, and the simulator's
example recipe `../ffsim/examples/recipe-K60.json` describes it.

## Launch command

Every rehearsal and official upload ran with the organiser's launch command, one process per NeuronCore:

```bash
NEURON_LOGICAL_NC_CONFIG=1 torchrun --standalone --nproc_per_node=8 train.py
```

Software: the organiser-provided AWS Neuron environment on `trn2.3xlarge` (PyTorch 2.11.0, `torch-neuronx` 2.11.3,
`neuronx-cc` 2.26, build 2.26.6360.0).

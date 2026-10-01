🚀 **Stop waiting for trn2 capacity: test your ideas for free first**

Hey everyone 👋 FrontierForge here. Phase 1 is a wrap, and we've opened up the tool that got us from 0.9888 to 0.9617: a **Trainium simulator** that predicts your `val_bpb` *before* you spend a single chip hour.

📦 **Get it here (code, data, app and our best recipe):** https://github.com/Marker2601/trainium-simulator

**What it does**
• 🔮 *Predict a recipe*: set depth, width, LRs, cooldown, batch schedule and time budget, and get the predicted Trainium `val_bpb`, step count and step time, plus the expected official score
• ⚡ *Speed → score calculator*: "if my step time drops 10%, what do I gain?" (spoiler: about 0.005-0.006 bpb)
• 🖥️ *GPU proxy (in the repo)*: replays a Trainium run step for step on a cheap NVIDIA GPU
• 📊 Fitted on ~1,200 real Trainium run records and 26 official uploads; it also flags any knob outside what it was fitted on

**How to use it (2 minutes)**
1. `git clone` the repo → `pip install -r space/requirements.txt` → `python space/app.py` → open http://127.0.0.1:7860, start from our default recipe, and change one knob at a time
2. Compare the predicted bpb against the default; the app shows the uncertainty band too
3. Want an API? `/predict`, `/predict_overrides`, `/speed_to_score` and `/score_to_speed` are callable with `gradio_client` on your local app; examples are in `space/README.md`
4. Prefer the command line? `pip install -r requirements.txt` → `python -m ffsim simulate --recipe ffsim/examples/recipe-K60.json`

**Honest fine print** 🧾 It nails the big picture (what matters and roughly how much), with a typical error of about 0.001 bpb. For differences under about 0.0003 it's a coin flip, so confirm the winners on a real chip. It's calibrated to trn2.3xlarge and nanoGPT-style recipes like ours.

**Things we learned the hard way** (all in the repo):
• Shuffling micro-batch rows across a 256-batch pool: −0.0023 bpb, for free 🤯
• EMA weight blending + a one-line "prewarm" fix: −0.0005
• Bigger *and* smaller models both lost, so the rest of the gap is kernel speed

⭐ **If it saves you even one chip hour, please star the repo.** Our simulator predicts that each star lowers our val_bpb by 0.0001 (MAE: ±infinity; we're still calibrating 😄). Stars are the only gradients we accept this phase.

Congrats to the top 10, and good luck in Phase 2! 🏁

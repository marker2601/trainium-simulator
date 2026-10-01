# Where to submit: venue plan, Oct 2026 to mid 2027

> **Status.** Researched 1 Oct 2026 from official calls where they exist; the contest-rules section was updated in
> revision 1 (same day) after reading the public Devpost rules. Every date below has a source link.
> A venue whose 2027 call is not out is marked **not yet announced**. Where we give a previous year's date, it is
> labelled **pattern only**: it tells you roughly when to look, not when the deadline is.
>
> **Times.** Deadlines are given as the venue states them, plus US Central (Chicago) time. Daylight saving ends on
> 1 Nov 2026, so dates before then are CDT (UTC-5) and dates after are CST (UTC-6). "AoE" (Anywhere on Earth, UTC-12)
> 23:59 on day D is 06:59 CDT or 05:59 CST on day D+1 in Chicago. **Aim to submit a day early.**
>
> **What we are placing.** An empirical systems-plus-ML paper with four parts:
> 1. **Exact-oracle evaluation (C1).** Deterministic chip rehearsal plus an offset (text + per-chip part) predicts
>    official scores: frozen-calibration MAE 0.00035 on the final seven uploads, resolution about 0.001 for a
>    difference between two uploads.
> 2. **A run simulator (`ffsim`, C2)** with baselines: beats a steps-only model on held-out recipe pairs, at chance on
>    new mechanisms.
> 3. **Accounting (C3).** One price, 0.00057 bpb per 1% more steps (κ = 0.057; 0.063 from one long run), the row pool
>    (-0.0023 chip, one pair), the EMA-prewarm compile trap, seed/salt noise, negative results, the gap priced as a
>    kernel hypothesis.
> 4. **Released artifacts (C4)** and a guarded contribution pipeline.
>
> Revision 1 (1 Oct 2026) answered three internal reviews; see [`RESPONSE.md`](RESPONSE.md).
>
> The plan is in [`OUTLINE.md`](OUTLINE.md).

---

## TL;DR: the recommendation

### (a) First paper, now (October 2026)

| Rank | Venue | When | Why |
|---|---|---|---|
| **1** | **arXiv technical report** (cs.LG primary; cross-list cs.DC, cs.PF) | Post as soon as an endorsement is in hand. Target: week of 12 Oct 2026. | It puts a date on the work and gives a citable link. Every other venue below allows preprints. It also starts a public trail, which JOSS later requires. **Get the endorsement first, because it is now the bottleneck (see below).** |
| **2** | **MLSys 2027**, research track (stretch goal) | Submissions open 10 Oct 2026, 12:00 PDT (14:00 CDT). **Deadline 30 Oct 2026, 12:00 PDT (14:00 CDT).** | Best topical fit of any peer-reviewed venue with a deadline in reach. Training efficiency, hardware acceleration, benchmarks and tooling are all in its call. It needs a 10-page two-column anonymised version and at least the E1 and E7 experiments from the outline. The odds are low to moderate. |
| **3** | **TMLR** (rolling) | Any time. Submit after an MLSys rejection (decisions 28 Feb 2027), or right away if we skip MLSys. | Highest acceptance odds for this kind of paper. TMLR judges whether the claims are supported and whether some readers care, not novelty. Negative results and careful measurement fit well. **No dual submission with MLSys.** |

**Suggested path.** Post to arXiv first. Then do one of two things:
- **(i)** If the extra experiments land by about 25 Oct, submit to MLSys on 30 Oct. If it is rejected, revise and go to TMLR in March 2027.
- **(ii)** If they do not land, go straight to TMLR in November 2026 with the full-length version.

Path (ii) is the safer way to get a peer-reviewed paper. Path (i) has the higher upside.

### (b) Follow-up paper, Nov-Dec 2026

The natural second paper narrows to one of two topics:
- **The simulator and exact-oracle methodology**, for a performance-analysis audience.
- **The speed-run lessons**, as a short workshop paper (steps-as-currency, row-pool decorrelation, negative results).

| Rank | Venue | When | Why |
|---|---|---|---|
| **1** | **An ICLR 2027 workshop** on efficient training, data or scaling | The workshop list comes out after **29 Nov 2026**. ICLR suggests workshops take papers by **1 Feb 2027**, with authors notified by **26 Feb 2027**. Workshops run 29-30 Apr 2027 in California. | The audience is large and relevant, and short papers are the norm. Workshops usually do not publish proceedings, so this stays compatible with TMLR. Write the 4-page version in December. |
| **2** | **ISPASS 2027** (IEEE performance analysis) | **Not yet announced.** Pattern only: the ISPASS 2026 deadline was 12 Dec 2025. | Strongest fit for the **simulator paper**. Its call names simulation, analytical modelling, deep-learning workloads, and tool or benchmark papers. Watch for the call in Oct-Nov 2026. |
| **3** | **EuroMLSys 2027** (workshop at EuroSys 2027, Rabat, 19 Apr 2027) | **Not yet announced.** EuroSys workshop proposals are due 16 Oct 2026, with acceptances on 30 Oct 2026. Pattern only: the EuroMLSys 2026 deadline was 24 Feb 2026 (extended). | A 6-page ML-systems paper, published in the ACM Digital Library unless you opt out. Good fit for the simulator or the Trainium-determinism angle. |
| **4** | **CPAL 2027, Recent Spotlight track** (Tokyo, Mar 2027) | Deadline **18 Jan 2027 AoE** (19 Jan, 05:59 CST). | Cheap visibility: a 250-word abstract and no proceedings. The archival proceedings track requires a parsimony framing that our work does not have. |
| **5** | **AAAI-27 workshop**, only if a matching one appears | Workshop calls were due to AAAI on 2 Oct 2026. Authors must be notified by 2 Dec 2026, so paper deadlines are likely late Oct to Nov. Workshops run 22-23 Feb 2027 in Montréal. | Only worth it if a workshop on efficient training or ML systems is listed. AAAI's audience is less systems-focused. |

### Mid 2027 (for planning)

- **JOSS**, for the `ffsim` software. Eligible from about **April 2027**, because the repo must have been public for more than 6 months with active development. It went public on 1 Oct 2026.
- **ICML 2027 / NeurIPS 2027.** Neither has announced its deadlines officially. Only worth trying if the generalisation experiments (beyond one chip and one contest) work out.
- **JMLR MLOSS.** Only once there is an active user community.

---

## Before anything: an arXiv endorsement is required

- **Who needs it.** arXiv requires endorsement before your first paper on arXiv or in a new category
  ([arXiv help: endorsement](https://info.arxiv.org/help/endorsement.html)).
- **What changed in 2026.** Since **21 Jan 2026**, an institutional email alone is no longer enough. Automatic
  endorsement now needs **both** an academic or research institutional email **and** prior authorship of a paper
  already accepted to arXiv in that endorsement domain. Everyone else must get a **personal endorsement from an
  established arXiv author in the same domain**
  ([arXiv blog, 21 Jan 2026](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/)).
- **What to do.**
  1. Start the submission. arXiv emails a personal endorsement link.
  2. Send the link, the draft PDF and the repo to someone you know who has published in **cs.LG**.
  3. Endorsement is per endorsement domain. All three categories (cs.LG, cs.DC, cs.PF) are in Computer Science, so
     the cs.LG endorsement should cover the cross-lists. Confirm this in the arXiv submission screen.
  4. Allow several days for this. It is the most likely cause of delay.
- **Categories.** Primary **cs.LG** (Machine Learning). Cross-list **cs.DC** (Distributed, Parallel, and Cluster
  Computing) and **cs.PF** (Performance).
- **Licence.** Choose **CC BY 4.0** for the arXiv version. It matches TMLR's required licence, so there is no
  conflict later ([TMLR author guide](https://jmlr.org/tmlr/author-guide.html)).
- **The challenge terms are public** ([Devpost rules](https://trainium-frontier.devpost.com/rules), checked
  1 Oct 2026; cited in the paper as `trainiumfrontier2026rules`). What matters for publishing:
  - The **top 10 teams of Round 1** advance (Round 2: 7 Oct to 4 Nov 2026; finalists announced by 11 Nov 2026).
    FrontierForge finished outside the top 10, so the contest's own paper track does not apply to us.
  - **Finalists** must submit a **4-8 page technical paper in the PMLR NeurIPS Competitions volume template**
    (architecture, kernels, ablations on val_bpb and CORE, lessons) by **12 Dec 2026**; the final round is a
    **NeurIPS 2026 competition event, 6-12 Dec 2026**. Winners are listed on the contest GitHub repository.
  - Winners must release training code and weights under **Apache 2.0**; AWS gets a non-exclusive right to publish
    submission materials. We found **no rule restricting participants from publishing their own methods or
    results**, and winners are encouraged to write blog posts and tutorials.
  - The rules define the metric as val_bpb "on the pinned validation shard, computed over ~20M tokens", do not say
    whether that shard is disjoint from the public validation data, state the 30-minute budget "excluding
    startup/compilation", and set the Round 1 deadline at 11:59 PM PT on 30 Sep 2026 (our last upload, K82s7, was at
    11:41 PM PT). The 262,144-byte file limit is not in the published rules (the portal enforced it).
  - Each team is limited to "one entry"; the portal scored repeated uploads from an entry. The rules do not say
    whether standings use a team's best or last upload.
  - The contest provides AWS promotional credits to participating teams; the paper's compute statement should say
    which part of our compute they covered (owner to confirm).

---

## Venue-by-venue detail

Fit and odds are our judgement. Dates, limits and formats come from the linked sources.

### 1. arXiv (preprint, not peer reviewed)

| | |
|---|---|
| **Deadline** | None. Announcements follow arXiv's daily schedule. |
| **Length / format** | Any length; PDF built from LaTeX. The OUTLINE target is 6-8 pages plus appendix. |
| **Fit** | High. This is the expected home for a technical report on a contest campaign. |
| **Odds** | Moderation only. The paper is original empirical research, not a review or position paper. |
| **Gate** | The endorsement (above). |
| **Sources** | [Endorsement help](https://info.arxiv.org/help/endorsement.html), [2026 policy](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/) |

### 2. MLSys 2027 (conference; research track; industry track)

| | |
|---|---|
| **Deadline** | Submissions open **10 Oct 2026, 12:00 PDT**. **Paper deadline 30 Oct 2026, 12:00 PDT** (14:00 CDT). The call page gives "20:00 UTC" (15:00 CDT); go by the earlier time. Reviews out 18 Jan 2027, responses due 21 Jan 2027, decisions 28 Feb 2027. |
| **Conference** | 22-24 Jun 2027 in Indio, California; Industry Day 25 Jun 2027. |
| **Length / format** | **Up to 10 pages excluding references.** Two-column MLSys LaTeX style. Unlimited optional appendix. |
| **Review** | **Double-blind.** The research track requires full anonymisation, so use an anonymised repo mirror and no team name. The industry track anonymises authors but allows company and product details. |
| **Preprints** | arXiv is allowed, and so is concurrent submission to non-archival workshops. |
| **Artifacts** | Optional ACM-style artifact evaluation. Our tests, simulator and dataset make this a strength. |
| **Fit** | **High on topic.** The call lists training and inference efficiency, hardware acceleration and hardware-efficient ML methods, and ML benchmarks, datasets and tooling. |
| **Odds** | **Low to moderate.** The weak spots are a single platform, a single contest and a 17-day campaign. To be competitive, the paper needs at least E1 (simulator held-out validation) and E7 (oracle error budget), and ideally a second platform or the GPU proxy track. |
| **Sources** | [Dates](https://mlsys.org/Conferences/2027/Dates), [Call for research papers](https://mlsys.org/Conferences/2027/CallForResearchPapers) |

### 3. TMLR, Transactions on Machine Learning Research (journal, rolling)

| | |
|---|---|
| **Deadline** | **Rolling.** Submit any time on OpenReview. |
| **Length / format** | Any length, "justified by its content". Unusually long papers get a longer review. Use the TMLR LaTeX style. Up to 100 MB of supplementary material is encouraged. |
| **Review** | Double-blind, with reviews public on OpenReview. An action editor is assigned within a week. The final recommendation comes at least two weeks after all three reviews are public. |
| **Acceptance criteria** | (1) Are the claims backed by accurate and convincing evidence? (2) Would some TMLR readers be interested, and is the paper clear? There is no novelty or significance bar, which suits careful negative results. |
| **Dual submission** | **Not allowed** while the paper is under review at another archival venue. Overlap is fine with arXiv and with workshops that declare themselves non-archival. |
| **Extras** | ISSN 2835-8856. Optional certifications, including the Journal-to-Conference track for presenting at NeurIPS, ICML or ICLR. Licence is CC BY 4.0. |
| **Fit** | **High.** |
| **Odds** | **Good**, if every claim is scoped to what the repo supports. |
| **Sources** | [Author guide](https://jmlr.org/tmlr/author-guide.html), [Acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html), [Editorial policies](https://jmlr.org/tmlr/editorial-policies.html), [TMLR home](https://jmlr.org/tmlr/) |

### 4. ICLR 2027, main conference: **closed**

| | |
|---|---|
| **Deadline** | Abstracts closed 18 Sep 2026 and papers closed 25 Sep 2026 (both AoE). Decisions are due 16 Dec 2026. |
| **Conference** | 26-28 Apr 2027 (main), 29-30 Apr 2027 (workshops), California. |
| **Status** | Missed. The next ICLR main deadline would be for ICLR 2028, around Sep 2027. |
| **Sources** | [ICLR 2027 dates](https://iclr.cc/Conferences/2027/Dates), [ICLR home](https://iclr.cc/) |

### 5. ICLR 2027 workshops

| | |
|---|---|
| **Deadlines** | Workshop proposals were due 9 Oct 2026; accepted workshops are announced **29 Nov 2026**. ICLR suggests workshops take papers by **1 Feb 2027**. Organisers must notify authors and post accepted papers on OpenReview by **26 Feb 2027**. Each workshop sets its own exact deadline. |
| **Workshops** | **29-30 Apr 2027**, San Francisco. The workshop page reads "April 29 and 30, 2026", an evident typo for 2027. |
| **Length / format** | Set by each workshop. Usually 4-page short papers or up to about 8-9 pages. |
| **Archival** | Each workshop must state its archival status. Most are non-archival, so they are compatible with TMLR. Check each one. |
| **Fit** | **High** if a workshop on efficient training, data, scaling or reproducibility is accepted. That is likely, but the list is not yet published. |
| **Odds** | **Good** for a solid 4-page paper. |
| **Sources** | [Call for workshops](https://iclr.cc/Conferences/2027/CallForWorkshops), [Workshop guidelines](https://iclr.cc/Conferences/2027/WorkshopGuidelines) |

### 6. ICLR 2027 blog post track

| | |
|---|---|
| **Deadline** | **Not yet announced.** The 2027 call page says the document "is not yet available". Pattern only: the 2026 deadline was 7 Dec 2025, with notification on 21 Feb 2026. |
| **Format** | Markdown or HTML post in the track's template. |
| **Fit** | **Low to medium.** Posts are expected to discuss previously published work, and the track has conflict-of-interest rules. It would only fit a post about published speed-run methods (time-to-loss evaluation, data ordering) that uses our numbers as evidence. It is a good popular-audience channel if the 2027 rules allow it. |
| **Sources** | [2027 call page (not yet available)](https://iclr.cc/Conferences/2027/CallForBlogPosts), [2026 track: about and dates](https://iclr-blogposts.github.io/2026/about/) |

### 7. ICML 2027

| | |
|---|---|
| **Deadline** | **Not yet announced** on icml.cc. The official future-meetings page says only "2027 -- South America". Third-party trackers list late January 2027 (one says 22 Jan, another 28 Jan); treat these as unverified. |
| **Fit / odds** | **Medium fit, low odds** as the paper stands. A main-track ML venue would want the findings shown to generalise beyond one contest. ICML 2027 workshops (around May 2027) are a better route. |
| **Sources** | [ICML future meetings](https://icml.cc/Conferences/FutureMeetings) |

### 8. NeurIPS 2026 workshops (Dec 2026): **effectively closed**

| | |
|---|---|
| **Status** | The workshop tracker's latest open deadline was 28 Sep 2026 (a fast-track call in an unrelated area). The relevant workshops (ML for Systems, OPT, efficiency and scaling workshops) closed between 29 Aug and 6 Sep 2026. |
| **Conference** | Dec 2026. The official page lists Sydney, 6-12 Dec 2026, plus two satellite meetings. The tracker also lists Atlanta and Paris. |
| **Sources** | [NeurIPS 2026 workshop tracker](https://aiworkshoptracker.com/conference/neurips/2026/), [NeurIPS future meetings](https://neurips.cc/Conferences/FutureMeetings), [Announcing the NeurIPS 2026 workshops](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) |

### 9. NeurIPS 2027 (main track; Datasets and Benchmarks track)

| | |
|---|---|
| **Deadline** | **Not yet announced.** The official page says only "2027 -- Europe". Trackers project May 2027; unverified. |
| **Fit** | The **run dataset of about 1,200 chip-run records plus the simulator** could suit a Datasets and Benchmarks-style track, if NeurIPS 2027 runs one. Check when the call is out (around spring 2027). |
| **Sources** | [NeurIPS future meetings](https://neurips.cc/Conferences/FutureMeetings) |

### 10. AAAI-27 workshops

| | |
|---|---|
| **Deadlines** | Proposals were due 28 Aug 2026 and organisers were told on 25 Sep 2026. Workshop calls were due to AAAI on **2 Oct 2026**. **Authors must be notified by 2 Dec 2026**, so paper deadlines are likely late Oct to Nov. The accepted-workshop list is **not yet published**. |
| **Workshops** | 22-23 Feb 2027 (AAAI-27 runs 16-23 Feb 2027, Montréal). |
| **Fit / odds** | **Low to medium.** It depends on a matching workshop existing. |
| **Sources** | [AAAI-27 workshop call](https://aaai.org/conference/aaai/aaai-27/workshops-call/) |

### 11. EuroSys 2027 and EuroMLSys 2027

| | |
|---|---|
| **EuroSys 2027** | 19-23 Apr 2027, Rabat, Morocco. Workshops on 19 Apr 2027, in person only. Workshop proposals due 16 Oct 2026, acceptances 30 Oct 2026. |
| **EuroMLSys 2027** | **Not yet announced.** Pattern only: the 6th EuroMLSys (at EuroSys '26) had a deadline of 24 Feb 2026 (extended), **6 pages two-column** (SIGPLAN template, references not counted), and ACM Digital Library publication with an opt-out. |
| **Fit / odds** | **High fit, moderate odds** for a 6-page systems paper on the simulator, deterministic rehearsal or Trainium step-time. If we publish in the ACM DL, the paper becomes archival and blocks a TMLR version with the same content, so opt out if TMLR is the plan. |
| **Sources** | [EuroSys 2027 call for workshops](https://2027.eurosys.org/workshop.html), [EuroMLSys](https://euromlsys.eu/) |

### 12. ASPLOS 2027: **closed**; ASPLOS 2028: not yet announced

| | |
|---|---|
| **Status** | ASPLOS 2027 cycles closed on 15 Apr 2026 and 9 Sep 2026. The conference is 11-15 Apr 2027 in Heraklion, Crete. 11 pages two-column. ASPLOS 2028 cycles are not yet announced; pattern only, April and September. |
| **Fit** | Low for this paper, which is an ML campaign and not an architecture contribution. |
| **Source** | [ASPLOS 2027 call for papers](https://www.asplos-conference.org/asplos2027/cfp/) |

### 13. SC-adjacent workshops

| | |
|---|---|
| **Status** | **SC26 workshops closed.** SC26 is 15-20 Nov 2026 in Chicago, and its AI/ML workshops had August 2026 deadlines (for example AI on HPC: 9 Aug; AI4S: 14 Aug). SC27 workshops are not yet announced. |
| **Sources** | [SC26 workshops](https://sc26.supercomputing.org/program/workshops/), [AI on HPC @ SC26](https://ai-on-hpc.github.io/sc26/), [AI4S 2026](https://ai4s.github.io/) |

### 14. ISPASS 2027 (IEEE Symposium on Performance Analysis of Systems and Software)

| | |
|---|---|
| **Deadline** | **Not yet announced.** Pattern only: the ISPASS 2026 full-paper deadline was **12 Dec 2025** AoE, and the conference was 26-28 Apr 2026 in Seoul. |
| **Fit** | **High for the simulator paper.** Its call lists simulation techniques, analytical modelling, and deep-learning workloads, and it welcomes tool and benchmark papers. |
| **Odds** | Moderate, with a validated simulator, held-out error bars and a comparison against simple baselines. |
| **Source** | [ISPASS 2026 call (SIGARCH)](https://www.sigarch.org/call-contributions/ispass-2026/) |

### 15. CPAL 2027 (Conference on Parsimony and Learning)

| | |
|---|---|
| **Deadlines** | Abstract registration **23 Nov 2026**, proceedings paper **5 Dec 2026**, Recent Spotlight **18 Jan 2027**. All are 23:59 AoE, which is 05:59 CST the next day. |
| **Conference** | March 2027, Tokyo. |
| **Length / format** | Proceedings track: **up to 9 pages**, excluding references and appendix. Double-blind on OpenReview, published in **PMLR**. Spotlight track: a **250-word abstract** plus supporting material, single-blind and **non-archival**. |
| **Fit** | **Medium for Spotlight, low for Proceedings.** Compute-efficient and data-efficient training are in scope, but every topic "must demonstrate how parsimony (sparsity, low rank, modularity, or compression) enables" the work, and ours does not. |
| **Sources** | [Key dates](https://cpal.cc/deadlines/), [Call for papers](https://cpal.cc/call_for_papers/) |

### 16. COLM 2027

| | |
|---|---|
| **Deadline** | **Not yet announced** (the 2027 call page returns 404). Trackers list 31 Mar 2027; unverified. |
| **Fit** | Medium. It is a language-modelling venue and efficient pretraining is in scope, but the paper's core is systems and evaluation. |
| **Source** | [COLM 2027 call (404 at time of check)](https://colm.cc/Conferences/2027/CallForPapers) |

### 17. JMLR MLOSS track (open-source ML software)

| | |
|---|---|
| **Deadline** | Rolling. |
| **Length / format** | **Up to 4 pages** in JMLR format, plus references, and a cover letter. |
| **Requirements** | An open-source licence. Installation instructions, tutorials, non-trivial examples and **full API documentation**. **Test coverage "close to 100%"**. A comparison against existing implementations. **CI**, ideally on all supported platforms. **Evidence of an active user community.** |
| **Fit / odds** | **Not yet.** There is no user community yet, and `ffsim` is built for one contest. Revisit in late 2027 if others adopt it. |
| **Source** | [MLOSS submission info](https://jmlr.org/mloss/mloss-info.html) |

### 18. JOSS, Journal of Open Source Software (for the software)

| | |
|---|---|
| **Deadline** | Rolling, but **eligible from about 1 Apr 2027 at the earliest**. |
| **Requirements** | An OSI-approved licence (we have MIT, plus Apache-2.0 parts). The repo must have been "public for more than six months prior to submission, with active development spanning that period". There must be "evidence that the software is being used for research"; our arXiv paper counts, and outside users count more. No "minor utility" packages. Tests, documentation, and for multi-author projects, visible issues and pull requests. |
| **Fit / odds** | **Good from April 2027**, if we keep developing in public (issues, releases, CI, CONTRIBUTING.md) between now and then. The repo was made public on 1 Oct 2026. |
| **Source** | [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html) |

---

## Calendar (Chicago time)

| Date | What |
|---|---|
| **Now to 10 Oct 2026** | Get the arXiv endorsement and finish the arXiv draft. (Challenge terms checked 1 Oct: no publication restriction found.) |
| 9 Oct 2026 | ICLR 2027 workshop proposals close; nothing for us to do. |
| **10 Oct 2026, 14:00 CDT** | MLSys 2027 submissions open. Register the paper early. |
| **~12-16 Oct 2026** | Post to arXiv (cs.LG; cross-list cs.DC and cs.PF). |
| 16 Oct / 30 Oct 2026 | EuroSys 2027 workshop proposals due / accepted. Watch for EuroMLSys 2027. |
| **30 Oct 2026, 14:00 CDT** | **MLSys 2027 paper deadline** (if we take path (i)). |
| Oct-Nov 2026 | Watch for the ISPASS 2027 call and AAAI-27 workshop calls. |
| Nov 2026 | Path (ii): submit the full version to TMLR. |
| 23 Nov / 5 Dec 2026 | CPAL 2027 abstract and paper deadlines (proceedings track; only if we reframe around parsimony). |
| **29 Nov 2026** | ICLR 2027 workshops announced. Pick the target and start the 4-page version. |
| ~Dec 2026 (pattern) | Possible ISPASS 2027 deadline: the simulator paper. |
| 18 Jan 2027 | CPAL 2027 Recent Spotlight deadline (05:59 CST on 19 Jan). |
| **~1 Feb 2027** | Suggested ICLR 2027 workshop paper deadline. |
| 28 Feb 2027 | MLSys 2027 decisions. If rejected, revise and submit to TMLR. |
| ~Feb 2027 (pattern) | Possible EuroMLSys 2027 deadline. |
| **From ~1 Apr 2027** | JOSS submission for `ffsim`, once 6 months public. |
| Spring 2027 | ICML 2027 and NeurIPS 2027 calls (check officially), and their workshops. |

---

## Submission hygiene (applies everywhere)

- **Anonymisation.** MLSys, TMLR, CPAL Proceedings and ISPASS are double-blind. Remove the team name, repo URL,
  GitHub handle and acknowledgements, and link to an anonymised mirror of the repo. The arXiv preprint is allowed to
  exist, but do not cite it as our own work in the anonymised version.
- **Other teams.** Refer to them by leaderboard rank only, as in the outline and the repo.
- **Numbers.** Every number must trace to a repo file: `results/official-scores.csv`, `docs/*.md` or
  `research/sim-data/`. Mark anything not yet measured as pending, and never round in our favour.
- **Archival conflicts.** If TMLR is the long-term home, prefer non-archival workshops, or opt out of ACM Digital
  Library publication (EuroMLSys).
- **Artifacts.** Include the test suite (384 tests at revision 1), the figure script and the simulator. Apply for MLSys artifact badges.

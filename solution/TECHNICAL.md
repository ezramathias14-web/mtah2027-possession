# Frame-level possession as a three-state sequence model

*More than a Hack 2027 — Phase 1 technical document. Rename to `<team_name>.md` (or export to
`<team_name>.pdf`) before submitting.*

## 1. Approach

The match is decoded as one sequence of three states, **H**, **A** and **D**, with the Viterbi
algorithm. Each frame contributes evidence for each state, and **every change of state has a
price**:

| transition | cost (nats) | meaning |
|---|---|---|
| H ↔ A | ln 300 ≈ 5.7 | a Markov chain whose possessions last ~12 s |
| H/A ↔ D | ln 750 ≈ 6.6 | a chain whose live spells last ~30 s |

A hand-over is therefore something the other team has to *earn* with accumulated evidence, rather
than a per-frame decision that is smoothed afterwards. That is the core difference from the
baseline (nearest player plus a majority vote), and it is how the challenge's possession rules are
implemented (section 4).

We chose a hand-built generative sequence model over a learned classifier for two reasons. First,
we are not allowed to train on any labels of this match, and we used no labels from other matches.
Second, spellF1 rewards whole, correctly bounded possessions, and a decoder with explicit switch
costs optimises exactly that structure.

### Evidence per frame

**Ball control (H vs A)**, used only where the tracking sees the ball. A team gets control
evidence on a frame when all of the following hold:

- its nearest player is within **1.5 m** of the ball, and closer than any opponent;
- the ball moves slower than **10 m/s**, so it is not a pass or a shot in flight;
- the ball is below **1.5 m**, where its height is known.

If both teams are within 1.5 m, the nearer team gets half the evidence. Everything is done at team
level, because the 46 track slots are tracklets rather than players. As a result, evidence from
different team-mates adds up, and a completed passing sequence counts as control just as a long
dribble does.

**Dead ball (D vs alive)**:

- **Collective activity.** The median running speed of the tracked players, averaged over ±1 s.
  The evidence is linear around a pivot of **1.3 m/s** (walking pace) and clipped. Players walk at
  a stoppage and run during play, and this stays observable when the ball is not.
- **Ball out of play.** Applies while the ball is more than 1 m outside the lines. *v2.1:* the
  evidence is also carried through a tracking gap that starts with the ball outside, for up to
  10 s, because the tracker often loses a ball as soon as it leaves the pitch.
- **Ball moving fast inside the pitch.** Weak evidence for play.

Control evidence is neutral between alive and dead: a player standing over a dead ball is still
near it.

**Team shape (H vs A, v2)**, used only where the ball is missing. See section 3.

**Smart Tagging (H vs A)**, used only where the ball is missing, with a very small weight
(0.02 nats per frame):

- possession phases (Build Up, Progression, Creating Chances, Attacking Transition) count for the
  tagged team;
- defending phases (Defending in …, Defensive Transition) count for the other team.

### How the constants were chosen

No constant was chosen on the public five minutes. Every value comes from football or from the
meaning of the cost:

- the two transition costs are log-odds of the stated mean durations;
- 1.5 m is playing distance;
- 1.3 m/s is walking pace;
- the shape evidence is scaled by a measured autocorrelation time (section 3).

The one number with no football value is **how many seconds of uncontested control pay for one
hand-over**. It is chosen **without labels** by a self-consistency check. We decode the full match
for each candidate (0.25 to 2 s), then keep the candidate whose decoded mean possession length
matches the 12 s the switch cost assumes. A fixed 1 s gave ~29 s possessions, i.e. a model
contradicting its own prior. The check picks 0.25 s, the lowest candidate, which still yields
~16 s. Section 6 explains what that tells us.

## 2. Assets used, and assets dropped

| asset | used for | notes |
|---|---|---|
| Tactical ATD (players + ball) | control, activity, out-of-play, team shape | the backbone |
| Tactical ATD metadata | track slot → team | first team listed is home |
| Smart Tagging | weak H/A prior in ball gaps | coarse phases, often spanning several possessions |
| Homography | **only in the parked detector** (pitch ↔ pixel) | not used by the submitted model |
| Tactical video | **only in the parked detector** | see below |
| Broadcast video, ATD, homography | not used | own timeline, cuts and replays, 34% of frames without homography |
| Public 5-minute ground truth | pipeline check only | never for training, tuning or choosing between versions |

**Dropped: a camera-specific ball detector** (`ball_detector.py`, kept in the repository and off by
default). Of the 64,643 graded frames where the tracker has no ball, 69% fall in stoppages our
model already labels D. Only ~20,000 frames, about 13 minutes, are ball-missing during live play.

To fill them we did the following:

1. Trained YOLO11n (Ultralytics, from public COCO weights) on 320 px native-resolution crops.
2. Took the labels from the tracker's own ball, projected into the video through the homography.
   These are automatic pseudo-labels, with no human involved.
3. Held out every 5th minute for validation.

After 15 epochs on CPU it reached **mAP50 0.022** on the held-out minutes, which is not usable. An
off-the-shelf COCO yolo11s found the tracker's ball as its top on-pitch detection on about half of
the held-out frames, with ~60% precision. That is too many false balls, and each one would buy a
spurious hand-over. The likely cause is a 4–6 px ball whose pseudo-labels are a few pixels off.
An IoU-trained detector cannot learn from that.

## 3. Training data

**No human-labelled data of any kind, for this match or any other.** Two components learn from
pseudo-labels that an algorithm produced, which RULES.md allows:

- **Team-shape model (v2).** A ridge logistic regression with no intercept, fitted on the frames
  where the ball is visible and v1 decoded H or A. Its six features are antisymmetric: swapping
  the teams and mirroring the pitch negates each one. The features are:
  - block position (mean x of all players);
  - mean x-velocity of all players;
  - difference in team spread along x;
  - difference in team spread along y;
  - difference in defensive-line height;
  - difference in team length.

  With no intercept, the model cannot learn "Barcelona usually has the ball", only what possession
  looks like for either side. It is **cross-fitted**: the model applied to each half is trained on
  the other half. Held-out agreement with the pseudo-labels is **81% and 78%**. Each frame's logit
  is divided by the integrated autocorrelation time of its **within-possession residual** (~7 s),
  so that a long gap is not counted as hundreds of independent observations.
- **Ball detector (dropped).** Trained on tracker ball positions, as described in section 2.

## 4. Possession rules implemented

- **Contested and loose balls stay with the last team in possession.** This is not a rule we
  wrote; it falls out of the switch cost. A touch, a deflection or a lost aerial duel adds too
  little control evidence to pay for a hand-over.
- **A change of possession needs controlled possession** by the other team: about 0.25 s or more
  of net uncontested control, accumulated across team-mates. A short pass sequence or a reception
  plus a touch qualifies.
- **Aerial balls and balls in flight** (height above 1.5 m, or speed above 10 m/s) give no control
  evidence to anyone.
- **Dead ball** comes from the players stopping or the ball being out of play. It is never a
  possession, and the side owning a dead ball is not modelled, as the challenge asks.
- **Restarts.** D → H and D → A cost the same, so the restarting team is whichever team the
  evidence after the restart supports.
- **Halves** are decoded independently. Frames outside the graded halves are written `X`. Inside
  the graded halves the model always answers H, A or D.

## 5. Results

On the full match, without labels, v2 decodes **237 possessions** with a mean of **15.7 s**. 86
stoppages leave **58.6%** of graded time alive, in line with typical effective playing time.

Two label-free checks against line crossings visible in the tracking:

- in 34 of 46 crossings, a predicted D starts within 10 s;
- the median D onset is **0.1 s** after the crossing.

Public 5-minute window, reported as a **pipeline check only**. It holds 17 possessions, and the
bootstrap noise band is about ±0.33 spellF1.

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| this model (v2) | 0.796 | 0.438 | 15 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

The macroF1 gain mostly comes from dead-ball detection: alive/dead F1 is 0.94 here against 0.90
for the baselines, and D recall is 68% against 8–34%. The spellF1 differences are within noise.
Only the private full-match score can rank this model.

## 6. Limitations, and what we would do with more time

- **Blind stretches.** 41% of graded frames have no ball, in gaps of ~11 s median. There is no
  control evidence there, so possession can only change at the edges of a gap. The self-calibration
  hits its lowest candidate and still decodes ~16 s possessions against the 12 s prior. That is the
  model saying it misses hand-overs, and they are mostly inside these gaps.
- **Team shape is weak evidence.** It agrees with the pseudo-labels ~80% of the time, but only
  "the team in possession plays wider" holds in both halves. At its honest weight it moves ~2% of
  frames.
- **Late restarts.** In a quarter of stoppages the predicted D ends ~4 s before the restart kick,
  because players start moving before the ball does.
- **Pseudo-labels inherit tracker errors.** This applies to both the shape model and the detector.
- **With more time:** a working ball detector (2× upscaled crops, looser boxes, GPU training);
  using the broadcast feed to fill tactical ball gaps after aligning the two timelines; and a
  restart detector that waits for the ball to move.

## 7. Reproduction

Python 3.11. The submitted model needs only numpy and pandas (`requirements.txt` at the repository
root). The match data stays outside the repository: point `MTAH_DATA_DIR` at the unpacked bundle.

```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/mtah2027             # PowerShell: $env:MTAH_DATA_DIR = "..."
python solution/possession_hmm.py                  # -> evaluation/results/possession_hmm_v2.csv
python evaluation/validate_submission.py evaluation/results/possession_hmm_v2.csv
```

The run is deterministic and takes about 20 s on a laptop. The first run also parses the raw ATD
into a cache under `$MTAH_DATA_DIR/cache`. Two consecutive runs produce byte-identical files.

`--stage 1` writes the model without team shape. The parked detector has its own pinned
requirements (`solution/requirements-detector.txt`, CPU PyTorch + Ultralytics) and is not needed
for the submission.

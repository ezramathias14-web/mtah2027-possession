"""Possession as a three-state sequence model: H, A or D, decoded with Viterbi.

The baseline decides every frame on its own and then smooths away the flicker. This does the
opposite: it treats the match as a sequence of states and asks for the cheapest path through it,
where every change of state has a price. A possession is then something that has to be *earned* by
accumulated evidence, which is what the challenge means by "controlled":

  - H <-> A costs C_SWITCH. The other team has to show a fraction of a second to a second of
    uncontested control of the ball (calibrated, see below) before the path pays for a hand-over. A touch, a deflection or a lost aerial duel does
    not buy enough evidence, so possession stays with the last team that had it -- the rule in
    CHALLENGE.md, implemented as a price instead of an if-statement.
  - alive <-> dead costs C_DEAD. A stoppage needs several seconds of "nobody is playing" evidence.

    pip install -r requirements.txt          # numpy and pandas, nothing else
    python solution/possession_hmm.py --stage 1   # -> evaluation/results/possession_hmm_v1.csv
    python solution/possession_hmm.py             # + team shape: the submission   ..._v2.csv
    python solution/possession_hmm.py --stage 3   # + video ball -- experimental, parked: the
                                                  #   detector in ball_detector.py does not yet
                                                  #   find the ball reliably enough to use

Evidence, per frame
-------------------
Control (H vs A), only where the tracking sees the ball:
  a team's nearest player is within R_CONTROL of the ball, closer than any opponent, the ball is
  slow enough to be at someone's feet and not high in the air. Team-level on purpose: the 46 track
  slots are tracklets, not people, so "a team keeps the ball" is the only statement the tracking
  can support. Evidence from different team-mates adds up, so a completed pass sequence counts as
  control just as a long dribble does.

Dead ball (D vs alive):
  - collective activity -- the median running speed of the tracked players. Players walk during a
    stoppage and run during play; this is visible even when the ball is not, which matters
    because the ball is missing on 41% of graded frames.
  - the ball outside the pitch lines.
  - the ball moving fast inside the pitch, weakly, as evidence of play.
  Control evidence is deliberately neutral between alive and dead: a player standing over the
  ball before a restart is "near the ball" too.

Team shape (H vs A), only where the ball is missing -- v2, see shape_evidence.py:
  a self-trained model of what possession looks like in the positions of the 22 players, learned
  from v1's own decoding of the ball-visible frames.

Smart Tagging (H vs A), only where the ball is missing:
  phases whose team tag implies possession (Build Up, Progression, ...) or its absence (Defending
  in ..., Defensive Transition). Tiny weight, because the phases are automatic, coarse and often
  span several possessions -- they can carry a long blind stretch, never overrule the ball.

How the numbers were chosen
---------------------------
Not on the public five minutes -- RULES.md forbids it and EVALUATION.md shows it picks the wrong
answer anyway. Every constant below is set from football and from the meaning of the costs:

  - Viterbi costs are log-odds. C_SWITCH = ln(300) is the switch price of a Markov chain whose
    possessions last 12 s on average (300 frames), the usual order of magnitude for an elite match.
    C_DEAD = ln(750) is the price of a chain whose alive spells last 30 s.
  - The control weight -- how many seconds of uncontested control pay one switch -- is the one
    number with no obvious football value, so calibrate() picks it without labels: the candidate
    whose decoded full match has the 12 s mean possession the switch cost assumes. A fixed 1 s
    decoded ~29 s possessions, i.e. a model contradicting its own prior.
  - ACTIVITY_PIVOT 1.3 m/s is walking pace; an outfield player averages ~1.8-2 m/s over a match
    and walks or stands at a stoppage.

The public window is used only to check that the pipeline is correct -- run the scorer.
"""

import argparse
import io
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import HOME, AWAY, team_of_each_track, write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402
from solution.shape_evidence import shape_evidence  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_DIR = os.path.join(DATA_DIR, "cache")  # derived from the gated data, so it lives with it
N_SLOTS = 46
FPS = 25

# Viterbi transition costs, in nats.
C_SWITCH = float(np.log(300))   # H <-> A: possessions of ~12 s
C_DEAD = float(np.log(750))     # alive <-> D: alive spells of ~30 s

# Control evidence.
R_CONTROL = 1.5        # m -- within playing distance of the ball
V_CONTROL = 10.0       # m/s -- faster than this the ball is a pass or a shot, not at someone's feet
Z_CONTROL = 1.5        # m -- above this (where the height is known) it is an aerial ball
CONTESTED_FACTOR = 0.5  # both teams within R_CONTROL: half the evidence to the nearer one

# How many seconds of uncontested control pay for one hand-over. Not fixed by hand: calibrate()
# picks the candidate whose decoded match has the mean possession length the transition cost
# assumes (PRIOR_POSSESSION_S). That is a self-consistency check on the model -- the costs say
# "possessions last ~12 s", the evidence weight should not decode a match that says otherwise --
# and it reads no labels of any kind.
PRIOR_POSSESSION_S = 12.0
CONTROL_SECONDS_CANDIDATES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

# Dead-ball evidence.
ACTIVITY_PIVOT = 1.3    # m/s -- median player speed at which play and stoppage are equally likely
ACTIVITY_GAIN = 0.2     # nats per frame per m/s away from the pivot
ACTIVITY_CLIP = 1.0     # m/s -- cap so one tracker glitch cannot dominate
ACTIVITY_WINDOW = FPS   # frames, half-width of the centred mean
MIN_TRACKED = 8         # fewer players than this and the activity reading is not trusted
OUT_MARGIN = 1.0        # m beyond the line before "out" counts, for tracking noise
OUT_GAIN = 0.5          # nats per frame towards D while the ball is out
OUT_CARRY_FRAMES = 10 * FPS  # a ball last seen out stays out through a gap, up to 10 s
MOVING_BALL = 4.0       # m/s -- a ball moving this fast inside the pitch suggests play
MOVING_GAIN = 0.05      # nats per frame towards alive

# Smart Tagging evidence, only where the ball is missing.
TAG_GAIN = 0.02         # nats per frame: 10 s of a tagged phase is ~1 switch
TAG_TEAMS = {"Barcelona": HOME, "Atletic Club": AWAY}
TAG_IN_POSSESSION = ("build up", "progression", "creating chances", "attacking transition")
TAG_OUT_OF_POSSESSION = ("defending in attacking third", "defending in middle third",
                         "defending in defensive third", "defensive transition")

# Video ball detections (v3), see ball_detector.py. The thresholds are read off the detector's
# precision against the tracker ball on held-out minutes -- automatic labels, not ground truth.
DETECTIONS = os.path.join(CACHE_DIR, "ball_detections.npz")
DETECTION_SURE_CONF = 0.5   # trusted alone
DETECTION_MIN_CONF = 0.2    # trusted with a neighbour that agrees
DETECTION_MAX_SPEED = 30.0  # m/s, a hard shot
DETECT_GAP = 2              # frames between searched frames (ball_detector.DETECT_STRIDE)

STATES = np.array(["H", "A", "D"])


def load_tracking(raw_path, n_frames):
    """(players, ball) arrays from the raw ATD: players (n, 46, 3) x/y/speed, ball (n, 3) x/y/z.

    x and y are normalised 0-1 as in the file. Parsed with pandas' C reader after turning both
    separators into commas -- seconds rather than the minutes a Python loop takes -- and cached
    next to the data, keyed on the raw file's size and timestamp so a new export is re-read.
    """
    stat = os.stat(raw_path)
    key = "{}_{}_{}".format(os.path.basename(raw_path), stat.st_size, int(stat.st_mtime))
    cache = os.path.join(CACHE_DIR, key + ".npz")
    if os.path.isfile(cache):
        arrays = np.load(cache)
        return arrays["players"], arrays["ball"]

    import pandas as pd

    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).to_numpy()
    if table.shape != (n_frames, 1 + 3 * N_SLOTS + 3):
        raise ValueError("{}: unexpected shape {}".format(raw_path, table.shape))
    if not np.array_equal(table[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{}".format(raw_path, n_frames))

    players = table[:, 1:1 + 3 * N_SLOTS].reshape(n_frames, N_SLOTS, 3)
    ball = table[:, 1 + 3 * N_SLOTS:]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, players=players, ball=ball)
    return players, ball


def fill_ball(ball, detections_path, pitch, sure_conf=DETECTION_SURE_CONF,
              min_conf=DETECTION_MIN_CONF):
    """The tracker ball with video detections (ball_detector.py) filled into its live gaps.

    A candidate is kept if it is confident on its own (>= sure_conf), or if it is at least
    min_conf and a candidate on a neighbouring searched frame sits where a ball could have
    travelled in between -- a false positive on the grass rarely has a partner that agrees with
    it, a real ball almost always does. Kept detections on frames two apart are joined linearly
    so the frame between them is filled too. Height stays unknown (NaN) for every filled frame.
    """
    length, width = pitch
    found = np.load(detections_path)
    frames, cands = found["frames"], found["candidates"].astype(np.float64)
    reach = DETECTION_MAX_SPEED / FPS  # m per frame

    def neighbour_agrees(i, k):
        x, y = cands[i, k, 0] * length, cands[i, k, 1] * width
        for j in (i - 1, i + 1):
            if not 0 <= j < len(frames) or abs(int(frames[j]) - int(frames[i])) > 2 * DETECT_GAP:
                continue
            gap = abs(int(frames[j]) - int(frames[i]))
            others = cands[j][np.isfinite(cands[j, :, 2]) & (cands[j, :, 2] >= min_conf)]
            if len(others) and np.min(np.hypot(others[:, 0] * length - x,
                                               others[:, 1] * width - y)) <= reach * gap + 1.0:
                return True
        return False

    kept = np.full((len(frames), 2), np.nan)
    for i in range(len(frames)):
        for k in range(cands.shape[1]):
            score = cands[i, k, 2]
            if not np.isfinite(score) or score < min_conf:
                continue
            if score >= sure_conf or neighbour_agrees(i, k):
                kept[i] = cands[i, k, :2]
                break

    filled = ball.copy()
    has = np.isfinite(kept[:, 0])
    filled[frames[has], 0], filled[frames[has], 1] = kept[has, 0], kept[has, 1]
    for i in range(len(frames) - 1):
        gap = int(frames[i + 1]) - int(frames[i])
        if has[i] and has[i + 1] and 1 < gap <= DETECT_GAP:
            for step in range(1, gap):
                f = int(frames[i]) + step
                if not np.isfinite(ball[f, 0]):
                    filled[f, :2] = kept[i] + (kept[i + 1] - kept[i]) * step / gap
    return filled, int(has.sum()), len(frames)


def centred_mean(values, window):
    """Mean over +/- window frames, ignoring NaN. NaN where the whole window is NaN."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window)
    high = np.minimum(len(values), index + window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def ball_speed(bx, by):
    """Ball speed in m/s from a centred difference over +/- 2 frames, NaN where it is unknown."""
    speed = np.full(len(bx), np.nan)
    step = 2
    dx = bx[2 * step:] - bx[:-2 * step]
    dy = by[2 * step:] - by[:-2 * step]
    speed[step:-step] = np.hypot(dx, dy) * FPS / (2 * step)
    return speed


def control_evidence(players, ball, team, pitch):
    """(n, 2) per-frame indicator of control for H and A, 1 for clear, CONTESTED_FACTOR for
    contested, 0 for none. Scaled to nats by the caller, once the weight is calibrated."""
    length, width = pitch
    bx, by, bz = ball[:, 0] * length, ball[:, 1] * width, ball[:, 2]
    px, py = players[:, :, 0] * length, players[:, :, 1] * width

    distance = np.hypot(px - bx[:, None], py - by[:, None])
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = np.stack([np.where(team[None, :] == side, distance, np.inf).min(axis=1)
                        for side in (HOME, AWAY)], axis=1)

    speed = ball_speed(bx, by)
    playable = np.isfinite(bx) & ~(speed > V_CONTROL) & ~(bz > Z_CONTROL)

    evidence = np.zeros((len(ball), 2))
    for side in (HOME, AWAY):
        mine, theirs = nearest[:, side], nearest[:, 1 - side]
        close = playable & (mine <= R_CONTROL) & (mine < theirs)
        contested = theirs <= R_CONTROL
        evidence[:, side] = np.where(close, np.where(contested, CONTESTED_FACTOR, 1.0), 0.0)
    return evidence, speed


def last_seen_outside(visible, outside, max_frames=OUT_CARRY_FRAMES):
    """Frames inside a tracking gap whose last sighting of the ball was outside the pitch.

    A ball that leaves the pitch is often lost by the tracker right after -- among the crowd,
    the advertising boards, the ball boys -- and the out-of-play evidence used to stop with it.
    It stays out until it is seen back in, so the evidence is carried through the gap, for at
    most max_frames: long enough to cover a typical restart, short enough that a missed re-entry
    does not hold the match dead.
    """
    carried = np.zeros(len(visible), bool)
    last_out, since = False, 0
    for t in range(len(visible)):
        if visible[t]:
            last_out, since = bool(outside[t]), 0
        else:
            since += 1
            carried[t] = last_out and since <= max_frames
    return carried


def dead_evidence(players, ball, pitch, speed):
    """Nats per frame for D over alive. Positive means stoppage."""
    length, width = pitch

    tracked = np.isfinite(players[:, :, 0])
    running = np.where(tracked, players[:, :, 2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows at halftime are expected
        activity = np.nanmedian(running, axis=1)
    activity[tracked.sum(axis=1) < MIN_TRACKED] = np.nan
    activity = centred_mean(activity, ACTIVITY_WINDOW)
    from_activity = np.clip(ACTIVITY_PIVOT - activity, -ACTIVITY_CLIP, ACTIVITY_CLIP)
    evidence = ACTIVITY_GAIN * np.nan_to_num(from_activity, nan=0.0)

    bx, by = ball[:, 0] * length, ball[:, 1] * width
    visible = np.isfinite(bx)
    outside = visible & ((bx < -OUT_MARGIN) | (bx > length + OUT_MARGIN)
                         | (by < -OUT_MARGIN) | (by > width + OUT_MARGIN))
    evidence[outside | last_seen_outside(visible, outside)] += OUT_GAIN

    moving = visible & ~outside & (speed > MOVING_BALL)
    evidence[moving] -= MOVING_GAIN
    return evidence, activity


def tagging_evidence(n_frames, ball_visible):
    """(n, 2) nats per frame for H and A from Smart Tagging phases, where the ball is missing."""
    evidence = np.zeros((n_frames, 2))
    for event in load_smart_tagging():
        code = event["code"].strip().lower()
        side = TAG_TEAMS.get(event["tags"].get("Team"))
        if side is None:
            continue
        if code in TAG_OUT_OF_POSSESSION:
            side = 1 - side
        elif code not in TAG_IN_POSSESSION:
            continue
        start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        evidence[start:end, side] += TAG_GAIN
    evidence[ball_visible] = 0.0
    return evidence


def viterbi(scores):
    """Cheapest path through H/A/D given per-frame scores (n, 3) in nats. Returns state indices.

    Transition costs: C_SWITCH between H and A, C_DEAD between either and D. Plain Python floats
    in the loop -- three states, so numpy per frame would only add overhead.
    """
    n = len(scores)
    cost = [[0.0, C_SWITCH, C_DEAD],
            [C_SWITCH, 0.0, C_DEAD],
            [C_DEAD, C_DEAD, 0.0]]
    back = np.zeros((n, 3), dtype=np.int8)
    value = [float(s) for s in scores[0]]
    rows = scores.tolist()
    for t in range(1, n):
        row, new = rows[t], [0.0, 0.0, 0.0]
        for j in range(3):
            best, arg = -1e18, 0
            for i in range(3):
                candidate = value[i] - cost[i][j]
                if candidate > best:
                    best, arg = candidate, i
            new[j] = best + row[j]
            back[t, j] = arg
        value = new
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(value))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(control, tags, dead, control_seconds, game=GAME):
    """Labels for the whole video with control evidence weighted for `control_seconds`.

    `tags` is every H-vs-A prior that is not the ball: Smart Tagging and, in v2, team shape.
    D is credited with the larger control score (a player over a dead ball is still near it) but
    only the mean of the priors -- crediting D with their max too would make a stoppage the cheap
    way out whenever a prior disagreed with the team in possession.
    """
    control = control * (C_SWITCH / (control_seconds * FPS))
    team_scores = control + tags
    scores = np.column_stack([team_scores[:, HOME], team_scores[:, AWAY],
                              dead + control.max(axis=1) + tags.mean(axis=1)])

    labels = np.full(len(scores), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = STATES[viterbi(scores[first:last + 1])]
    return labels


def summary(labels, game=GAME):
    """(possessions, mean possession seconds, stoppages, alive share) from the labels alone,
    counting runs within each half so a possession cannot straddle halftime."""
    n_poss = n_dead = n_alive = n_graded = 0
    for _, first, last in half_video_blocks(game):
        half = labels[first:last + 1]
        kinds = half[np.concatenate(([0], np.flatnonzero(half[1:] != half[:-1]) + 1))]
        n_poss += int(np.sum(kinds != "D"))
        n_dead += int(np.sum(kinds == "D"))
        n_alive += int(np.sum(half != "D"))
        n_graded += len(half)
    return n_poss, n_alive / max(n_poss, 1) / FPS, n_dead, n_alive / n_graded


def calibrate(control, tags, dead, game=GAME):
    """The control weight whose decoded match agrees with PRIOR_POSSESSION_S. No labels read."""
    best = None
    for seconds in CONTROL_SECONDS_CANDIDATES:
        labels = decode(control, tags, dead, seconds, game)
        n_poss, mean_s, _, _ = summary(labels)
        gap = abs(np.log(mean_s / PRIOR_POSSESSION_S))
        print("  control {:>4} s per switch -> {:>4} possessions, mean {:5.1f} s".format(
            seconds, n_poss, mean_s))
        if best is None or gap < best[0]:
            best = (gap, seconds, labels)
    print("  chosen: {} s".format(best[1]))
    return best[2]


def predict(game=GAME, data_dir=None, stage=2):
    """One label per video frame: H/A/D inside the graded halves, X outside them.

    v1 is the ball, the activity and the tagging. v2 decodes v1 first, learns team shape from
    v1's ball-visible frames (shape_evidence.py) and decodes again with that evidence filling the
    frames where the ball is missing. v3 adds the ball the video detector found in v2's live gaps
    (ball_detector.py, run beforehand) and decodes once more.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]

    players, ball = load_tracking(raw_path, n_frames)
    team = team_of_each_track(meta_path)
    visible = np.isfinite(ball[:, 0])

    control, speed = control_evidence(players, ball, team, pitch)
    dead, _ = dead_evidence(players, ball, pitch, speed)
    tags = tagging_evidence(n_frames, visible)

    print("v1: ball, activity, tagging")
    labels = calibrate(control, tags, dead, game)
    if stage < 2:
        return labels

    shape, report = shape_evidence(players, team, half_video_blocks(game), pitch, labels, visible,
                                   home=HOME)
    print("v2: team shape learned from v1's ball-visible frames")
    print("  held-out agreement with v1 per half: {}".format(
        ", ".join("{:.1%}".format(a) for a in report["agreement"])))
    print("  logit autocorrelation time: {:.0f} frames ({:.1f} s)".format(
        report["tau_frames"], report["tau_frames"] / FPS))
    for half, weights in enumerate(report["weights"], 1):
        print("  half {} model: {}".format(half, weights))
    labels = calibrate(control, tags + shape, dead, game)
    if stage < 3:
        return labels

    if not os.path.isfile(DETECTIONS):
        raise IOError("{} not found -- run solution/ball_detector.py detect first".format(
            DETECTIONS))
    filled, n_kept, n_searched = fill_ball(ball, DETECTIONS, pitch)
    seen = np.isfinite(filled[:, 0])
    print("v3: video ball detections in live gaps")
    print("  kept a detection on {} of {} searched frames; ball now known on {} more frames"
          .format(n_kept, n_searched, int(np.sum(seen & ~visible))))
    control, speed = control_evidence(players, filled, team, pitch)
    dead, _ = dead_evidence(players, filled, pitch, speed)
    priors = tagging_evidence(n_frames, seen) + np.where(seen[:, None], 0.0, shape)
    return calibrate(control, priors, dead, game)


def describe(labels):
    """Possession count and mean duration, from the submission alone -- no ground truth."""
    graded = labels[labels != "X"]
    n_poss, mean_s, stoppages, alive_share = summary(labels)
    print("graded frames {}  H {}  A {}  D {}".format(
        len(graded), int(np.sum(graded == "H")), int(np.sum(graded == "A")),
        int(np.sum(graded == "D"))))
    print("possessions {}  mean {:.1f} s  stoppages {}  alive share {:.1%}".format(
        n_poss, mean_s, stoppages, alive_share))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=2,
                        help="1: ball+activity+tagging, 2: + team shape (default), "
                             "3: + video ball (experimental)")
    args = parser.parse_args()

    labels = predict(stage=args.stage)
    describe(labels)
    name = "possession_hmm_v{}".format(args.stage)
    path = write_submission(os.path.join(args.out, name + ".csv"), labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("\nNow validate and score it:\n  python evaluation/validate_submission.py {}\n"
          "  python evaluation/run_evaluation.py".format(os.path.relpath(path, ROOT)))

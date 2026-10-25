"""Who has the ball when the tracking cannot see it: possession read off the shape of both teams.

The ball is missing on 41% of graded frames, in long blocks -- the median gap is ~11 s. v1 has no
evidence there and simply holds its last answer. But the 22 players are still tracked, and a team
with the ball does not look like a team without it: it stretches the pitch, pushes up, and the
whole block moves towards the opponent's goal while the other team drops and compacts.

This module learns that relationship from the match itself, without a single human label:

  1. Pseudo-labels. On frames where the ball IS visible and v1 decoded an alive state, v1's label
     is the target. These are produced by an algorithm, which RULES.md explicitly allows
     ("pseudo-labels produced by an algorithm, provided no person decided any of them").
  2. Features that do not depend on the ball: block position, team spread, team velocity in the
     direction each team attacks, line heights. All are *antisymmetric* -- swapping the two teams
     (and mirroring the pitch) flips their sign -- and the classifier has no intercept. So the
     model cannot learn "Barcelona usually has the ball"; it can only learn what possession looks
     like, for either side.
  3. Cross-fitting. The model applied to the first half is trained on the second, and vice versa,
     so no frame is scored by a model that saw its own pseudo-label.
  4. Evidence weight. A logistic model's logit is a log-odds for ONE frame, but consecutive frames
     are nearly copies of each other. Summing raw logits over a 10 s gap would count the same
     observation hundreds of times. Each frame's logit is therefore divided by the integrated
     autocorrelation time of its noise -- the logit's fluctuation within a pseudo-labelled run,
     measured on the match -- the standard correction for the effective sample size of a
     correlated series.

Used only where the ball is missing. Where it is visible the ball is better evidence, and the
pseudo-labels came from there in the first place.
"""

import warnings

import numpy as np

# Params
FPS = 25
VELOCITY_HALF_WINDOW = 12  # frames each side for per-slot displacement, ~1 s span in total
MIN_PER_TEAM = 6           # fewer tracked players than this and the frame has no shape evidence
LOGIT_CLIP = 4.0           # one frame never claims more than ~98% on shape alone
RIDGE = 1.0                # L2 penalty for the logistic fit, on standardised features
MAX_LAG = 2000             # frames searched for the autocorrelation time

FEATURES = ("block_x", "all_vx", "spread_x", "spread_y", "line_height", "length")


def attack_direction(players, team, first, last, home=0):
    """+1 if the home team attacks towards x = 1 in this half, -1 otherwise, read from the data:
    a team's mean position sits on its own side of the pitch over a half. No metadata needed,
    though for BAR-ATH it agrees with the ATD's attack_direction_first_half."""
    x = players[first:last + 1, :, 0]
    with np.errstate(all="ignore"):
        mean_home = np.nanmedian(np.nanmean(np.where(team[None, :] == home, x, np.nan), axis=1))
        mean_away = np.nanmedian(np.nanmean(np.where(team[None, :] != home, x, np.nan), axis=1))
    return 1.0 if mean_home < mean_away else -1.0


def slot_velocity(coordinate, half_window=VELOCITY_HALF_WINDOW):
    """Per-slot velocity (units per second) from a centred difference. NaN where the slot is not
    tracked at both ends -- slots appear and vanish, so a team centroid's own derivative would
    jump every time one does; averaging per-slot displacements does not."""
    velocity = np.full(coordinate.shape, np.nan, dtype=np.float32)
    step = half_window
    velocity[step:-step] = (coordinate[2 * step:] - coordinate[:-2 * step]) * FPS / (2 * step)
    return velocity


def features(players, team, blocks, pitch, home=0):
    """(n, len(FEATURES)) antisymmetric team-shape features, NaN where they cannot be read.

    Coordinates are turned so that +x is the direction the HOME team attacks, in metres. For every
    feature, swapping the teams and mirroring the pitch negates it.
    """
    length, width = pitch
    n = len(players)
    out = np.full((n, len(FEATURES)), np.nan)
    is_home = team == home

    for _, first, last in blocks:
        sign = attack_direction(players, team, first, last, home)
        x = players[first:last + 1, :, 0].astype(np.float64)
        y = players[first:last + 1, :, 1].astype(np.float64) * width
        x = (x if sign > 0 else 1.0 - x) * length
        vx = slot_velocity(x)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: halftime, gaps
            h_x, a_x = np.where(is_home, x, np.nan), np.where(~is_home, x, np.nan)
            h_y, a_y = np.where(is_home, y, np.nan), np.where(~is_home, y, np.nan)
            n_h, n_a = np.isfinite(h_x).sum(1), np.isfinite(a_x).sum(1)

            all_x = np.nanmean(x, axis=1)
            # Velocity enters once, for everyone. The antisymmetric difference of the two teams'
            # own-forward velocities, h_vx - (-a_vx), is just h_vx + a_vx again -- the same signal
            # twice, which made the fitted weights trade sign between halves.
            all_vx = np.nanmean(vx, axis=1)

            h_lo, h_hi = np.nanpercentile(h_x, [10, 90], axis=1)
            a_lo, a_hi = np.nanpercentile(a_x, [10, 90], axis=1)

            block = np.column_stack([
                all_x - length / 2,                                  # where play is
                all_vx,                                              # where everyone is going
                np.nanstd(h_x, axis=1) - np.nanstd(a_x, axis=1),     # who stretches the pitch
                np.nanstd(h_y, axis=1) - np.nanstd(a_y, axis=1),     # ... and its width
                h_lo - (length - a_hi),                              # defensive line heights
                (h_hi - h_lo) - (a_hi - a_lo),                       # team lengths
            ])
        block[(n_h < MIN_PER_TEAM) | (n_a < MIN_PER_TEAM)] = np.nan
        out[first:last + 1] = block
    return out


def fit_logistic(x, y, ridge=RIDGE, iterations=25):
    """Ridge logistic regression with no intercept, by Newton's method. numpy only.

    No intercept is a modelling decision, not an omission: with antisymmetric features it makes
    p(home | shape) = 1 - p(away | mirrored shape), so the model has no team prior to lean on.
    """
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        gradient = x.T @ (p - y) + ridge * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def autocorrelation_time(series, max_lag=MAX_LAG):
    """Integrated autocorrelation time in frames: 1 + 2 * sum of autocorrelations up to the
    first non-positive one. NaN-tolerant by working on the finite stretches only."""
    finite = np.isfinite(series)
    values = np.where(finite, series - np.nanmean(series), 0.0)
    variance = np.sum(values[finite] ** 2) / finite.sum()
    tau = 1.0
    for lag in range(1, max_lag):
        both = finite[lag:] & finite[:-lag]
        if both.sum() < 100:
            break
        rho = np.sum(values[lag:][both] * values[:-lag][both]) / both.sum() / variance
        if rho <= 0:
            break
        tau += 2 * rho
    return tau


def within_run_residual(logit, labels):
    """The logit minus its mean over each run of one pseudo-label, NaN outside alive runs.

    The redundancy to correct for is in the *noise*, not in the state. The raw logit stays high
    for as long as a possession lasts, so its autocorrelation time would mostly measure possession
    length and shrink the evidence to nothing. Within a run the state is fixed, so what is left is
    the observation noise whose correlation decides how many independent looks a gap holds.
    """
    out = np.full(len(logit), np.nan)
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [len(labels)]))
    for start, end in zip(starts[:-1], starts[1:]):
        if labels[start] not in ("H", "A"):
            continue
        run = logit[start:end]
        if np.isfinite(run).sum() > 1:
            out[start:end] = run - np.nanmean(run)
    return out


def shape_evidence(players, team, blocks, pitch, pseudo_labels, ball_visible, home=0):
    """(n, 2) nats per frame for H and A from team shape, non-zero only where the ball is missing.

    pseudo_labels is v1's decoded output ("H"/"A"/"D"/"X" per frame); only its alive frames with
    the ball visible are used for training. Returns (evidence, report) where report holds the
    label-free numbers worth printing: held-out agreement with the pseudo-labels, tau, weights.
    """
    shape = features(players, team, blocks, pitch, home)
    usable = np.all(np.isfinite(shape), axis=1)
    target = np.isin(pseudo_labels, ("H", "A")) & ball_visible & usable
    scale = np.nanstd(shape[usable], axis=0)
    shape = shape / scale

    logit = np.full(len(players), np.nan)
    report = {"agreement": [], "weights": []}
    for i, (_, first, last) in enumerate(blocks):
        here = np.zeros(len(players), bool)
        here[first:last + 1] = True
        train = target & ~here
        w = fit_logistic(shape[train], (pseudo_labels[train] == "H").astype(float))
        logit[here & usable] = shape[here & usable] @ w
        held_out = target & here
        report["agreement"].append(
            float(np.mean((logit[held_out] > 0) == (pseudo_labels[held_out] == "H"))))
        report["weights"].append(dict(zip(FEATURES, np.round(w, 3))))

    logit = np.clip(logit, -LOGIT_CLIP, LOGIT_CLIP)
    tau = autocorrelation_time(within_run_residual(logit, pseudo_labels))
    report["tau_frames"] = tau

    per_frame = np.nan_to_num(logit, nan=0.0) / tau
    per_frame[ball_visible] = 0.0
    evidence = np.zeros((len(players), 2))
    evidence[:, home] = np.maximum(per_frame, 0.0)
    evidence[:, 1 - home] = np.maximum(-per_frame, 0.0)
    return evidence, report

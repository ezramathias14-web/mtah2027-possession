"""A ball detector for this camera, trained on the tracker's own ball -- no human labels.

STATUS: parked. The first training run (yolo11n, 15 epochs, 320 px native crops, 8 px boxes)
reached mAP50 ~0.02 on the held-out minutes -- not usable. Most likely cause: a 4-6 px ball with
labels a few pixels off, which an IoU-based detector cannot learn from. Next attempt, if any:
upscale crops 2x and use looser boxes. possession_hmm.py does not use it by default.

The provided tracking sees the ball on 59% of graded frames. Where it does, the ball position
projected through the homography is an automatic label for the video: the pixel where the ball is.
That is enough to train a detector that knows this camera's ball -- its size, its blur, the grass
it rolls on -- which a COCO "sports ball" detector does not (yolo11s finds it on about half of
the frames where the tracker does; `evaluate --weights yolo11s.pt` measures it). The detector then
looks at the frames where the tracker lost the ball during play.

RULES.md allows this explicitly: pseudo-labels produced by an algorithm, no person deciding any of
them. Every step runs locally and prints aggregate numbers only -- the competition terms forbid
sharing video or extracts of it with any hosted service.

    pip install -r solution/requirements-detector.txt
    python solution/ball_detector.py dataset      # crops + labels from ball-visible frames
    python solution/ball_detector.py train        # yolo11n on CPU, ~2 h on a laptop
    python solution/ball_detector.py evaluate     # recall/precision on held-out minutes
    python solution/ball_detector.py detect       # live ball-missing frames -> detections.npz

Everything it writes goes under DATA_DIR/cache, next to the data it was derived from, never into
the repository.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DATASET_DIR = os.path.join(CACHE_DIR, "ball_dataset")
TRAIN_DIR = os.path.join(CACHE_DIR, "ball_training")
MODELS_DIR = os.path.join(CACHE_DIR, "models")
WEIGHTS = os.path.join(MODELS_DIR, "ball_yolo11n.pt")
DETECTIONS = os.path.join(CACHE_DIR, "ball_detections.npz")
LABELS_CSV = os.path.join(ROOT, "evaluation", "results", "possession_hmm_v2.csv")

FPS = 25
SEED = 0
CROP = 320               # px, native resolution: the detector is trained and run at one scale
N_POSITIVE = 2400        # ball-visible frames sampled for training crops
NEGATIVES_PER_FRAME = 1  # a second crop from the same frame, away from the ball
EDGE_MARGIN = 16         # px the ball keeps from a crop edge
NEGATIVE_CLEARANCE = 48  # px between a negative crop and the ball
VAL_EVERY_MINUTES = 5    # every 5th minute of video is held out
BALL_DIAMETER = 0.22     # m
BOX_SCALE = 1.5          # label box relative to the projected diameter, for projection error
MIN_BOX = 8              # px
MAX_LABEL_SPEED = 25.0   # m/s -- faster balls are streaks and probably airborne
MAX_LABEL_HEIGHT = 0.5   # m, where the tracker reports a height

EPOCHS = 15             # from COCO weights on ~4k crops; ~8 min per epoch on a 4-core laptop
BATCH = 32
INFER_SIZE = 1920        # full frame at native scale
PITCH_MARGIN = 1.0       # m outside the lines a detection may sit
MATCH_RADIUS = 1.5       # m between a detection and the tracker ball to call it the same ball
DETECT_STRIDE = 2        # every 2nd live ball-missing frame
TOP_K = 3                # candidates kept per frame for the temporal filter
COCO_SPORTS_BALL = 32


def load_homography(path):
    """(n, 3, 3) matrices mapping normalised pitch coordinates to pixels."""
    import pandas as pd

    table = pd.read_csv(path)
    columns = ["h{}{}".format(i, j) for i in range(3) for j in range(3)]
    return table[columns].to_numpy().reshape(-1, 3, 3)


def project(matrix, x, y):
    """Normalised pitch (x, y) -> pixel (u, v)."""
    p = matrix @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def unproject(matrix, u, v):
    """Pixel (u, v) -> normalised pitch (x, y), assuming the point is on the ground."""
    p = np.linalg.solve(matrix, np.array([u, v, 1.0]))
    return p[0] / p[2], p[1] / p[2]


def ball_pixels(matrix, x, y, pitch):
    """Projected ball diameter in pixels at a pitch position."""
    length, width = pitch
    u, v = project(matrix, x, y)
    du = np.hypot(*np.subtract(project(matrix, x + BALL_DIAMETER / length, y), (u, v)))
    dv = np.hypot(*np.subtract(project(matrix, x, y + BALL_DIAMETER / width), (u, v)))
    return (du + dv) / 2


def is_validation(frame):
    return (frame // (FPS * 60)) % VAL_EVERY_MINUTES == 0


def load_inputs(game=GAME):
    from solution.possession_hmm import load_tracking

    meta, raw, homography, video = require(["atd_meta", "atd_raw", "homography", "video"], game)
    players, ball = load_tracking(raw, source_frames("tactical", game))
    return players, ball, load_homography(homography), video


def graded_mask(n, game=GAME):
    mask = np.zeros(n, bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def label_frames(ball, pitch, game=GAME):
    """Frames whose tracker ball makes a trustworthy label: graded, on the pitch, low and slow."""
    from solution.possession_hmm import ball_speed

    length, width = pitch
    x, y, z = ball[:, 0], ball[:, 1], ball[:, 2]
    speed = ball_speed(x * length, y * width)
    ok = graded_mask(len(ball), game) & np.isfinite(x)
    ok &= (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    ok &= ~(speed > MAX_LABEL_SPEED) & ~(z > MAX_LABEL_HEIGHT)
    return np.flatnonzero(ok)


def read_frames(video, frames):
    """Yield (frame, image) for sorted frame indices, reading forward and seeking only far ahead."""
    import cv2

    capture = cv2.VideoCapture(video)
    position = -1
    for frame in frames:
        if frame - position > 50 or frame <= position:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            position = frame - 1
        while position < frame:
            ok, image = capture.read()
            position += 1
            if not ok:
                raise IOError("could not read frame {}".format(position))
        yield frame, image
    capture.release()


def build_dataset(game=GAME):
    """Write YOLO-format crops: one positive and NEGATIVES_PER_FRAME negatives per sampled frame."""
    import cv2

    pitch = GAMES[game]["pitch"]
    _, ball, homography, video = load_inputs(game)
    rng = np.random.default_rng(SEED)
    frames = np.sort(rng.choice(label_frames(ball, pitch, game), N_POSITIVE, replace=False))

    if os.path.isdir(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(DATASET_DIR, kind, split))

    sizes, counts = [], {"train": 0, "val": 0}
    for frame, image in read_frames(video, frames):
        height, width = image.shape[:2]
        u, v = project(homography[frame], ball[frame, 0], ball[frame, 1])
        if not (EDGE_MARGIN <= u < width - EDGE_MARGIN and EDGE_MARGIN <= v < height - EDGE_MARGIN):
            continue
        box = max(MIN_BOX, BOX_SCALE * ball_pixels(homography[frame], ball[frame, 0],
                                                    ball[frame, 1], pitch))
        sizes.append(box)
        split = "val" if is_validation(frame) else "train"

        low_x, high_x = max(0, int(u) + EDGE_MARGIN - CROP), min(width - CROP, int(u) - EDGE_MARGIN)
        low_y, high_y = max(0, int(v) + EDGE_MARGIN - CROP), min(height - CROP, int(v) - EDGE_MARGIN)
        x0, y0 = int(rng.integers(low_x, high_x + 1)), int(rng.integers(low_y, high_y + 1))
        stem = "f{:06d}".format(frame)
        cv2.imwrite(os.path.join(DATASET_DIR, "images", split, stem + "_p.jpg"),
                    image[y0:y0 + CROP, x0:x0 + CROP])
        with open(os.path.join(DATASET_DIR, "labels", split, stem + "_p.txt"), "w") as handle:
            handle.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                (u - x0) / CROP, (v - y0) / CROP, box / CROP, box / CROP))
        counts[split] += 1

        for k in range(NEGATIVES_PER_FRAME):
            for _ in range(20):
                nx, ny = int(rng.integers(0, width - CROP)), int(rng.integers(0, height - CROP))
                clear_x = u < nx - NEGATIVE_CLEARANCE or u > nx + CROP + NEGATIVE_CLEARANCE
                clear_y = v < ny - NEGATIVE_CLEARANCE or v > ny + CROP + NEGATIVE_CLEARANCE
                if clear_x or clear_y:
                    name = "{}_n{}".format(stem, k)
                    cv2.imwrite(os.path.join(DATASET_DIR, "images", split, name + ".jpg"),
                                image[ny:ny + CROP, nx:nx + CROP])
                    open(os.path.join(DATASET_DIR, "labels", split, name + ".txt"), "w").close()
                    break

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(
            DATASET_DIR.replace("\\", "/")))
    print("positives: train {train}  val {val}".format(**counts))
    print("label box px: median {:.1f}  p10 {:.1f}  p90 {:.1f}".format(
        *np.percentile(sizes, [50, 10, 90])))


def train():
    from ultralytics import YOLO

    os.makedirs(MODELS_DIR, exist_ok=True)
    base = os.path.join(MODELS_DIR, "yolo11n.pt")
    model = YOLO(base if os.path.isfile(base) else "yolo11n.pt")
    started = time.time()
    model.train(data=os.path.join(DATASET_DIR, "data.yaml"), imgsz=CROP, epochs=EPOCHS,
                batch=BATCH, device="cpu", workers=2, seed=SEED, deterministic=True,
                project=TRAIN_DIR, name="ball", exist_ok=True, plots=False, verbose=False,
                # Mosaic is off: it rescales and stitches crops, and the point of training at
                # native resolution is that the ball keeps the one size it has in this video.
                fliplr=0.5, mosaic=0.0, scale=0.1)
    shutil.copy(os.path.join(TRAIN_DIR, "ball", "weights", "best.pt"), WEIGHTS)
    print("trained in {:.0f} min -> {}".format((time.time() - started) / 60, WEIGHTS))


def candidates(model, image, matrix, pitch, conf=0.05):
    """[(x, y, conf)] in normalised pitch coordinates, on-pitch detections only, best first."""
    length, width = pitch
    classes = [COCO_SPORTS_BALL] if len(model.names) > 1 else None  # a COCO model, for comparison
    result = model.predict(image, imgsz=INFER_SIZE, conf=conf, classes=classes, verbose=False,
                           device="cpu")[0]
    out = []
    for (cx, cy, _, _), score in zip(result.boxes.xywh.numpy(), result.boxes.conf.numpy()):
        x, y = unproject(matrix, cx, cy)
        if (-PITCH_MARGIN <= x * length <= length + PITCH_MARGIN
                and -PITCH_MARGIN <= y * width <= width + PITCH_MARGIN):
            out.append((x, y, float(score)))
    return sorted(out, key=lambda c: -c[2])


def evaluate(n_frames=300, game=GAME, weights=WEIGHTS):
    """Against the tracker ball on held-out minutes: is the top detection the same ball?"""
    from ultralytics import YOLO

    pitch = GAMES[game]["pitch"]
    length, width = pitch
    _, ball, homography, video = load_inputs(game)
    rng = np.random.default_rng(SEED + 1)
    held_out = [f for f in label_frames(ball, pitch, game) if is_validation(f)]
    frames = np.sort(rng.choice(held_out, n_frames, replace=False))

    model = YOLO(weights)
    top, started = [], time.time()
    for frame, image in read_frames(video, frames):
        found = candidates(model, image, homography[frame], pitch)
        if found:
            x, y, score = found[0]
            miss = np.hypot((x - ball[frame, 0]) * length, (y - ball[frame, 1]) * width)
            top.append((score, miss <= MATCH_RADIUS))
        else:
            top.append((0.0, False))
    seconds = (time.time() - started) / n_frames
    top = np.array(top)
    print("{} held-out frames, {:.2f} s/frame".format(n_frames, seconds))
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        kept = top[:, 0] >= threshold
        right = int(np.sum(kept & (top[:, 1] > 0)))
        print("  conf >= {:<4}: detection on {:5.1%} of frames, top one is the ball on {:5.1%} "
              "(precision {:5.1%})".format(threshold, kept.mean(), right / n_frames,
                                          right / max(kept.sum(), 1)))


def detect(game=GAME, weights=WEIGHTS):
    """Top-K candidates on every DETECT_STRIDE-th live frame the tracker has no ball on.

    "Live" is v2's decoding -- dead-ball frames are skipped because where the ball lies during a
    stoppage does not change the label.
    """
    import pandas as pd
    from ultralytics import YOLO

    pitch = GAMES[game]["pitch"]
    _, ball, homography, video = load_inputs(game)
    labels = pd.read_csv(LABELS_CSV)["label"].to_numpy()
    wanted = graded_mask(len(ball), game) & ~np.isfinite(ball[:, 0]) & np.isin(labels, ("H", "A"))
    frames = np.flatnonzero(wanted)
    frames = frames[frames % DETECT_STRIDE == 0]

    model = YOLO(weights)
    found = np.full((len(frames), TOP_K, 3), np.nan, dtype=np.float32)
    started = time.time()
    for i, (frame, image) in enumerate(read_frames(video, frames)):
        for k, candidate in enumerate(candidates(model, image, homography[frame], pitch)[:TOP_K]):
            found[i, k] = candidate
        if (i + 1) % 500 == 0:
            rate = (time.time() - started) / (i + 1)
            print("  {}/{} frames, {:.2f} s/frame, ~{:.0f} min left".format(
                i + 1, len(frames), rate, rate * (len(frames) - i - 1) / 60), flush=True)
    np.savez(DETECTIONS, frames=frames, candidates=found)
    with_any = np.isfinite(found[:, 0, 0])
    print("{} frames searched, a candidate on {:.1%}, saved {}".format(
        len(frames), with_any.mean(), DETECTIONS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("dataset", "train", "evaluate", "detect"))
    parser.add_argument("--weights", default=WEIGHTS,
                        help="for evaluate/detect, e.g. a COCO model for comparison")
    args = parser.parse_args()
    if args.step == "dataset":
        build_dataset()
    elif args.step == "train":
        train()
    elif args.step == "evaluate":
        evaluate(weights=args.weights)
    else:
        detect(weights=args.weights)

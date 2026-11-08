"""Analyze the hHv1 S1-S1' trajectories and plot Figures 4, S2, and S3.

Workflow (three stages)
  1. extract : MDAnalysis -> 80-ps Calpha-Calpha distance features (.npz)
  2. analyze : beta-VAE training (5 seeds x 3 betas), LDA, voting, gradient attribution, CSV tables, figures
  3. plot    : redraw Figure 4 / S2 / S3 from existing CSVs

Settings marked [manuscript] are specified in the revised manuscript.
The README records the implementation defaults for choices the manuscript
leaves open. Exact numerical reproduction requires the original run records.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

# ======================= USER-ADJUSTABLE PARAMETERS =======================
# Canonical hHv1 residue ranges (inclusive) of S1-S4.                  [manuscript]
HELICES = {"S1": (107, 124), "S2": (134, 151), "S3": (170, 188), "S4": (197, 220)}
S1_RANGE = HELICES["S1"]          # S1 residues are the "anchor" end of every distance feature
# Helix groups paired with S1; a trailing ' means the partner protomer.  [manuscript]
GROUPS = ("S2", "S3", "S4", "S1'", "S2'", "S3'", "S4'")
# Canonical residue n -> topology resid: protomer A = n-79, protomer B = n+108.   [manuscript]
OFFSETS = (-79, 108)
# Residue-name sanity check (canonical number -> 3-letter name): I121, A197, L201.
LANDMARKS = {121: "ILE", 197: "ALA", 201: "LEU"}
# Model-training random seeds (also determine the train/validation block split). [manuscript]
SEEDS = (11, 29, 47, 71, 101)
# beta (KL weight) values compared in Figure S2; PRIMARY_BETA is used for all
# downstream analysis (LDA, voting, attribution, Figures 4 and S3).       [manuscript]
BETAS = (0.1, 0.5, 1.0)
PRIMARY_BETA = 0.1
# Frame stride used for distance features (ps) and block length for train/validation split (ns). [manuscript]
STEP_PS = 80
BLOCK_NS = 20
# A test trajectory is "Path 1-like" for one seed if >= this fraction of its frames lie on
# the Path 1 side of the LDA boundary; the final call needs >= MIN_SEED_VOTES seeds agreeing. [manuscript]
PATH1_FRACTION_THRESHOLD = 0.80
MIN_SEED_VOTES = 3
# Default sampling windows (ps). The full reference window is configurable.
# The final 100 ns test window is specified in the manuscript.
REFERENCE_WINDOW_PS = (0, 500_000)
TEST_WINDOW_PS = (400_000, 500_000)
# IDs the manifest must use for the seven additional (test) trajectories.   [manuscript]
TEST_IDS = tuple(f"T{i}" for i in range(3, 10))
# Fraction of 20-ns reference blocks held out for validation: see --validation-fraction (default 0.2).
# Network / optimiser settings: see --epochs (300), --batch-size (128), --learning-rate (1e-3).
# ===========================================================================

BLOCK_FRAMES = BLOCK_NS * 1000 // STEP_PS
assert BLOCK_FRAMES > 0 and BLOCK_FRAMES * STEP_PS == BLOCK_NS * 1000, "A block must contain an integer number of sample intervals."
assert PRIMARY_BETA in BETAS, "PRIMARY_BETA must be one of BETAS"


def feature_map():
    for group in GROUPS:
        target = group.rstrip("'")
        for i in range(S1_RANGE[0], S1_RANGE[1] + 1):
            for j in range(HELICES[target][0], HELICES[target][1] + 1):
                yield group, i, j


FEATURES = tuple(feature_map())
N_FEATURES = len(FEATURES)  # 2,520 for the manuscript settings (18 x (18+19+24) + 18 x (18+18+19+24))


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    items = config["trajectories"]
    ids = [item["id"] for item in items]
    roles = [item["role"] for item in items]
    if len(set(ids)) != 2 + len(TEST_IDS) or sorted(roles) != ["path1", "path2"] + ["test"] * len(TEST_IDS):
        raise ValueError(f"Use {2 + len(TEST_IDS)} unique IDs: one path1, one path2, and {len(TEST_IDS)} test trajectories.")
    if {item["id"] for item in items if item["role"] == "test"} != set(TEST_IDS):
        raise ValueError(f"Use {TEST_IDS[0]} through {TEST_IDS[-1]} for the test trajectories.")
    return config


def topology_indices(universe):
    indices = {}
    for direction, offset in enumerate(OFFSETS):
        for (low, high) in HELICES.values():
            for number in range(low, high + 1):
                atoms = universe.select_atoms(f"name CA and resid {number + offset}")
                if len(atoms) != 1:
                    raise ValueError(
                        f"Expected exactly one CA for protomer {'AB'[direction]}, "
                        f"canonical residue {number}, topology resid {number + offset}."
                    )
                indices[direction, number] = atoms.indices[0]
    # Verify chain correspondence and residue-number landmarks from the manuscript.
    landmarks = LANDMARKS
    for direction, number in indices:
        if direction != 0:
            continue
        names = [universe.atoms[indices[d, number]].resname for d in (0, 1)]
        if names[0] != names[1] or names[0] != landmarks.get(number, names[0]):
            raise ValueError(f"The Amber residue mapping does not match canonical residue {number}: {names}.")
    starts, ends = [], []
    for d in (0, 1):
        starts.append([indices[d, i] for group, i, j in FEATURES])
        ends.append([indices[1 - d if group.endswith("'") else d, j] for group, i, j in FEATURES])
    return np.asarray(starts), np.asarray(ends)


def sample_indices(n_frames, first_ps, frame_dt_ps, window):
    start, end = map(float, window)
    if not all(math.isfinite(v) for v in (first_ps, frame_dt_ps, start, end)):
        raise ValueError("Time and frame spacing must be finite.")
    if frame_dt_ps <= 0 or start >= end:
        raise ValueError("Frame spacing must be positive and the sampling window must increase.")
    stride = round(STEP_PS / frame_dt_ps)
    if stride < 1 or not math.isclose(stride * frame_dt_ps, STEP_PS, abs_tol=1e-6):
        raise ValueError(f"An 80-ps sampling stride is incompatible with a {frame_dt_ps:g}-ps trajectory spacing.")
    count = round((end - start) / STEP_PS)
    if not math.isclose(count * STEP_PS, end - start, abs_tol=1e-6):
        raise ValueError("The sampling window must span an integer number of 80-ps intervals.")
    first = max(0, math.ceil((start - first_ps) / frame_dt_ps - 1e-8))
    indices = first + stride * np.arange(count)
    times = first_ps + frame_dt_ps * indices
    if not count or times[-1] > end + 1e-6 or indices[-1] >= n_frames:
        raise ValueError(f"Trajectory does not cover the requested {start:g}–{end:g} ps sampling window.")
    return indices, times


def extract(config_path, output):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_bonds

    config = load_config(config_path)
    base = config_path.resolve().parent
    pbc = config.get("pbc", "minimum_image")
    if pbc not in {"minimum_image", "whole"}:
        raise ValueError("pbc must be minimum_image or whole.")
    payload = {"config": np.array(json.dumps(config)), "mdanalysis": np.array(mda.__version__)}
    used_files = set()
    for k, item in enumerate(config["trajectories"]):
        files = item["files"]
        if not files:
            raise ValueError(f"No files specified for {item['id']}.")
        files = [(base / name).resolve() for name in files]
        if used_files.intersection(files) or len(set(files)) != len(files):
            raise ValueError(f"Duplicate trajectory segment in {item['id']}.")
        used_files.update(files)
        options = {"format": item["format"]} if item.get("format") else {}
        topology = item.get("topology") or config["topology"]
        topology_format = item.get("topology_format", config.get("topology_format"))
        if topology_format:
            options["topology_format"] = topology_format
        universe = mda.Universe(
            str(base / topology), [str(file) for file in files], **options
        )
        try:
            starts, ends = topology_indices(universe)
            window = (
                config.get("test_window_ps", TEST_WINDOW_PS)
                if item["role"] == "test"
                else config.get("reference_window_ps", REFERENCE_WINDOW_PS)
            )
            indices, times = sample_indices(
                len(universe.trajectory), float(item["first_ps"]), float(item["frame_dt_ps"]), window
            )
            x = np.empty((len(indices), 2, N_FEATURES), dtype=np.float32)
            for frame, ts in enumerate(universe.trajectory[indices]):
                box = ts.dimensions if pbc == "minimum_image" else None
                if pbc == "minimum_image" and (box is None or not np.all(np.isfinite(box)) or np.any(box <= 0)):
                    raise ValueError(f"{item['id']} frame {ts.frame} has no valid periodic cell.")
                x[frame] = calc_bonds(
                    ts.positions[starts.ravel()], ts.positions[ends.ravel()], box=box
                ).reshape(2, N_FEATURES)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Non-finite distances in {item['id']}.")
        finally:
            universe.trajectory.close()
        payload[f"x{k}"], payload[f"t{k}"] = x, times
        print(f"{item['id']}: {len(times)} frames, {times[0]:g}–{times[-1]:g} ps", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez(file, **payload)


def reference_split(trajectories, seed, fraction, recorded_blocks):
    rng = np.random.default_rng(seed)
    splits = {}
    for name, (x, _) in trajectories.items():
        n_blocks, remainder = divmod(len(x), BLOCK_FRAMES)
        if n_blocks < 2 or remainder:
            raise ValueError("Each reference must contain at least two complete 20-ns blocks.")
        if recorded_blocks is None:
            n_valid = min(n_blocks - 1, max(1, round(n_blocks * fraction)))
            valid_blocks = np.sort(rng.choice(n_blocks, n_valid, replace=False))
        else:
            blocks = recorded_blocks[name]
            if (not 0 < len(blocks) < n_blocks or len(set(blocks)) != len(blocks)
                    or any(type(block) is not int or not 0 <= block < n_blocks for block in blocks)):
                raise ValueError(f"Invalid validation block indices for {name}, seed {seed}.")
            valid_blocks = np.array(sorted(blocks))
        mask = np.repeat(np.isin(np.arange(n_blocks), valid_blocks), BLOCK_FRAMES)
        splits[name] = (mask, valid_blocks.tolist())
    return splits


def train_model(train, validation, torch, device, seed, beta, epochs, batch_size, lr):
    from torch import nn

    class BetaVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(N_FEATURES, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.mu = nn.Linear(128, 2)
            self.logvar = nn.Linear(128, 2)
            self.decoder = nn.Sequential(
                nn.Linear(2, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, N_FEATURES), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h)

    def loss(x, y, mu, logvar):
        reconstruction = (y - x).square().sum(dim=1).mean()
        return reconstruction + beta * kl_per_sample(mu, logvar).mean()

    torch.manual_seed(seed)
    model = BetaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    train = torch.from_numpy(train).to(device)
    validation = torch.from_numpy(validation).to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        # Equal-sized chunks avoid a singleton final batch in BatchNorm.
        n_batches = min(len(train) // 2, math.ceil(len(train) / batch_size))
        total = 0.0
        for batch in np.array_split(order, n_batches):
            x = train[torch.as_tensor(batch, device=device)]
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model.encode(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            objective = loss(x, model.decoder(z), mu, logvar)
            objective.backward()
            optimizer.step()
            total += objective.item() * len(x)
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for x in validation.split(batch_size):
                mu, logvar = model.encode(x)
                val_total += loss(x, model.decoder(mu), mu, logvar).item() * len(x)
        if not math.isfinite(total + val_total):
            raise ValueError(f"Training produced a non-finite loss at seed {seed}, beta {beta}, epoch {epoch}.")
        history.append((beta, seed, epoch, total / len(train), val_total / len(validation)))
    return model, history


def kl_per_sample(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1)


def project(x, model, torch, device, batch_size):
    """Encode frames with the posterior mean (no sampling) and score the reconstruction.

    Returns (latent, R2, mean KL, SSE, SST):
      latent : per-frame posterior mean, averaged over the two protomer directions
      R2     : 1 - SSE/SST on the [0, 1]-scaled features, decoded from the posterior mean;
               SST = total variance of the trajectory's own features (variance-weighted R2).
               For the reference trajectories this is in-sample (training + validation blocks);
               for T3-T9 it is out-of-sample (models are not retrained).
    """
    latent, kl = [], []
    sse = 0.0
    flat = x.reshape(-1, N_FEATURES)
    with torch.no_grad():
        for batch in torch.from_numpy(flat).split(batch_size):
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            latent.append(mu.cpu().numpy())
            kl.append(kl_per_sample(mu, logvar).cpu().numpy())
            sse += (model.decoder(mu) - batch).square().sum(dtype=torch.float64).item()
    latent = np.concatenate(latent).reshape(len(x), x.shape[1], 2).mean(axis=1)
    # Variance-weighted R² relative to each feature's mean in this dataset.
    sst = np.var(flat, axis=0, dtype=np.float64).sum() * len(flat)
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    return latent, r2, float(np.concatenate(kl).mean()), sse, sst


def attribution(x1, x2, model, direction, torch, device, batch_size):
    gradient = np.zeros(N_FEATURES, dtype=np.float64)
    w = torch.as_tensor(direction, dtype=torch.float32, device=device)
    count = 0
    for x in (x1, x2):
        for batch in torch.from_numpy(x.reshape(-1, N_FEATURES)).split(batch_size):
            batch = batch.to(device).requires_grad_(True)
            mu, _ = model.encode(batch)
            derivative = torch.autograd.grad((mu @ w).sum(), batch)[0]
            gradient += derivative.abs().sum(dim=0, dtype=torch.float64).cpu().numpy()
            count += len(batch)
    gradient /= count
    delta = x1.mean(axis=(0, 1), dtype=np.float64) - x2.mean(axis=(0, 1), dtype=np.float64)
    return gradient, delta, gradient * np.abs(delta)


def analyze(path, output, args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from sklearn import __version__ as sklearn_version
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config"]))
        mda_version = str(data["mdanalysis"])
        trajectories = {}
        direction_slice = {"both": slice(None), "A": slice(0, 1), "B": slice(1, 2)}[args.directions]
        for i, item in enumerate(config["trajectories"]):
            x = data[f"x{i}"]
            t = data[f"t{i}"]
            if x.shape != (len(t), 2, N_FEATURES) or len(t) < 2 or not np.allclose(np.diff(t), STEP_PS):
                raise ValueError(f"Invalid distance shape or time spacing for {item['id']}.")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
                raise ValueError(f"Non-finite distances or times for {item['id']}.")
            trajectories[item["id"]] = (x[:, direction_slice, :], t)
    roles = {item["role"]: item["id"] for item in config["trajectories"] if item["role"] != "test"}
    p1, p2 = roles["path1"], roles["path2"]
    refs = {key: trajectories[key] for key in (p1, p2)}
    betas = (PRIMARY_BETA,) if args.skip_beta_sweep else BETAS

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment. Use --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    for key in ("features", "output"):
        settings[key] = str(settings[key].resolve())
    settings.update(seeds=SEEDS, betas=betas, numpy=np.__version__, torch=str(torch.__version__),
                    sklearn=sklearn_version, mdanalysis=mda_version, extraction=config)
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    metrics, histories, projections, votes, rankings, scores, reference_diagnostics, clipping = ([] for _ in range(8))
    blocks_by_seed = {}
    split_rows = []
    recorded_splits = config.get("validation_blocks")
    if recorded_splits is not None and set(recorded_splits) != {str(seed) for seed in SEEDS}:
        raise ValueError("validation_blocks must provide a split for each of the five seeds.")
    group_names = np.array([group for group, _, _ in FEATURES])
    labels = np.r_[np.ones(len(refs[p1][0]), dtype=int), np.zeros(len(refs[p2][0]), dtype=int)]
    with threadpool_limits(limits=args.threads):
        for seed in SEEDS:
            recorded = None if recorded_splits is None else recorded_splits[str(seed)]
            splits = reference_split(refs, seed, args.validation_fraction, recorded)
            blocks_by_seed[seed] = {key: blocks for key, (_, blocks) in splits.items()}
            for key, (_, times) in refs.items():
                valid_mask = splits[key][0]
                for index, first in enumerate(range(0, len(times), BLOCK_FRAMES)):
                    split_rows.append((seed, key, index, "validation" if valid_mask[first] else "train",
                                       times[first], times[first + BLOCK_FRAMES - 1], BLOCK_FRAMES))
            for key, (_, times) in trajectories.items():
                if key not in refs:
                    split_rows.append((seed, key, "", "test", times[0], times[-1], len(times)))
            train_raw = np.concatenate([x[~splits[key][0]] for key, (x, _) in refs.items()])
            axis = (0, 1) if args.scaling == "feature" else None
            lo = np.broadcast_to(train_raw.min(axis=axis), (N_FEATURES,)).copy()
            span = train_raw.max(axis=axis) - lo
            del train_raw
            scaled = {}
            for name, (x, _) in trajectories.items():
                y = (x - lo) / np.where(span > 0, span, 1)
                y[..., span == 0] = 0
                clipping.append((seed, name, np.count_nonzero((y < 0) | (y > 1)) / y.size))
                if args.clip:
                    np.clip(y, 0, 1, out=y)
                scaled[name] = y
            train = np.concatenate([scaled[key][~splits[key][0]] for key in refs]).reshape(-1, N_FEATURES)
            valid = np.concatenate([scaled[key][splits[key][0]] for key in refs]).reshape(-1, N_FEATURES)

            for beta in betas:
                model, history = train_model(
                    train, valid, torch, device, seed, beta, args.epochs, args.batch_size, args.learning_rate
                )
                histories.extend(history)
                ref_results = {key: project(scaled[key], model, torch, device, args.batch_size) for key in refs}
                ref_latent = np.concatenate([ref_results[key][0] for key in refs])
                sil = silhouette_score(ref_latent, labels)
                kl = sum(ref_results[key][2] * len(refs[key][0]) for key in refs) / len(labels)
                pooled = np.concatenate([scaled[key] for key in refs]).reshape(-1, N_FEATURES)
                sst = np.var(pooled, axis=0, dtype=np.float64).sum() * len(pooled)
                del pooled
                r2 = 1 - sum(ref_results[key][3] for key in refs) / sst if sst > 0 else float("nan")
                reference_diagnostics.append((beta, seed, r2, kl, sil))
                print(f"Seed {seed}, beta {beta}: reference R2={r2:.4f}, KL={kl:.4f}, silhouette={sil:.4f}",
                      flush=True)
                if beta != PRIMARY_BETA:  # only the primary beta feeds LDA, voting and attribution
                    continue

                lda = LinearDiscriminantAnalysis().fit(ref_latent, labels)
                norm = np.linalg.norm(lda.coef_[0])
                if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                    raise ValueError(f"No usable LDA direction for seed {seed}. Check reference distances.")
                w, intercept = lda.coef_[0] / norm, float(lda.intercept_[0] / norm)
                torch.save({
                    "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                    "min": torch.from_numpy(lo), "range": torch.from_numpy(span),
                    "lda_direction": torch.from_numpy(w), "lda_intercept": intercept,
                    "directions": args.directions, "seed": seed, "beta": beta,
                }, output / f"seed_{seed}.pt")
                for name, (_, times) in trajectories.items():
                    if name in refs:
                        z, r2, kl, _, _ = ref_results[name]
                    else:
                        z, r2, kl, _, _ = project(scaled[name], model, torch, device, args.batch_size)
                    distance = z @ w + intercept
                    assigned = distance > 0
                    count = int(assigned.sum())
                    fraction = count / len(times)
                    votes.append((name, seed, len(times), count, fraction, int(fraction >= PATH1_FRACTION_THRESHOLD)))
                    metrics.append((name, seed, r2, kl))
                    projections.extend(
                        (name, seed, time, point[0], point[1], d, int(label))
                        for time, point, d, label in zip(times, z, distance, assigned)
                    )

                g, delta, score = attribution(scaled[p1], scaled[p2], model, w, torch, device, args.batch_size)
                scores.extend((seed, group, i, j, g[k], delta[k], score[k])
                              for k, (group, i, j) in enumerate(FEATURES))
                means = {group: float(score[group_names == group].mean()) for group in GROUPS}
                ranked = sorted(GROUPS, key=means.__getitem__, reverse=True)
                rankings.extend((seed, group, rank, means[group]) for rank, group in enumerate(ranked, 1))

    def summarize(name):
        rows = [row for row in votes if row[0] == name]
        fractions = [row[4] for row in rows]
        n_path1 = sum(row[5] for row in rows)
        assignment = (
            "Path 1 reference" if name == p1 else
            "Path 2 reference" if name == p2 else
            "Path 1-like" if n_path1 >= MIN_SEED_VOTES else "Path 2-like"
        )
        return name, np.mean(fractions), np.std(fractions, ddof=1), n_path1, assignment

    write_csv(output / "assignments.csv", ("trajectory", "mean_fraction", "sd_fraction", "path1_votes", "assignment"),
              (summarize(name) for name in trajectories))
    write_csv(output / "seed_votes.csv", ("trajectory", "seed", "n_frames", "n_path1", "path1_fraction", "path1_vote"),
              votes)
    write_csv(output / "latent.csv", ("trajectory", "seed", "time_ps", "mu1", "mu2", "lda_margin", "path1_frame"),
              projections)
    write_csv(output / "reconstruction.csv", ("trajectory", "seed", "r2", "mean_kl"), metrics)
    write_csv(output / "reconstruction_summary.csv", ("trajectory", "r2_mean", "r2_sample_sd"),
              ((name, np.mean(values), np.std(values, ddof=1))
               for name in trajectories
               for values in [[row[2] for row in metrics if row[0] == name]]))
    write_csv(output / "beta_diagnostics.csv", ("beta", "seed", "reference_r2", "mean_kl", "silhouette"),
              reference_diagnostics)
    write_csv(output / "training.csv", ("beta", "seed", "epoch", "train_loss", "validation_loss_at_mu"), histories)
    write_csv(output / "clipping.csv", ("seed", "trajectory", "fraction_outside_unit_interval"), clipping)
    write_csv(output / "dataset_split.csv",
              ("seed", "trajectory", "block", "split", "first_ps", "last_ps", "frames"), split_rows)
    write_csv(output / "attribution.csv", ("seed", "helix_pair", "s1_resid", "target_resid", "g_i", "delta_i", "s_i"),
              scores)
    write_csv(output / "helix_ranking.csv", ("seed", "helix_pair", "rank", "mean_score"), rankings)
    (output / "validation_blocks.json").write_text(json.dumps(blocks_by_seed, indent=2), encoding="utf-8")
    plot_results(output)


def plot_results(output, diagnostic_seed=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rows(name):
        with (output / name).open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def save(fig, name):
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    config = json.loads((output / "settings.json").read_text(encoding="utf-8"))["extraction"]
    ids = {item["role"]: item["id"] for item in config["trajectories"] if item["role"] != "test"}
    ref_ids = (ids["path1"], ids["path2"])
    tests = TEST_IDS
    order = (*ref_ids, *tests)
    colors = dict(zip(order, ("#2d8496", "#cc6656", "#0560bd", "#66249b", "#007d67",
                              "#b97900", "#ce4965", "#765140", "#485763")))
    labels = dict(zip(order, ("Path 1 ref.", "Path 2 ref.", *tests)))
    points = {name: [] for name in order}
    for row in rows("latent.csv"):
        if int(row["seed"]) == SEEDS[0]:
            points[row["trajectory"]].append((float(row["mu1"]), float(row["mu2"])))
    points = {name: np.array(values) for name, values in points.items()}
    figure, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    a, b, c, d = axs.flat
    for name in ref_ids:
        x, y = points[name].T
        a.scatter(x, y, s=4, alpha=0.55, c=colors[name], label=labels[name], rasterized=True)
        b.scatter(x, y, s=4, alpha=0.14, c=colors[name], label=labels[name], rasterized=True)
    for name in tests:
        x, y = points[name].T
        b.scatter(x, y, s=5, alpha=0.55, c=colors[name], label=name, rasterized=True)
        b.scatter(x.mean(), y.mean(), marker="X", s=36, c=colors[name], edgecolor="black", linewidth=0.6)
    for ax in (a, b):
        ax.set(xlabel="Latent posterior mean 1", ylabel="Latent posterior mean 2")
        ax.set_xlim(b.get_xlim())
        ax.set_ylim(b.get_ylim())
        ax.legend(loc="best", fontsize=8, ncol=3)

    helix_order = ("S1'", "S2", "S2'", "S3", "S3'", "S4'", "S4")
    scores = {(row["helix_pair"], int(row["s1_resid"]), int(row["target_resid"])): float(row["s_i"])
              for row in rows("attribution.csv") if int(row["seed"]) == SEEDS[0]}
    targets = [(group, j) for group in helix_order
               for j in range(HELICES[group.rstrip("'")][0], HELICES[group.rstrip("'")][1] + 1)]
    s1_residues = range(S1_RANGE[0], S1_RANGE[1] + 1)
    heat = np.array([[scores[group, i, j] for group, j in targets] for i in s1_residues])
    bounds = np.r_[0, np.cumsum([sum(g == group for g, _ in targets) for group in helix_order])]
    image = c.imshow(heat, aspect="auto", origin="lower", cmap="viridis", vmin=0)
    for boundary in bounds[1:-1]:
        c.axvline(boundary - 0.5, color="white", linewidth=0.7)
    c.set_xticks((bounds[:-1] + bounds[1:] - 1) / 2, helix_order)
    c.set_yticks(np.arange(0, len(s1_residues), 3), np.arange(S1_RANGE[0], S1_RANGE[1] + 1, 3))
    c.set(xlabel="Paired helix group", ylabel="S1 residue number")
    figure.colorbar(image, ax=c, label="Selection score")

    assignments = {row["trajectory"]: row for row in rows("assignments.csv")}
    for i, name in enumerate(order):
        row = assignments[name]
        d.errorbar(i, float(row["mean_fraction"]), yerr=float(row["sd_fraction"]),
                   fmt="o", color=colors[name], capsize=3)
    d.axhline(PATH1_FRACTION_THRESHOLD, color="#444444", linestyle="--", linewidth=0.8, label="Per-seed threshold")
    d.set_xticks(range(len(order)), [labels[name] for name in order], rotation=25, ha="right")
    d.set(xlabel="Trajectory", ylabel="Fraction of Path 1-like frames")
    d.legend(fontsize=8)
    for letter, ax in zip("ABCD", axs.flat):
        ax.set_title(f"({letter})", loc="left")
    save(figure, "Figure_4")

    diagnostics = rows("beta_diagnostics.csv")
    beta_values = sorted({float(row["beta"]) for row in diagnostics})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    a, b = axs
    reconstruction = {row["trajectory"]: row for row in rows("reconstruction_summary.csv")}
    for i, name in enumerate(order):
        row = reconstruction[name]
        a.errorbar(i, float(row["r2_mean"]), yerr=float(row["r2_sample_sd"]),
                   fmt="o", color=colors[name], capsize=3)
    a.set_xticks(range(len(order)), [labels[name] for name in order], rotation=25, ha="right")
    a.set(xlabel="Trajectory", ylabel="Normalized distance reconstruction $R^2$")
    b.axis("off")
    columns = ("silhouette", "mean_kl", "reference_r2")
    table_data = [[f"{np.mean([float(row[key]) for row in diagnostics if float(row['beta']) == beta and (diagnostic_seed is None or int(row['seed']) == diagnostic_seed)]):.3f}"
                   for beta in beta_values] for key in columns]
    b.table(cellText=table_data, rowLabels=["Silhouette", "Mean KL", "Reconstruction $R^2$"],
            colLabels=[f"β = {beta:g}" for beta in beta_values], cellLoc="center",
            bbox=(0.31, 0.25, 0.67, 0.5))
    b.text(0.31, 0.85, "Mean across seeds" if diagnostic_seed is None else f"Seed {diagnostic_seed}",
           transform=b.transAxes)
    for letter, ax in zip("AB", axs):
        ax.set_title(f"({letter})", loc="left")
    save(fig, "Figure_S2")

    rank = {(row["helix_pair"], int(row["seed"])): int(row["rank"])
            for row in rows("helix_ranking.csv")}
    ranks = [[rank[group, seed] for seed in SEEDS] for group in helix_order]
    fig, ax = plt.subplots(figsize=(6, 4.6), layout="constrained")
    image = ax.imshow(ranks, aspect="auto", cmap="YlGnBu_r", vmin=1, vmax=7)
    for i, values in enumerate(ranks):
        for j, value in enumerate(values):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value <= 3 else "black")
    ax.set_xticks(range(len(SEEDS)), SEEDS)
    ax.set_yticks(range(len(helix_order)), [f"S1–{group}" for group in helix_order])
    ax.set(xlabel="Model-training random seed", ylabel="Helix pair")
    fig.colorbar(image, ax=ax, label="Rank")
    save(fig, "Figure_S3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    extract_parser = subparsers.add_parser("extract", help="Extract 80-ps C-alpha distance features using MDAnalysis")
    extract_parser.add_argument("manifest", type=Path)
    extract_parser.add_argument("output", type=Path)
    analyze_parser = subparsers.add_parser("analyze", help="Train beta-VAE models and analyze the frozen projections")
    analyze_parser.add_argument("features", type=Path)
    analyze_parser.add_argument("output", type=Path)
    # ---- adjustable training options ----
    analyze_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                                help="compute device (results are deterministic per device, not across devices)")
    analyze_parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    analyze_parser.add_argument("--epochs", type=int, default=300, help="training epochs [manuscript: 300]")
    analyze_parser.add_argument("--batch-size", type=int, default=128, help="mini-batch size")
    analyze_parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate")
    analyze_parser.add_argument("--validation-fraction", type=float, default=0.2,
                                help="fraction of 20-ns reference blocks held out for validation (random per seed)")
    analyze_parser.add_argument("--directions", choices=("both", "A", "B"), default="both",
                                help="both = pool the two protomer directions as separate samples and average "
                                     "their latent means per frame (implementation default); A or B = a single direction")
    analyze_parser.add_argument("--scaling", choices=("feature", "global"), default="feature",
                                help="min-max scaling per feature or with one global min/max; "
                                     "min/max are taken from training blocks only")
    analyze_parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True,
                                help="clip scaled validation/test values to [0, 1]; the fraction outside [0, 1] "
                                     "is written to clipping.csv")
    analyze_parser.add_argument("--skip-beta-sweep", action="store_true", help="Train only the primary beta=0.1 models")
    plot_parser = subparsers.add_parser("plot", help="Redraw figures from existing result CSVs")
    plot_parser.add_argument("output", type=Path)
    plot_parser.add_argument("--diagnostic-seed", type=int, choices=SEEDS,
                             help="Show one seed instead of the five-seed mean in Figure S2B")
    args = parser.parse_args()
    try:
        if args.stage == "extract":
            extract(args.manifest, args.output)
        elif args.stage == "analyze":
            if not (args.epochs > 0 and args.threads > 0 and args.batch_size >= 2
                    and 0 < args.validation_fraction < 1 and math.isfinite(args.learning_rate)
                    and args.learning_rate > 0):
                raise ValueError("Invalid training parameters. See analyze --help.")
            analyze(args.features, args.output, args)
        else:
            plot_results(args.output, args.diagnostic_seed)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

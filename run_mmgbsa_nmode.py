#!/usr/bin/env python3
"""Sample and score one hHv1 replica with AmberTools, or summarize replicas.

Sub-commands
  run        score ONE independent trajectory (cpptraj sampling -> MMPBSA.py GB [-> NMODE] -> MSMS SASA)
  summarize  mean +/- sample SD across replicas (Table 1 / S6 / S7); optional Welch t-test between two models
  blocks     block-wise Delta-H over consecutive sub-windows of the 400-500 ns samples (Table S3)

"""

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
from pathlib import Path


# ======================= USER-ADJUSTABLE PARAMETERS =======================
WINDOW_NS = (400.0, 500.0)           # analysis window of every trajectory (ns)            [manuscript]
AQUEOUS_SNAPSHOTS = 100              # snapshots per aqueous replica (also used for NMODE) [manuscript]
REPLICAS = {"aqueous": 3, "bilayer": 2}   # independent trajectories required per model   [manuscript]
RESIDUES_PER_PROTOMER = 187          # SHM tag (3) + canonical 83-266 (184); override with
                                     # "receptor_residues" / "ligand_residues" in the JSON  [manuscript]
BLOCK_NS = 20                        # duration of each Table S3 block (ns)               [manuscript]
WELCH_CONFIDENCE = 0.95              # confidence level of the Welch interval              [manuscript]
# ===========================================================================
COMPONENTS = ("vdw", "electrostatic", "gb_polar", "nonpolar", "delta_h", "minus_t_delta_s", "delta_g_bind")
AMBER_TERMS = dict(VDWAALS="vdw", EEL="electrostatic", EGB="gb_polar", ESURF="nonpolar")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def write_csv(path, names, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def resolve(base, value):
    return (base / value).resolve()


def sampling(data, mode):
    first, last, stride = (integer(data[key], key) for key in ("first_frame", "last_frame", "stride"))
    if first > last or (last - first) % stride:
        raise ValueError("last_frame must be one of the frames selected by first_frame and stride.")
    frames = list(range(first, last + 1, stride))
    dt = finite(data["frame_spacing_ns"], "frame_spacing_ns")
    start = finite(data["first_saved_time_ns"], "first_saved_time_ns")
    if dt <= 0:
        raise ValueError("frame_spacing_ns must be positive.")
    times = [start + (frame - 1) * dt for frame in frames]
    if times[0] < WINDOW_NS[0] - 1e-6 or times[-1] > WINDOW_NS[1] + 1e-6:
        raise ValueError(f"Selected frames span {times[0]:g}–{times[-1]:g} ns; use 400–500 ns.")
    if mode == "aqueous":
        if len(frames) != AQUEOUS_SNAPSHOTS:
            raise ValueError(f"Each aqueous replica requires exactly {AQUEOUS_SNAPSHOTS} snapshots.")
        gap = stride * dt
        if times[0] > WINDOW_NS[0] + gap + 1e-6 or times[-1] < WINDOW_NS[1] - gap - 1e-6:
            raise ValueError("The aqueous samples must cover 400–500 ns to within one sampling interval.")
    return frames, times, stride


def settings(path):
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    mode = cfg["mode"]
    if mode not in {"aqueous", "bilayer"}:
        raise ValueError("mode must be aqueous or bilayer.")
    for key in ("model", "replica"):
        if not str(cfg[key]).strip():
            raise ValueError(f"{key} must identify this run.")
    frames, times, stride = sampling(cfg["sampling"], mode)
    mask = cfg.get("strip_mask")
    if not isinstance(mask, str) or any(ch.isspace() for ch in mask):
        raise ValueError("Set strip_mask explicitly as an Amber mask, or an empty string for dry input.")
    surface = cfg["surface"]
    if surface["method"] not in {"msms", "lcpo", "molsurf"}:
        raise ValueError("surface.method must be msms, lcpo, or molsurf.")
    for key in ("gamma", "probe_angstrom"):
        if finite(surface[key], key) <= 0:
            raise ValueError(f"surface.{key} must be positive.")
    if surface["method"] == "msms" and not surface.get("radii_file"):
        raise ValueError("For MSMS, set surface.radii_file to the atom radii used for SASA.")
    ranks = integer(cfg["mpi_ranks"], "mpi_ranks")
    if ranks > len(frames):
        raise ValueError("mpi_ranks cannot exceed the number of sampled frames.")
    return cfg, frames, times, stride


def amber_input(cfg, count):
    gb = cfg["gb"]
    igb = integer(gb["igb"], "igb")
    salt = finite(gb["saltcon_molar"], "saltcon_molar")
    if igb not in (1, 2, 5, 7, 8) or salt < 0:
        raise ValueError("Amber20 MMPBSA.py supports igb=1,2,5,7,8 and nonnegative saltcon.")
    overrides = {key: finite(gb[key], key) for key in ("intdiel", "extdiel") if gb.get(key) is not None}
    if any(value <= 0 for value in overrides.values()):
        raise ValueError("Dielectric constants must be positive.")
    surface = cfg["surface"]
    method = surface["method"]
    gamma = 0.0 if method == "msms" else float(surface["gamma"])
    keep = integer(cfg.get("keep_files", 1), "keep_files", 0)
    if keep > 2:
        raise ValueError("keep_files must be 0, 1, or 2.")
    if cfg.get("amber_entropy", 0) != 0:
        raise ValueError("Set amber_entropy to 0 or remove it. Amber20 enables NMODE via &nmode.")
    lines = [
        "&general",
        f"  startframe=1, endframe={count}, interval=1,",
        f"  use_sander={int(bool(overrides) or gb.get('use_sander', False))},",
        f"  keep_files={keep}, netcdf={int(bool(cfg.get('netcdf_intermediates', False)))},",
        "  verbose=1, entropy=0,",
        "/",
        "&gb",
        f"  igb={igb}, saltcon={salt:g},",
        f"  surften={gamma:g}, surfoff=0.0, molsurf={int(method == 'molsurf')},",
    ]
    if method == "molsurf":
        lines.append(f"  probe={float(surface['probe_angstrom']):g},")
    lines.append("/")
    if cfg["mode"] == "aqueous":
        nm = cfg["nmode"]
        model = integer(nm["igb"], "nmode.igb", 0)
        drms = finite(nm["drms"], "nmode.drms")
        salt = finite(nm["saltcon_molar"], "nmode.saltcon_molar")
        dielc = finite(nm["dielc"], "nmode.dielc")
        if model not in (0, 1) or drms <= 0 or salt < 0 or dielc <= 0:
            raise ValueError("Check nmode.igb (0 or 1), drms, saltcon_molar and dielc.")
        lines.extend([
            "&nmode",
            f"  nmstartframe=1, nmendframe={count}, nminterval=1,",
            f"  nmode_igb={model}, nmode_istrng={salt:g},",
            f"  drms={drms:g}, maxcyc={integer(nm['maxcyc'], 'nmode.maxcyc')}, dielc={dielc:g},",
            "/",
        ])
    return "\n".join(lines) + "\n", overrides


def topology_counts(path):
    match = re.search(r"(?m)^%FLAG\s+POINTERS[^\n]*\n%FORMAT[^\n]*\n([^%]+)", path.read_text(encoding="ascii"))
    if match is None:
        raise ValueError(f"Missing Amber POINTERS in {path}")
    pointers = [int(x) for x in match[1].split()]
    return pointers[0], pointers[11]  # NATOM and NRES


def program(value, base):
    name = str(resolve(base, value)) if "/" in value or "\\" in value else value
    found = shutil.which(name)
    if found is None:
        raise ValueError(f"Executable not found: {name}. Load the AmberTools/MSMS environment.")
    return found


def call(command, output, log):
    with (output / log).open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(map(str, command)) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=output, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise ValueError(f"{Path(command[0]).name} exited with {proc.returncode}. See {output / log}")


def patch_dielectrics(path, values):
    text = path.read_text(encoding="ascii")
    head = re.search(r"&cntrl\b", text, re.IGNORECASE)
    if head is None:
        raise ValueError(f"Missing &cntrl in {path}")
    body = text[head.end():]
    for key, value in values.items():
        body, n = re.subn(rf"\b{key}\s*=\s*{NUMBER}", f"{key}={value:g}", body, count=1, flags=re.IGNORECASE)
        if not n:
            body = f"\n  {key}={value:g}," + body
    path.write_text(text[:head.end()] + body, encoding="ascii")


def amber_value(text):
    return finite(text.replace("D", "E").replace("d", "e"), "Amber result")


def read_amber_summary(path, nmode):
    text = path.read_text(encoding="utf-8", errors="replace")
    gb = re.search(r"(?im)^\s*GENERALIZED BORN[^\n]*:\s*$", text)
    if gb is None:
        raise ValueError(f"Missing GENERALIZED BORN results in {path}")
    delta_header = r"(?im)^\s*(?:Differences|Delta)\s*\(\s*Complex\s*-\s*Receptor\s*-\s*Ligand\s*\)\s*:"
    delta = re.search(delta_header, text[gb.end():])
    if delta is None:
        raise ValueError(f"Missing GB binding differences in {path}")
    gb_delta = text[gb.end() + delta.end():]
    values = {}
    for term, key in AMBER_TERMS.items():
        item = re.search(rf"(?im)^\s*(?:DELTA\s+)?{term}\s+({NUMBER})(?=\s|$)", gb_delta)
        if item is None:
            raise ValueError(f"Missing GB difference term {term} in {path}")
        values[key] = amber_value(item[1])
    total = re.search(rf"(?im)^\s*DELTA\s+(?:TOTAL|G binding)\s*=?\s*({NUMBER})(?=\s|$)", gb_delta)
    if total is None or not math.isclose(sum(values.values()), amber_value(total[1]), abs_tol=0.002):
        raise ValueError("Amber's GB total does not match the four binding components; check the report and topologies.")
    if nmode:
        section = re.search(r"(?im)^.*ENTROPY RESULTS[^\n]*(?:NMODE|NORMAL MODE)[^\n]*$", text)
        if section is None:
            raise ValueError(f"Missing NMODE results in {path}; check &nmode, the AmberTools installation, and mmpbsa.log.")
        # Amber's summary reports T*deltaS in energy units.
        entropy = text[section.end():]
        ts = re.search(rf"(?im)^\s*DELTA\s+S\s+total\s*=\s*({NUMBER})(?=\s|$)", entropy)
        if ts is None:
            delta = re.search(delta_header, entropy)
            if delta is not None:
                ts = re.search(rf"(?im)^\s*(?:DELTA\s+)?TOTAL\s*:?\s+({NUMBER})(?=\s|$)",
                               entropy[delta.end():])
        if ts is None:
            raise ValueError(f"Missing NMODE binding entropy total in {path}; check for unconverged frames.")
        values["minus_t_delta_s"] = -amber_value(ts[1])
    return values


def msms_sasa(executable, xyz, radii, probe, folder, label, frame):
    # Zero radii in an explicit radii file omit the corresponding atoms.
    active = radii > 0
    xyzr = folder / f"{label}.xyzr"
    area = folder / f"{label}.area"
    if area.exists():
        area.unlink()
    with xyzr.open("w", encoding="ascii") as handle:
        for (x, y, z), radius in zip(xyz[active], radii[active]):
            handle.write(f"{x:.8f} {y:.8f} {z:.8f} {radius:.8f}\n")
    call([executable, "-if", xyzr.name, "-af", area.name, "-probe_radius", f"{probe:g}"],
         folder, f"{frame:04d}_{label}.log")
    rows = area.read_text(encoding="ascii").splitlines()
    if not rows or not re.search(r"\bSAS", rows[0], re.IGNORECASE):
        raise ValueError(f"Unexpected MSMS area header in {area}")
    areas = [line.split() for line in rows[1:] if line.strip()]
    indices = [int(row[0]) for row in areas]
    if indices not in (list(range(int(active.sum()))), list(range(1, int(active.sum()) + 1))):
        raise ValueError(f"Incomplete MSMS per-atom SAS areas for frame {frame}, {label}.")
    sas = [finite(row[2], "MSMS SASA") for row in areas]
    if any(value < 0 for value in sas):
        raise ValueError(f"Negative MSMS per-atom SAS area for frame {frame}, {label}.")
    return math.fsum(sas)


def run(cfg_path, output, prepare_only):
    cfg, frames, times, stride = settings(cfg_path)
    base = cfg_path.resolve().parent
    output = output.resolve()
    files = {key: resolve(base, cfg[key]) for key in ("trajectory", "trajectory_topology",
                                                      "complex_topology", "receptor_topology", "ligand_topology")}
    amber, overrides = amber_input(cfg, len(frames))
    a_res = integer(cfg.get("receptor_residues", RESIDUES_PER_PROTOMER), "receptor_residues")
    b_res = integer(cfg.get("ligand_residues", RESIDUES_PER_PROTOMER), "ligand_residues")
    amber = amber.replace("&general\n", f'&general\n  receptor_mask=":1-{a_res}", ligand_mask=":{a_res + 1}-{a_res + b_res}",\n', 1)
    trajectory = files["trajectory"].as_posix()
    if any(ch in trajectory for ch in ('"', "#", "\n", "\r")):
        raise ValueError("The cpptraj trajectory path cannot contain quotes, #, or newlines.")
    sample = [f'trajin "{trajectory}" {frames[0]} {frames[-1]} {stride}']
    if cfg.get("autoimage", False):
        sample.append("autoimage")
    if cfg["strip_mask"]:
        sample.append(f"strip {cfg['strip_mask']}")
    sample.extend(("trajout sampled.nc netcdf nobox", "run"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "cpptraj.in").write_text("\n".join(sample) + "\n", encoding="utf-8")
    (output / "mmpbsa.in").write_text(amber, encoding="ascii")
    record = dict(cfg)
    record.update({key: str(value) for key, value in files.items()})
    (output / "settings.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_csv(output / "sampled_frames.csv", ("sample", "trajectory_frame", "time_ns"),
              ({"sample": i, "trajectory_frame": frame, "time_ns": f"{time:.9g}"}
               for i, (frame, time) in enumerate(zip(frames, times), 1)))
    if prepare_only:
        print(f"Prepared inputs for {len(frames)} frames in {output}")
        return

    import numpy as np
    from scipy.io import netcdf_file

    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    c_atoms, c_res = topology_counts(files["complex_topology"])
    a_atoms, observed_a_res = topology_counts(files["receptor_topology"])
    b_atoms, observed_b_res = topology_counts(files["ligand_topology"])
    if (c_atoms != a_atoms + b_atoms or (observed_a_res, observed_b_res) != (a_res, b_res)
            or c_res != a_res + b_res):
        raise ValueError("Dry topology counts do not match the configured protomer partition.")
    exe = cfg["programs"]
    surface = cfg["surface"]
    if surface["method"] == "msms":
        msms = program(exe["msms"], base)
        radii = np.loadtxt(resolve(base, surface["radii_file"]), ndmin=2)
        if radii.shape != (c_atoms, 1) or not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("MSMS radii file must contain one nonnegative radius per dry complex atom.")
        radii = radii[:, 0]
        if not ((radii[:a_atoms] > 0).any() and (radii[a_atoms:] > 0).any()):
            raise ValueError("Both protomers must have atoms with positive MSMS radii.")
    call([program(exe["cpptraj"], base), "-p", str(files["trajectory_topology"]), "-i", "cpptraj.in"],
         output, "cpptraj.log")
    with netcdf_file(str(output / "sampled.nc"), "r", mmap=False) as nc:
        coords = nc.variables["coordinates"]
        if coords.shape != (len(frames), c_atoms, 3):
            raise ValueError(f"Sampled trajectory has shape {coords.shape}; check the frame range and strip mask.")
        xyz = coords[:].copy() if surface["method"] == "msms" else None
    if xyz is not None and not np.isfinite(xyz).all():
        raise ValueError("Sampled trajectory contains nonfinite coordinates.")

    if cfg["mpi_ranks"] > 1:
        cmd = [program(exe["mpi_launcher"], base), "-np", str(cfg["mpi_ranks"]),
               program(exe["mmpbsa_mpi"], base)]
    else:
        cmd = [program(exe["mmpbsa"], base)]
    cmd += ["-O", "-i", "mmpbsa.in", "-o", "FINAL_RESULTS_MMPBSA.dat", "-eo", "amber_per_frame.csv",
            "-cp", str(files["complex_topology"]), "-rp", str(files["receptor_topology"]),
            "-lp", str(files["ligand_topology"]), "-y", "sampled.nc"]
    if overrides:
        call(cmd + ["-make-mdins"], output, "make_mdins.log")
        patch_dielectrics(output / "_MMPBSA_gb.mdin", overrides)
        shutil.copyfile(output / "_MMPBSA_gb.mdin", output / "effective_gb.mdin")
        cmd.append("-use-mdins")
    call(cmd, output, "mmpbsa.log")
    values = read_amber_summary(output / "FINAL_RESULTS_MMPBSA.dat", cfg["mode"] == "aqueous")

    if xyz is not None:
        folder = output / "msms"
        folder.mkdir()
        areas = []
        for i, frame in enumerate(xyz, 1):
            probe = float(surface["probe_angstrom"])
            complex_area = msms_sasa(msms, frame, radii, probe, folder, "complex", i)
            receptor_area = msms_sasa(msms, frame[:a_atoms], radii[:a_atoms], probe, folder, "receptor", i)
            ligand_area = msms_sasa(msms, frame[a_atoms:], radii[a_atoms:], probe, folder, "ligand", i)
            delta_area = complex_area - receptor_area - ligand_area
            areas.append({"sample": i, "time_ns": times[i - 1], "sasa_complex": complex_area,
                          "sasa_receptor": receptor_area, "sasa_ligand": ligand_area,
                          "delta_sasa": delta_area, "nonpolar": float(surface["gamma"]) * delta_area})
        write_csv(output / "msms_sasa.csv", tuple(areas[0]), areas)
        values["nonpolar"] = statistics.mean(row["nonpolar"] for row in areas)

    values["delta_h"] = sum(values[key] for key in COMPONENTS[:4])
    if cfg["mode"] == "aqueous":
        values["delta_g_bind"] = values["delta_h"] + values["minus_t_delta_s"]
    metadata = {"model": cfg["model"], "replica": str(cfg["replica"]), "mode": cfg["mode"],
                "surface_method": surface["method"], "snapshots": len(frames)}
    row = {**metadata, **values}
    write_csv(output / "replica_summary.csv", (*metadata, *COMPONENTS), [row])
    print(f"Replica results: {output / 'replica_summary.csv'}")


def read_replica_rows(paths):
    groups, seen = {}, set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            (row,) = tuple(csv.DictReader(handle))
        key = (row["mode"], row["model"], row["surface_method"])
        identity = (row["mode"], row["model"], row["replica"])
        if identity in seen:
            raise ValueError(f"Duplicate replica: {identity}")
        seen.add(identity)
        groups.setdefault(key, []).append(row)
    return groups


def welch(a, b):
    """Two-sided Welch t-test of mean(a) - mean(b) with a WELCH_CONFIDENCE interval."""
    from scipy import stats

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Welch's test needs at least two values per group.")
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    diff = statistics.mean(a) - statistics.mean(b)
    se = math.sqrt(va + vb)
    if se == 0:  # identical values in every replica: the test is undefined
        return diff, math.nan, math.nan, math.nan, math.nan, math.nan
    t_stat = diff / se
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    p_value = 2 * stats.t.sf(abs(t_stat), dof)
    half = stats.t.ppf(0.5 + WELCH_CONFIDENCE / 2, dof) * se
    return diff, t_stat, dof, p_value, diff - half, diff + half


def summarize(paths, output, compares=()):
    groups = read_replica_rows(paths)
    summary = []
    for (mode, model, method), rows in groups.items():
        expected = REPLICAS[mode]
        if len(rows) != expected:
            raise ValueError(f"{mode}/{model}/{method} requires {expected} independent replica summaries.")
        for term in COMPONENTS if mode == "aqueous" else COMPONENTS[:5]:
            values = [finite(row[term], term) for row in rows]
            summary.append(dict(mode=mode, model=model, surface_method=method, n=expected,
                                term=term, mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    welch_rows = []
    for first, second in compares:
        if first == second:
            raise ValueError("--compare requires two distinct aqueous models.")
        found = False
        for (mode, model, method), rows in groups.items():
            other = groups.get((mode, second, method))
            if mode != "aqueous" or model != first or other is None:
                continue
            found = True
            term = "delta_g_bind"
            a = [finite(row[term], term) for row in rows]
            b = [finite(row[term], term) for row in other]
            diff, t_stat, dof, p_value, low, high = welch(a, b)
            welch_rows.append(dict(mode=mode, surface_method=method, first=first, second=second, term=term,
                                   n_first=len(a), n_second=len(b), mean_difference=diff, t=t_stat,
                                   dof=dof, p_two_sided=p_value, ci_low=low, ci_high=high))
        if not found:
            raise ValueError(f"No matching aqueous replicas for --compare {first} {second}.")
    welch_path = output.with_name(output.stem + "_welch.csv")
    if output.exists():
        raise FileExistsError(output)
    if welch_rows and welch_path.exists():
        raise FileExistsError(welch_path)
    write_csv(output, ("mode", "model", "surface_method", "n", "term", "mean", "sample_sd"), summary)
    print(f"Across-replica results: {output}")
    if welch_rows:
        write_csv(welch_path, tuple(welch_rows[0]), welch_rows)
        print(f"Welch comparisons (mean_difference = first - second): {welch_path}")


def read_per_frame_delta(path):
    """Per-frame GB binding terms (VDWAALS, EEL, EGB, ESURF) from MMPBSA.py's -eo CSV.

    Looks for the first 'Delta'/'Differences' block after the GENERALIZED BORN heading. The exact
    layout of that file differs a little between AmberTools releases, so the parser fails loudly
    instead of guessing; check the layout of amber_per_frame.csv if it does.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        lines = [[cell.strip() for cell in row] for row in csv.reader(handle) if any(row)]
    begin = next((i for i, row in enumerate(lines) if re.search(r"generalized born", row[0], re.I)), None)
    if begin is None:
        raise ValueError(f"No GENERALIZED BORN section in {path}")
    i, header = begin + 1, None
    while i < len(lines):
        text = lines[i][0]
        if re.match(r"(?i)^(delta|differences?)\b", text) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt[0].strip().lower().startswith("frame"):
                header, i = [cell.strip().upper() for cell in nxt], i + 2
                break
        i += 1
    if header is None:
        raise ValueError(f"No Delta/Differences table with a 'Frame #' header in {path}")
    need = ("VDWAALS", "EEL", "EGB", "ESURF")
    if any(name not in header for name in need):
        raise ValueError(f"Missing column(s) {[n for n in need if n not in header]} in {path}")
    columns = [header.index(name) for name in need]
    frames = {}
    while i < len(lines) and re.fullmatch(r"\d+", lines[i][0]):
        cells = lines[i]
        frame = int(cells[0])
        if frame in frames or len(cells) < len(header):
            raise ValueError(f"Duplicate frame or malformed row in {path}: {cells!r}")
        frames[frame] = tuple(amber_value(cells[column]) for column in columns)
        i += 1
    if not frames:
        raise ValueError(f"No frame values in the GB binding differences in {path}")
    return frames


def replica_blocks(folder, boundary):
    """Block means of per-frame Delta-H (kcal/mol) for one finished `run` output folder."""
    folder = folder.resolve()
    with (folder / "replica_summary.csv").open(newline="", encoding="utf-8") as handle:
        (summary,) = tuple(csv.DictReader(handle))
    if summary["mode"] != "aqueous" or int(summary["snapshots"]) != AQUEOUS_SNAPSHOTS:
        raise ValueError(f"{folder}: Table S3 requires an aqueous replica with {AQUEOUS_SNAPSHOTS} snapshots.")
    with (folder / "sampled_frames.csv").open(newline="", encoding="utf-8") as handle:
        sampled = list(csv.DictReader(handle))
    sample_ids = list(range(1, AQUEOUS_SNAPSHOTS + 1))
    if [int(row["sample"]) for row in sampled] != sample_ids:
        raise ValueError(f"{folder}: invalid sampling manifest.")
    frames = read_per_frame_delta(folder / "amber_per_frame.csv")
    if set(frames) != set(sample_ids):
        raise ValueError(f"{folder}: Amber per-frame indices do not match sampled_frames.csv.")
    times = [finite(row["time_ns"], "time_ns") for row in sampled]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"{folder}: sampled times must increase.")
    delta_h = [sum(frames[i]) for i in sample_ids]
    if summary["surface_method"] == "msms":
        with (folder / "msms_sasa.csv").open(newline="", encoding="utf-8") as handle:
            surface_rows = list(csv.DictReader(handle))
        if [int(row["sample"]) for row in surface_rows] != sample_ids or any(
            not math.isclose(time, finite(row["time_ns"], "MSMS time_ns"), abs_tol=1e-6)
            for time, row in zip(times, surface_rows)
        ):
            raise ValueError(f"{folder}: MSMS frames do not match sampled_frames.csv.")
        delta_h = [sum(frames[i][:3]) + finite(row["nonpolar"], "MSMS nonpolar")
                   for i, row in zip(sample_ids, surface_rows)]
    if abs(statistics.mean(delta_h) - float(summary["delta_h"])) > 0.05:
        raise ValueError(f"{folder}: per-frame mean Delta-H {statistics.mean(delta_h):.3f} differs from "
                         f"replica_summary.csv ({float(summary['delta_h']):.3f}).")
    buckets = [[] for _ in range(round((WINDOW_NS[1] - WINDOW_NS[0]) / BLOCK_NS))]
    for time, energy in zip(times, delta_h):
        position = (time - WINDOW_NS[0]) / BLOCK_NS
        if math.isclose(position, round(position), abs_tol=1e-8):
            position = round(position)
        block = math.ceil(position) - 1 if boundary == "right" else math.floor(position)
        if not 0 <= block < len(buckets):
            raise ValueError(f"{folder}: {time:g} ns lies outside the {boundary}-closed blocks.")
        buckets[block].append(energy)
    if any(not bucket for bucket in buckets):
        raise ValueError(f"{folder}: at least one 20-ns interval has no snapshots.")
    return summary, [(statistics.mean(bucket), len(bucket)) for bucket in buckets]


def block_analysis(folders, output, compares=(), boundary="right"):
    per_model = {}
    seen = set()
    methods = set()
    rows = []
    for folder in folders:
        summary, blocks = replica_blocks(folder, boundary)
        model, replica = summary["model"], summary["replica"]
        if (model, replica) in seen:
            raise ValueError(f"Duplicate replica: {model}/{replica}")
        seen.add((model, replica))
        methods.add(summary["surface_method"])
        per_model.setdefault(model, []).append([value for value, _ in blocks])
        for k, (value, n) in enumerate(blocks, 1):
            start = WINDOW_NS[0] + (k - 1) * BLOCK_NS
            rows.append(dict(model=model, replica=replica, block=k, start_ns=start,
                             end_ns=start + BLOCK_NS, n_frames=n, delta_h=value))
    if len(methods) != 1:
        raise ValueError("Use one surface method for a Table S3 comparison.")
    mean_by_block = {}
    for model, replicas in per_model.items():
        if len(replicas) != REPLICAS["aqueous"]:
            raise ValueError(f"{model}: supply three independent aqueous replicas.")
        mean_by_block[model] = [statistics.mean(values) for values in zip(*replicas)]
    model_rows = [dict(model=model, n_replicas=len(per_model[model]), block=k,
                       start_ns=WINDOW_NS[0] + (k - 1) * BLOCK_NS, end_ns=WINDOW_NS[0] + k * BLOCK_NS,
                       mean_delta_h=value)
                  for model, values in mean_by_block.items() for k, value in enumerate(values, 1)]
    difference_rows = []
    for first, second in compares:
        if first == second or first not in mean_by_block or second not in mean_by_block:
            raise ValueError(f"--compare {first} {second}: model not found among {sorted(mean_by_block)}.")
        for k, (a, b) in enumerate(zip(mean_by_block[first], mean_by_block[second]), 1):
            start = WINDOW_NS[0] + (k - 1) * BLOCK_NS
            difference_rows.append(dict(first=first, second=second, block=k, start_ns=start,
                                        end_ns=start + BLOCK_NS, delta_delta_h=a - b))
    outputs = [(output, rows), (output.with_name(output.stem + "_model_means.csv"), model_rows)]
    if difference_rows:
        outputs.append((output.with_name(output.stem + "_differences.csv"), difference_rows))
    for path, _ in outputs:
        if path.exists():
            raise FileExistsError(path)
    for path, table in outputs:
        write_csv(path, tuple(table[0]), table)
        print(f"Block results: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    job = sub.add_parser("run", help="Prepare and score one independent trajectory")
    job.add_argument("config", type=Path)
    job.add_argument("output", type=Path)
    job.add_argument("--prepare-only", action="store_true", help="Write inputs without launching any executables")
    aggregate = sub.add_parser("summarize", help="Calculate means and sample SD across replica summaries")
    aggregate.add_argument("output", type=Path)
    aggregate.add_argument("replicas", nargs="+", type=Path, help="replica_summary.csv files")
    aggregate.add_argument("--compare", nargs=2, action="append", default=[], metavar=("FIRST", "SECOND"),
                           help="run Welch's t-test on aqueous Delta-G: mean(FIRST) - mean(SECOND)")
    blk = sub.add_parser("blocks", help="Block-wise Delta-H from finished run folders (needs amber_per_frame.csv)")
    blk.add_argument("output", type=Path)
    blk.add_argument("folders", nargs="+", type=Path, help="output folders of aqueous `run` calculations")
    blk.add_argument("--boundary", choices=("left", "right"), default="right",
                     help="20-ns interval convention, left [a,b) or right (a,b] (default: right)")
    blk.add_argument("--compare", nargs=2, action="append", default=[], metavar=("FIRST", "SECOND"),
                     help="print block-wise Delta-H differences FIRST - SECOND (model means)")
    args = parser.parse_args()
    try:
        if args.action == "run":
            run(args.config, args.output, args.prepare_only)
        elif args.action == "summarize":
            summarize(args.replicas, args.output, args.compare)
        else:
            block_analysis(args.folders, args.output, args.compare, args.boundary)
    except (ValueError, IndexError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

"""Búsqueda base del TP3 en train y validación. Produce el listado para confirmación."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
from torch.nn import functional as F

from piloto import (ANCHORS, BATCH_SIZE, BETAS, LR, MLP, SEEDS, Anchor, evaluate,
                    prepare_data, read_json, write_json)


EPOCHS = 12
SLOTS = 4
SEED = 42
SAMPLER_SEED = 261119
PROMOTED = 2
OPTIMIZERS = ("Adam", "RMSprop", "SGD")
LEARNING_RATES = {
    "Adam": ((1e-4, 1e-2), (1e-4, 2e-2)),
    "RMSprop": ((1e-4, 5e-3), (1e-4, 1e-2)),
    "SGD": ((3e-3, 1e-1), (3e-3, 3e-1)),
}
REGULARIZATION = {
    "referencia": {},
    "l1": {"l1": FloatDistribution(1e-7, 1e-4, log=True)},
    "l2": {"l2": FloatDistribution(1e-6, 3e-3, log=True)},
    "l1_l2": {"l1": FloatDistribution(1e-7, 1e-4, log=True),
              "l2": FloatDistribution(1e-6, 3e-3, log=True)},
    "dropout": {"dropout": FloatDistribution(0.20, 0.50)},
    "batchnorm": {"bn_momentum": FloatDistribution(0.01, 0.20, log=True)},
}


def search_space(family, optimizer):
    extended = family in ("dropout", "batchnorm")
    low, high = LEARNING_RATES[optimizer][extended]
    return {"lr": FloatDistribution(low, high, log=True), **REGULARIZATION[family]}


def anchor_parameters(anchor):
    return {"lr": LR, **{key: getattr(anchor, key) for key in REGULARIZATION[anchor.name]}}


def configuration(family, optimizer, params):
    return {"family": family, "optimizer": optimizer, "lr": params["lr"],
            "regularization": asdict(Anchor(family, **{key: params[key] for key in REGULARIZATION[family]}))}


def build_optimizer(model, config):
    settings = {"lr": config["lr"], "weight_decay": 0.0}
    name = config["optimizer"]
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), betas=BETAS, **settings)
    if name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), alpha=0.99, momentum=0.0, **settings)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **settings)


def fit(config, trial_id, data, path):
    train_x, train_y, val_x, val_y = data
    begin = time.perf_counter()
    settings = Anchor(**config["regularization"])
    model = MLP(train_x.shape[1], settings, SEED)
    optimizer = build_optimizer(model, config)
    # Misma separación de RNG que el piloto, con una semilla de ajuste propia para el filtro.
    torch.manual_seed(SEED + 10_000)
    generator = torch.Generator().manual_seed(SEED + 20_000)
    order_hash = hashlib.sha256()
    kernels = model.kernels
    record = {"trial_id": trial_id, "configuration": config, "seed": SEED, "state": "RUNNING",
              "epochs": [], "updates": 0}
    write_json(path, record)
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        indices = torch.randperm(len(train_y), generator=generator)
        order_hash.update(indices.numpy().tobytes())
        online_ce, online_correct, penalty_sum = 0.0, 0, 0.0
        step_count = 0
        for batch in indices.split(BATCH_SIZE):
            x, y = train_x[batch], train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            penalty = ce.new_zeros(())
            if settings.l1:
                penalty = penalty + settings.l1 * sum(weight.abs().sum() for weight in kernels)
            if settings.l2:
                penalty = penalty + settings.l2 * sum(weight.square().sum() for weight in kernels)
            objective = ce + penalty
            if not torch.isfinite(objective).item():
                break
            objective.backward()
            optimizer.step()
            step_count += 1
            online_ce += ce.detach().item() * len(y)
            penalty_sum += penalty.detach().item() * len(y)
            online_correct += int((logits.detach().argmax(dim=1) == y).sum())
        record["updates"] += step_count
        update_s = time.perf_counter() - started
        if step_count != (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE:
            record.update(state="FAIL", reason="Objetivo no finito durante entrenamiento", failed_epoch=epoch)
            break

        started = time.perf_counter()
        train_ce, train_correct = evaluate(model, train_x, train_y)
        val_ce, val_correct = evaluate(model, val_x, val_y)
        eval_s = time.perf_counter() - started
        if not np.isfinite([online_ce, penalty_sum, train_ce, val_ce]).all():
            record.update(state="FAIL", reason="Métrica de evaluación no finita", failed_epoch=epoch)
            break
        record["epochs"].append({
            "epoch": epoch, "updates": step_count, "lr": config["lr"],
            "train_online_ce": online_ce / len(train_y), "train_online_acc": online_correct / len(train_y),
            "train_eval_ce": train_ce, "train_eval_acc": train_correct / len(train_y),
            "penalty": penalty_sum / len(train_y), "val_ce": val_ce,
            "val_correct": val_correct, "val_acc": val_correct / len(val_y),
            "update_s": update_s, "eval_s": eval_s,
        })
        write_json(path, record)

    if len(record["epochs"]) == EPOCHS:
        counts = np.array([row["val_correct"] for row in record["epochs"]], dtype=np.int64)
        summed = np.convolve(counts, np.ones(3, dtype=np.int64), mode="valid")
        smooth_peak, raw_peak = int(np.argmax(summed)), int(np.argmax(counts))
        record.update(state="COMPLETE", smooth_correct=int(summed[smooth_peak]),
                      smooth_epoch=smooth_peak + 3, smooth_acc=float(summed[smooth_peak] / (3 * len(val_y))),
                      raw_correct=int(counts[raw_peak]), raw_epoch=raw_peak + 1,
                      raw_acc=float(counts[raw_peak] / len(val_y)))
    record.update(wall_s=time.perf_counter() - begin, order_sha256=order_hash.hexdigest())
    write_json(path, record)
    return record


def study_trials(storage, output, data, anchor, optimizer, block):
    block_name = f"{anchor.name}_{optimizer.lower()}"
    space = search_space(anchor.name, optimizer)
    records = []
    for slot in range(SLOTS):
        trial_id = block * SLOTS + slot + 1
        sampler_seed = SAMPLER_SEED + trial_id
        study = optuna.create_study(study_name=block_name, direction="maximize", storage=storage,
                                    load_if_exists=True, sampler=optuna.samplers.RandomSampler(seed=sampler_seed),
                                    pruner=optuna.pruners.NopPruner())
        if not study.get_trials(deepcopy=False) and optimizer == "Adam":
            study.enqueue_trial(anchor_parameters(anchor))

        trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
        existing = trials.get(slot)
        if existing is None or existing.state == TrialState.WAITING:
            live = study.ask(fixed_distributions=space)
            if live.number != slot:
                raise RuntimeError(f"Secuencia de trials inesperada en {block_name}.")
            params = live.params
        else:
            params = existing.params
        if params.keys() != space.keys():
            raise RuntimeError(f"Faltan parámetros persistidos en {block_name}/{slot}.")

        config = configuration(anchor.name, optimizer, params)
        path = output / "trials" / f"trial_{trial_id:03}.json"
        saved = read_json(path) if path.is_file() else None
        if saved is not None and (saved["trial_id"] != trial_id or saved["configuration"] != config):
            raise ValueError(f"El historial del trial {trial_id:03} difiere de Optuna.")
        if existing is not None and existing.state.is_finished() and (
            saved is None or saved["state"] != existing.state.name
        ):
            raise ValueError(f"El trial {trial_id:03} terminó en Optuna sin un historial coherente.")

        if saved is None or saved["state"] == "RUNNING":
            print(f"Inicia {trial_id:02}/72 {anchor.name}/{optimizer} LR={params['lr']:.6g}.", flush=True)
            saved = fit(config, trial_id, data, path)
        saved.update(slot=slot + 1, study=block_name, sampler_seed=sampler_seed,
                     enqueued=slot == 0 and optimizer == "Adam", parameters=params)
        write_json(path, saved)
        if existing is None or not existing.state.is_finished():
            if saved["state"] == "COMPLETE":
                study.tell(slot, saved["smooth_acc"])
            else:
                study.tell(slot, state=TrialState.FAIL)
        note = f"media3={100 * saved['smooth_acc']:.2f}%" if saved["state"] == "COMPLETE" else saved["reason"]
        print(f"Cierra {trial_id:02}/72 {saved['state']} {note} {saved['wall_s']:.1f}s.", flush=True)
        records.append(saved)
    return records


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def report_promoted(output, records):
    promoted, candidates, coverage = [], [], []
    for anchor in ANCHORS:
        family = [record for record in records if record["configuration"]["family"] == anchor.name]
        baseline = configuration(anchor.name, "Adam", anchor_parameters(anchor))
        anchor_id = next(row["trial_id"] for row in family if row["enqueued"])
        ranked = sorted((row for row in family if row["state"] == "COMPLETE"),
                        key=lambda row: (-row["smooth_correct"], row["smooth_epoch"], row["trial_id"]))
        seen = set()
        for row in ranked:
            key = configuration_key(row["configuration"])
            if key in seen:
                continue
            seen.add(key)
            promoted.append({"rank": len(seen), "trial_id": row["trial_id"], **row["configuration"],
                             "smooth_acc": row["smooth_acc"], "smooth_epoch": row["smooth_epoch"],
                             "raw_acc": row["raw_acc"], "raw_epoch": row["raw_epoch"],
                             "reuse_pilot": row["configuration"] == baseline})
            if len(seen) == PROMOTED:
                break
        current = [row for row in promoted if row["family"] == anchor.name]
        for chosen in current:
            candidates.append({**chosen, "seeds": list(SEEDS), "epochs": 30})
        if all(not chosen["reuse_pilot"] for chosen in current):
            candidates.append({**baseline, "trial_id": anchor_id, "reuse_pilot": True,
                               "seeds": list(SEEDS), "epochs": 30})

        for optimizer in OPTIMIZERS:
            group = [row for row in family if row["configuration"]["optimizer"] == optimizer]
            ranges = {key: [min(row["parameters"][key] for row in group),
                            max(row["parameters"][key] for row in group)] for key in group[0]["parameters"]}
            coverage.append({"family": anchor.name, "optimizer": optimizer, "attempts": len(group),
                             "complete": sum(row["state"] == "COMPLETE" for row in group),
                             "enqueued": sum(row["enqueued"] for row in group), "observed": ranges})

    trial_rows = []
    for row in records:
        config = row["configuration"]
        trial_rows.append({"trial_id": row["trial_id"], "family": config["family"], "optimizer": config["optimizer"],
                           "state": row["state"], "enqueued": row["enqueued"], "sampler_seed": row["sampler_seed"],
                           "epochs": len(row["epochs"]), "updates": row["updates"], "wall_s": row["wall_s"],
                           **row["parameters"], **{key: row.get(key) for key in
                              ("smooth_acc", "smooth_epoch", "raw_acc", "raw_epoch", "failed_epoch", "reason")}})
    write_csv(output / "trials.csv", trial_rows)
    write_csv(output / "epocas.csv", [{"trial_id": row["trial_id"], "family": row["configuration"]["family"],
                                        "optimizer": row["configuration"]["optimizer"], **epoch}
                                       for row in records for epoch in row["epochs"]])
    write_csv(output / "promovidos.csv", [{**{k: v for k, v in row.items() if k != "regularization"},
                                           **{k: v for k, v in row["regularization"].items() if k != "name"}}
                                          for row in promoted])
    write_json(output / "cobertura.json", coverage)
    write_json(output / "confirmacion_pendiente.json", candidates)
    details = {"attempts": len(records), "complete": sum(r["state"] == "COMPLETE" for r in records),
               "epochs": sum(len(r["epochs"]) for r in records), "updates": sum(r["updates"] for r in records),
               "wall_trials_s": sum(r["wall_s"] for r in records), "promoted": promoted,
               "confirmation_total": len(candidates), "confirmation_new": sum(not r["reuse_pilot"] for r in candidates)}
    write_json(output / "busqueda_completa.json", details)
    print(f"Búsqueda completa: {details['attempts']} intentos, {details['complete']} completos, "
          f"{details['epochs']} épocas válidas y {len(promoted)} promovidos.", flush=True)


def configuration_key(config):
    return config["optimizer"], config["lr"], tuple(config["regularization"].items())


def run_search(data_dir, pilot_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pilot = read_json(pilot_dir / "protocolo.json")
    script = Path(__file__)
    pilot_code_hash = hashlib.sha256(script.with_name("piloto.py").read_bytes()).hexdigest()
    if pilot["script_sha256"] != pilot_code_hash or len(read_json(pilot_dir / "piloto_completo.json")) != 6:
        raise ValueError("Falta un piloto completo y compatible con el modelo reutilizado.")
    manifest = {"protocol": "base", "epochs": EPOCHS, "attempts_per_family_optimizer": SLOTS,
                "seed": SEED, "sampler_seed_base": SAMPLER_SEED, "sampler": "RandomSampler",
                "pruner": "NopPruner", "promoted_per_family": PROMOTED, "pilot_script_sha256": pilot_code_hash,
                "search_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "packages": {pkg: version(pkg) for pkg in ("torch", "numpy", "optuna", "scikit-learn", "nltk")}}
    manifest_path = output / "protocolo.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("El directorio de salida pertenece a otro protocolo.")
    write_json(manifest_path, manifest)
    print("Preparando y verificando los datos del piloto...", flush=True)
    data, info = prepare_data(data_dir)
    pilot_data = read_json(pilot_dir / "datos.json")
    if any(info[key] != pilot_data[key] for key in ("split_sha256", "vocabulary", "idf")):
        raise ValueError("Los datos transformados difieren del piloto.")
    if not (output / "datos.json").exists():
        write_json(output / "datos.json", {"split_sha256": info["split_sha256"], "input_dim": info["input_dim"],
                                         "preparation_s": info["preparation_s"]})
    print("Partición, vocabulario e IDF coinciden con el piloto.", flush=True)

    storage = optuna.storages.RDBStorage(url="sqlite:///" + (output / "optuna.sqlite3").as_posix())
    records = []
    for family_index, anchor in enumerate(ANCHORS):
        for optimizer_index, optimizer in enumerate(OPTIMIZERS):
            block = family_index * len(OPTIMIZERS) + optimizer_index
            records.extend(study_trials(storage, output, data, anchor, optimizer, block))
    report_promoted(output, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_search(args.data_dir.resolve(), args.pilot.resolve(), args.output.resolve())

"""Confirma los promovidos del TP3 exclusivamente con train interno y validación."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from busqueda_optuna import anchor_parameters, build_optimizer, configuration, write_csv
from piloto import (ANCHORS, BATCH_SIZE, EPOCHS, MLP, NUM_CLASSES, SEEDS, Anchor,
                    bn_scales, evaluate, layer_statistics, prepare_data, read_json,
                    save_model, write_json)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_config(item):
    return {key: item[key] for key in ("family", "optimizer", "lr", "regularization")}


def check_plan(search, pilot):
    selection = search / "confirmacion_pendiente.json"
    manifest = read_json(search / "protocolo.json")
    status = read_json(search / "verificacion.json")
    items = read_json(selection)
    assert manifest["search_script_sha256"] == digest(Path(__file__).with_name("busqueda_optuna.py"))
    assert manifest["pilot_script_sha256"] == digest(Path(__file__).with_name("piloto.py"))
    assert manifest["pilot_script_sha256"] == read_json(pilot / "protocolo.json")["script_sha256"]
    assert (status["attempts"], status["complete"], status["epochs"]) == (72, 72, 864)
    assert len(items) == 17 and len({item["trial_id"] for item in items}) == 17
    assert sum(not item["reuse_pilot"] for item in items) == 11
    assert all(item["seeds"] == list(SEEDS) and item["epochs"] == EPOCHS for item in items)
    expected = {anchor.name: configuration(anchor.name, "Adam", anchor_parameters(anchor)) for anchor in ANCHORS}
    assert {item["family"]: candidate_config(item) for item in items if item["reuse_pilot"]} == expected
    assert [item["trial_id"] for item in items if "rank" in item] == status["verified_promoted_ids"]
    for item in items:
        source = read_json(search / "trials" / f"trial_{item['trial_id']:03}.json")
        assert source["state"] == "COMPLETE" and candidate_config(item) == source["configuration"]
    return items


def fit_seed(item, seed, data, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    history_path = scratch / "corrida.json"
    if history_path.exists():
        record = read_json(history_path)
        if record["state"] in ("COMPLETE", "FAIL"):
            return record
        print(f"Reinicia la corrida incompleta #{item['trial_id']:02}/{seed}.", flush=True)

    train_x, train_y, val_x, val_y = data
    start = time.perf_counter()
    settings = Anchor(**item["regularization"])
    model = MLP(train_x.shape[1], settings, seed)
    optimizer = build_optimizer(model, item)
    torch.manual_seed(seed + 10_000)
    generator = torch.Generator().manual_seed(seed + 20_000)
    order_hash, init_hash = hashlib.sha256(), hashlib.sha256()
    kernels = model.kernels
    for kernel in kernels:
        init_hash.update(kernel.detach().numpy().tobytes())
    record = {"trial_id": item["trial_id"], "seed": seed, "configuration": candidate_config(item),
              "anchor": item["regularization"], "state": "RUNNING", "epochs": [],
              "initial_kernels_sha256": init_hash.hexdigest(), "initial_weights": layer_statistics(model.state_dict()),
              "updates": 0}
    io_s = 0.0
    write_json(history_path, record)
    print(f"Inicia #{item['trial_id']:02} {item['family']}/{item['optimizer']} semilla {seed}.", flush=True)
    for epoch in range(1, EPOCHS + 1):
        clock = time.perf_counter()
        model.train()
        indices = torch.randperm(len(train_y), generator=generator)
        order_hash.update(indices.numpy().tobytes())
        online_ce, online_correct, penalty_sum, updates = 0.0, 0, 0.0, 0
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
            updates += 1
            online_ce += ce.detach().item() * len(y)
            penalty_sum += penalty.detach().item() * len(y)
            online_correct += int((logits.detach().argmax(dim=1) == y).sum())
        update_s = time.perf_counter() - clock
        record["updates"] += updates
        if updates != (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE:
            record.update(state="FAIL", reason="Objetivo de entrenamiento no finito", failed_epoch=epoch)
            break

        clock = time.perf_counter()
        train_ce, train_correct = evaluate(model, train_x, train_y)
        val_ce, val_correct = evaluate(model, val_x, val_y)
        eval_s = time.perf_counter() - clock
        if not np.isfinite([online_ce, penalty_sum, train_ce, val_ce]).all():
            record.update(state="FAIL", reason="Métrica no finita", failed_epoch=epoch)
            break
        record["epochs"].append({"epoch": epoch, "lr": item["lr"], "updates": updates,
                                 "train_online_ce": online_ce / len(train_y), "train_online_acc": online_correct / len(train_y),
                                 "train_eval_ce": train_ce, "train_eval_acc": train_correct / len(train_y),
                                 "penalty": penalty_sum / len(train_y), "val_ce": val_ce, "val_correct": val_correct,
                                 "val_acc": val_correct / len(val_y), "update_s": update_s, "eval_s": eval_s})
        clock = time.perf_counter()
        save_model(scratch / f"epoca_{epoch:02}.pt", model, settings, seed, epoch, train_x.shape[1])
        write_json(history_path, record)
        io_s += time.perf_counter() - clock
        if epoch % 10 == 0:
            print(f"#{item['trial_id']:02}/{seed} e{epoch:02} val={val_correct / len(val_y):.4f}.", flush=True)

    if len(record["epochs"]) == EPOCHS:
        record["state"] = "COMPLETE"
    record.update(wall_s=time.perf_counter() - start, checkpoint_logging_s=io_s,
                  order_sha256=order_hash.hexdigest())
    write_json(history_path, record)
    print(f"Cierra #{item['trial_id']:02}/{seed} {record['state']} en {record['wall_s']:.1f}s.", flush=True)
    return record


def selected_epoch(records):
    counts = np.array([[row["val_correct"] for row in r["epochs"]] for r in records], dtype=np.int64)
    totals = counts.sum(axis=0)
    best = int(totals.argmax())
    return best + 1, int(totals[best])


def run_candidate(item, data, output):
    group = f"trial_{item['trial_id']:03}"
    destination = output / "corridas" / group
    summary = destination / "resumen.json"
    if summary.is_file():
        return
    scratch = output / "temporales" / group
    records = [fit_seed(item, seed, data, scratch / str(seed)) for seed in SEEDS]
    eligible = all(r["state"] == "COMPLETE" for r in records)
    epoch, total_correct = selected_epoch(records) if eligible else (None, None)
    for record in records:
        seed = record["seed"]
        source = scratch / str(seed)
        dest = destination / str(seed)
        dest.mkdir(parents=True, exist_ok=True)
        if eligible:
            checkpoint = source / f"epoca_{epoch:02}.pt"
            selected = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
            last = torch.load(source / f"epoca_{EPOCHS:02}.pt", weights_only=True, map_location="cpu")["state_dict"]
            record.update(selected_epoch=epoch, selected_weights=layer_statistics(selected),
                          last_weights=layer_statistics(last),
                          bn_scales={"selected": bn_scales(selected), "last": bn_scales(last)})
            shutil.copyfile(checkpoint, dest / "checkpoint.pt")
        write_json(dest / "historial.json", record)
    write_json(summary, {"trial_id": item["trial_id"], "eligible": eligible,
                         "selected_epoch": epoch, "val_correct_sum": total_correct})
    for seed in SEEDS:
        source = scratch / str(seed)
        for checkpoint in source.glob("epoca_*.pt"):
            checkpoint.unlink()
        (source / "corrida.json").unlink()
        source.rmdir()
    scratch.rmdir()
    print(f"Configuración #{item['trial_id']:02} cerrada; época común {epoch}.", flush=True)


@torch.inference_mode()
def validate_checkpoint(path, config, seed, epoch, data):
    _, _, x, y = data
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    assert (checkpoint["anchor"], checkpoint["seed"], checkpoint["epoch"], checkpoint["input_dim"]) == (
        config["regularization"], seed, epoch, x.shape[1])
    model = MLP(x.shape[1], Anchor(**config["regularization"]), seed)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    counts = torch.zeros(NUM_CLASSES**2, dtype=torch.int64)
    for batch, target in zip(x.split(BATCH_SIZE), y.split(BATCH_SIZE)):
        counts += torch.bincount(NUM_CLASSES * target + model(batch).argmax(dim=1), minlength=NUM_CLASSES**2)
    matrix = counts.reshape(NUM_CLASSES, NUM_CLASSES).numpy()
    matches = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2.0 * matches, denominator, out=np.zeros(NUM_CLASSES), where=denominator != 0)
    return int(matches.sum()), float(f1.mean())


def finish(items, data, output, pilot):
    candidates, seeds, histories = [], [], []
    reference_hashes = {}
    project = output.parent.parent
    for item in items:
        reuse = item["reuse_pilot"]
        directory = (pilot / "corridas" / item["family"] if reuse else
                     output / "corridas" / f"trial_{item['trial_id']:03}")
        records = [read_json(directory / str(seed) / "historial.json") for seed in SEEDS]
        eligible = all(len(r["epochs"]) == EPOCHS and r.get("state", "COMPLETE") == "COMPLETE" for r in records)
        config = candidate_config(item)
        if eligible:
            epoch, correct_sum = selected_epoch(records)
            accuracy, loss, f1_values, paths = [], [], [], []
            for record in records:
                seed = record["seed"]
                path = directory / str(seed) / "checkpoint.pt"
                assert record["selected_epoch"] == epoch
                count, f1 = validate_checkpoint(path, config, seed, epoch, data)
                row = record["epochs"][epoch - 1]
                assert count == row["val_correct"]
                accuracy.append(row["val_acc"])
                loss.append(row["val_ce"])
                f1_values.append(f1)
                paths.append({"seed": seed, "path": path.relative_to(project).as_posix(), "sha256": digest(path)})
            summary = {"epoch": epoch, "val_correct_sum": correct_sum,
                       "val_acc_mean": correct_sum / (len(SEEDS) * len(data[3])), "val_acc_sd": float(np.std(accuracy, ddof=1)),
                       "val_ce_mean": float(np.mean(loss)), "val_f1_mean": float(np.mean(f1_values)),
                       "val_f1_sd": float(np.std(f1_values, ddof=1)), "checkpoints": paths}
        else:
            summary = {}

        candidates.append({"trial_id": item["trial_id"], **config, "reuse_pilot": reuse, "eligible": eligible, **summary})
        for index, record in enumerate(records):
            seed = record["seed"]
            signature = record["initial_kernels_sha256"], record["order_sha256"]
            if len(record["epochs"]) == EPOCHS:
                assert reference_hashes.setdefault(seed, signature) == signature
            for row in record["epochs"]:
                histories.append({"trial_id": item["trial_id"], "family": item["family"],
                                  "optimizer": item["optimizer"], "seed": seed, "reuse_pilot": reuse, **row})
            seed_row = {"trial_id": item["trial_id"], "family": item["family"], "seed": seed, "reuse_pilot": reuse,
                        "state": record.get("state", "COMPLETE"), "epochs": len(record["epochs"]), "wall_s": record["wall_s"],
                        "updates": record.get("updates", sum(row["updates"] for row in record["epochs"]))}
            if eligible:
                seed_row.update(epoch=epoch, val_acc=accuracy[index], val_ce=loss[index], val_f1=f1_values[index])
            seeds.append(seed_row)

    winners = []
    for anchor in ANCHORS:
        options = [item for item in candidates if item["family"] == anchor.name and item["eligible"]]
        ranked = sorted(options, key=lambda item: (-item["val_correct_sum"], item["epoch"], item["trial_id"]))
        for rank, item in enumerate(ranked, start=1):
            item["validation_rank"] = rank
        winners.append(ranked[0])
    new = [row for row in seeds if not row["reuse_pilot"]]
    assert len(new) == 33 and len(seeds) == 51 and len(reference_hashes) == 3
    write_csv(output / "corridas.csv", seeds)
    write_csv(output / "epocas.csv", histories)
    write_csv(output / "configuraciones.csv", [
        {**{k: v for k, v in item.items() if k not in ("checkpoints", "regularization")},
         **{k: v for k, v in item["regularization"].items() if k != "name"}} for item in candidates])
    write_json(output / "seleccion_validacion.json", {"rule": "max_val_correct_sum_then_first_epoch_then_trial_id",
                                                       "val_samples": len(data[3]), "seeds": list(SEEDS), "families": winners})
    result = {"new_runs": len(new), "complete_new_runs": sum(row["state"] == "COMPLETE" for row in new),
              "new_epochs": sum(row["epochs"] for row in new), "new_updates": sum(row["updates"] for row in new),
              "new_wall_s": sum(row["wall_s"] for row in new), "eligible_configurations": sum(c["eligible"] for c in candidates),
              "selection": winners}
    write_json(output / "confirmacion_completa.json", result)
    print(f"Confirmación cerrada: {result['complete_new_runs']}/33 corridas nuevas completas, "
          f"{result['new_epochs']} épocas y {result['eligible_configurations']} configuraciones elegibles.", flush=True)


def run(data_dir, pilot, search, output):
    items = check_plan(search, pilot)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if shutil.disk_usage(output).free < 2 * 2**30:
        raise OSError("Espacio insuficiente para los checkpoints de confirmación.")
    manifest = {"source_sha256": digest(search / "confirmacion_pendiente.json"),
                "script_sha256": digest(Path(__file__)), "seeds": list(SEEDS), "epochs": EPOCHS,
                "new_configurations": 11, "reuse_pilot": 6,
                "packages": {p: version(p) for p in ("torch", "numpy", "nltk", "scikit-learn")}}
    target = output / "protocolo.json"
    if target.is_file() and read_json(target) != manifest:
        raise ValueError("El directorio de confirmación pertenece a otro protocolo.")
    write_json(target, manifest)
    print("Verificando la representación del piloto para la confirmación...", flush=True)
    data, info = prepare_data(data_dir)
    pilot_data = read_json(pilot / "datos.json")
    assert all(info[k] == pilot_data[k] for k in ("split_sha256", "vocabulary", "idf"))
    data_path = output / "datos.json"
    if not data_path.is_file():
        write_json(data_path, {"split_sha256": info["split_sha256"], "preparation_s": info["preparation_s"],
                              "input_dim": info["input_dim"]})
    print("Partición, vocabulario e IDF coinciden con el piloto.", flush=True)
    new = [item for item in items if not item["reuse_pilot"]]
    for i, item in enumerate(new, start=1):
        print(f"Configuración nueva {i}/{len(new)}: trial {item['trial_id']:02}.", flush=True)
        run_candidate(item, data, output)
    finish(items, data, output, pilot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "pilot", "search", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.pilot.resolve(), args.search.resolve(), args.output.resolve())

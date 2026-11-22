"""Ejecuta las cinco variantes complementarias del TP3 en train y validación."""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from confirmacion import digest
from piloto import (ANCHORS, BATCH_SIZE, BETAS, EPOCHS, LR, MLP, SEEDS, WIDTHS,
                    evaluate, layer_statistics, prepare_data, read_json, write_json)


@dataclass(frozen=True)
class Case:
    name: str
    anchor: str
    plateau: bool = False
    max_norm: float | None = None
    standardize: bool = False
    first_layer_gain: float = 1.0


CASES = (
    Case("dropout_plateau", "dropout", plateau=True),
    Case("dropout_max_norm", "dropout", max_norm=5.0),
    Case("dropout_plateau_max_norm", "dropout", plateau=True, max_norm=5.0),
    Case("xavier_r_x2", "referencia", first_layer_gain=2.0),
    Case("tfidf_estandarizado", "referencia", standardize=True),
)
PLATEAU = {"mode": "max", "factor": 0.5, "patience": 3, "threshold": 0.0, "min_lr": 1e-4}


def scaled_data(data):
    train_x, train_y, val_x, val_y = data
    if psutil.virtual_memory().available < (train_x.numel() + val_x.numel()) * 4 + 512 * 2**20:
        raise MemoryError("RAM insuficiente para mantener una copia estandarizada.")
    start = time.perf_counter()
    scaler = StandardScaler(copy=True)
    scaler.fit(train_x.numpy())
    train = scaler.transform(train_x.numpy())
    val = scaler.transform(val_x.numpy())
    elapsed = time.perf_counter() - start
    assert train.dtype == val.dtype == np.float32
    assert not np.shares_memory(train, train_x.numpy()) and not np.shares_memory(val, val_x.numpy())
    assert np.isfinite(train).all() and np.isfinite(val).all()
    document_frequency = np.zeros(train.shape[1], dtype=np.int64)
    for batch in train_x.split(BATCH_SIZE):
        document_frequency += torch.count_nonzero(batch, dim=0).numpy()
    amplification = {}
    for label, mask in (("2_a_5_documentos", (document_frequency >= 2) & (document_frequency <= 5)),
                        ("al_menos_100_documentos", document_frequency >= 100)):
        values = 1 / scaler.scale_[mask]
        amplification[label] = {"n": len(values), "median_inverse_sd": float(np.median(values)) if len(values) else None}
    stats = {"fit_scope": "train_internal", "train_samples": len(train), "val_samples": len(val),
             "fit_transform_s": elapsed, "scale_quantiles": np.quantile(scaler.scale_, (0, 0.05, 0.5, 0.95, 1)).tolist(),
             "amplification_by_document_frequency": amplification,
             "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    return (torch.from_numpy(train), train_y, torch.from_numpy(val), val_y), stats


@torch.no_grad()
def project_kernels(hidden, bound):
    hits = []
    for weight in hidden:
        norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
        hits.append(int((norms > bound).sum()))
        weight.mul_(bound / norms.clamp_min(bound))
    return hits


def fit(case, seed, data, output, pilot_record):
    if output.is_file():
        archived = read_json(output)
        if archived["state"] in ("COMPLETE", "FAIL"):
            return archived
        print(f"Reinicia {case.name}/{seed} desde la semilla original.", flush=True)
    train_x, train_y, val_x, val_y = data
    start = time.perf_counter()
    anchor = next(anchor for anchor in ANCHORS if anchor.name == case.anchor)
    model = MLP(train_x.shape[1], anchor, seed)
    kernels = model.kernels
    initial = hashlib.sha256()
    for kernel in kernels:
        initial.update(kernel.detach().numpy().tobytes())
    assert initial.hexdigest() == pilot_record["initial_kernels_sha256"]
    preactivation = None
    if case.first_layer_gain != 1.0:
        with torch.no_grad():
            first_linear = model.layers[0]
            examples = train_x[:BATCH_SIZE]
            before = first_linear(examples)
            kernels[0].mul_(case.first_layer_gain)
            after = first_linear(examples)
            torch.testing.assert_close(after, before * case.first_layer_gain)
            preactivation = {"samples": len(examples), "baseline_rms": before.square().mean().sqrt().item(),
                             "variant_rms": after.square().mean().sqrt().item()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **PLATEAU) if case.plateau else None
    torch.manual_seed(seed + 10_000)
    generator = torch.Generator().manual_seed(seed + 20_000)
    order = hashlib.sha256()
    record = {"case": asdict(case), "seed": seed, "state": "RUNNING", "epochs": [], "updates": 0,
              "base_initial_sha256": initial.hexdigest(), "initial_preactivation": preactivation,
              "initial_weights": layer_statistics(model.state_dict())}
    write_json(output, record)
    print(f"Inicia {case.name}/{seed}.", flush=True)
    for epoch in range(1, EPOCHS + 1):
        clock = time.perf_counter()
        used_lr = optimizer.param_groups[0]["lr"]
        model.train()
        indices = torch.randperm(len(train_y), generator=generator)
        order.update(indices.numpy().tobytes())
        online_ce, online_correct, updates = 0.0, 0, 0
        projections = [0] * len(WIDTHS)
        projection_s = 0.0
        for batch in indices.split(BATCH_SIZE):
            x, y = train_x[batch], train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            if not torch.isfinite(ce).item():
                break
            ce.backward()
            optimizer.step()
            if case.max_norm is not None:
                projection_start = time.perf_counter()
                hits = project_kernels(kernels[:-1], case.max_norm)
                projection_s += time.perf_counter() - projection_start
                projections = [previous + current for previous, current in zip(projections, hits)]
            online_ce += ce.detach().item() * len(y)
            online_correct += int((logits.detach().argmax(dim=1) == y).sum())
            updates += 1
        update_s = time.perf_counter() - clock
        record["updates"] += updates
        if updates != (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE:
            record.update(state="FAIL", reason="Entrenamiento con pérdida no finita", failed_epoch=epoch)
            break

        clock = time.perf_counter()
        train_ce, train_correct = evaluate(model, train_x, train_y)
        val_ce, val_correct = evaluate(model, val_x, val_y)
        eval_s = time.perf_counter() - clock
        if not np.isfinite([online_ce, train_ce, val_ce]).all():
            record.update(state="FAIL", reason="Evaluación con pérdida no finita", failed_epoch=epoch)
            break
        val_acc = val_correct / len(val_y)
        if scheduler is not None:
            scheduler.step(val_acc)
        post_norms = [torch.linalg.vector_norm(w.detach(), dim=1).max().item() for w in kernels[:-1]]
        if case.max_norm is not None:
            assert max(post_norms) <= case.max_norm + 1e-5
        record["epochs"].append({"epoch": epoch, "updates": updates, "lr": used_lr,
                                 "lr_next": optimizer.param_groups[0]["lr"], "train_online_ce": online_ce / len(train_y),
                                 "train_online_acc": online_correct / len(train_y), "train_eval_ce": train_ce,
                                 "train_eval_acc": train_correct / len(train_y), "val_ce": val_ce,
                                 "val_correct": val_correct, "val_acc": val_acc, "update_s": update_s,
                                 "eval_s": eval_s, "clipped_hidden_1": projections[0], "clipped_hidden_2": projections[1],
                                 "norm_hidden_1_max": post_norms[0], "norm_hidden_2_max": post_norms[1],
                                 "projection_s": projection_s})
        write_json(output, record)
        if epoch % 10 == 0:
            print(f"{case.name}/{seed} e{epoch:02}: val={val_acc:.4f}, LR={used_lr:.6g}.", flush=True)

    if len(record["epochs"]) == EPOCHS:
        assert order.hexdigest() == pilot_record["order_sha256"]
        record.update(state="COMPLETE", last_weights=layer_statistics(model.state_dict()))
    record.update(order_sha256=order.hexdigest(), wall_s=time.perf_counter() - start)
    write_json(output, record)
    print(f"Cierra {case.name}/{seed} {record['state']} en {record['wall_s']:.1f}s.", flush=True)
    return record


def run(data_dir, project, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    pilot = project / "resultados" / "piloto"
    frozen = pilot / "datos.json"
    history = {(anchor, seed): read_json(pilot / "corridas" / anchor / str(seed) / "historial.json")
               for anchor in ("referencia", "dropout") for seed in SEEDS}
    assert all(len(record["epochs"]) == EPOCHS for record in history.values())
    assert read_json(pilot / "protocolo.json")["script_sha256"] == digest(Path(__file__).with_name("piloto.py"))
    manifest = {"script_sha256": digest(Path(__file__)), "pilot_data_sha256": digest(frozen),
                "pilot_script_sha256": digest(Path(__file__).with_name("piloto.py")),
                "selection_sha256": digest(project / "resultados" / "confirmacion" / "seleccion_validacion.json"),
                "test_summary_sha256": digest(project / "resultados" / "test_final" / "test_completo.json"),
                "cases": [asdict(case) for case in CASES], "plateau": PLATEAU, "seeds": list(SEEDS),
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "betas": list(BETAS),
                "packages": {p: version(p) for p in ("torch", "numpy", "nltk", "scikit-learn")}}
    protocol = output / "protocolo.json"
    if protocol.exists() and read_json(protocol) != manifest:
        raise ValueError("Este directorio ya tiene resultados de otro protocolo.")
    write_json(protocol, manifest)
    print("Verificando la partición y el TF-IDF del piloto...", flush=True)
    data, info = prepare_data(data_dir)
    original = read_json(frozen)
    assert all(info[key] == original[key] for key in ("split_sha256", "vocabulary", "idf"))
    data_path = output / "datos.json"
    if not data_path.exists():
        write_json(data_path, {key: info[key] for key in ("split_sha256", "preparation_s", "input_dim")})
    records = []
    for case in CASES:
        if case.standardize:
            data, scaling = scaled_data(data)
            write_json(output / "escalado.json", scaling)
        for seed in SEEDS:
            destination = output / "corridas" / case.name / str(seed) / "historial.json"
            records.append(fit(case, seed, data, destination, history[case.anchor, seed]))

    write_json(output / "diagnosticos_completos.json", {
        "runs": len(records), "complete_runs": sum(r["state"] == "COMPLETE" for r in records),
        "epochs": sum(len(r["epochs"]) for r in records), "updates": sum(r["updates"] for r in records),
        "wall_runs_s": sum(r["wall_s"] for r in records), "scope": ["train_internal", "validation"],
    })
    print(f"Diagnósticos cerrados: {len(records)} corridas, {sum(len(r['epochs']) for r in records)} épocas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "project", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.project.resolve(), args.output.resolve())

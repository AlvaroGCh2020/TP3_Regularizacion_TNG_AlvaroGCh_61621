"""Piloto del TP3. Las únicas corridas con TNG son las seis anclas del plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import nltk
import numpy as np
import psutil
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


SEEDS = (101, 202, 303)
EPOCHS = 30
BATCH_SIZE = 128
WIDTHS = (256, 128)
NUM_CLASSES = 20
LR, BETAS = 0.003, (0.9, 0.999)
SPLIT_SEED = 42
TP2_SPLIT_HASH = "a2c0bbe460b45e4761476993705baf6afc8fc5f5e255dc6eda2a3c4e3fb4c87f"
HISTOGRAM_BINS = np.r_[0.0, np.geomspace(1e-7, 10, 57)]


@dataclass(frozen=True)
class Anchor:
    name: str
    l1: float = 0.0
    l2: float = 0.0
    dropout: float = 0.0
    bn_momentum: float | None = None


ANCHORS = (
    Anchor("referencia"),
    Anchor("l1", l1=1e-5),
    Anchor("l2", l2=1e-4),
    Anchor("l1_l2", l1=1e-5, l2=1e-4),
    Anchor("dropout", dropout=0.35),
    Anchor("batchnorm", bn_momentum=0.10),
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, anchor: Anchor, seed: int):
        super().__init__()
        layers = []
        init = torch.Generator().manual_seed(seed)
        for previous, width in zip((input_dim, *WIDTHS), (*WIDTHS, NUM_CLASSES)):
            hidden = width != NUM_CLASSES
            use_bn = hidden and anchor.bn_momentum is not None
            layer = nn.Linear(previous, width, bias=not use_bn)
            nn.init.xavier_uniform_(layer.weight, gain=1.0, generator=init)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layers.append(layer)
            if use_bn:
                layers.append(nn.BatchNorm1d(width, eps=1e-5, momentum=anchor.bn_momentum))
            if hidden:
                layers.append(nn.ReLU())
                if anchor.dropout:
                    layers.append(nn.Dropout(anchor.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @property
    def kernels(self):
        return [layer.weight for layer in self.layers if isinstance(layer, nn.Linear)]


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(path, model, anchor, seed, epoch, input_dim):
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "anchor": asdict(anchor), "seed": seed,
                "epoch": epoch, "input_dim": input_dim}, temporary)
    temporary.replace(path)


def prepare_data(data_dir):
    nltk.data.path.insert(0, str(data_dir / "nltk_data"))
    start = time.perf_counter()
    dataset = fetch_20newsgroups(subset="train", data_home=str(data_dir / "sklearn"),
                                 download_if_missing=False, remove=("headers", "footers", "quotes"))
    labels = np.asarray(dataset.target, dtype=np.int64)
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                         random_state=SPLIT_SEED, stratify=labels)
    split_hash = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
    if split_hash != TP2_SPLIT_HASH or (len(train_idx), len(val_idx)) != (9051, 2263):
        raise ValueError("La partición interna difiere de la verificada para el TP2.")
    loaded = time.perf_counter()

    # Mismo orden de operaciones del caso de sobreajuste del notebook base.
    english_stopwords = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    documents = [" ".join(lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text.lower())
                          if token.isalpha() and token not in english_stopwords) for text in dataset.data]
    processed = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=10_000, min_df=2,
                                dtype=np.float32, norm="l2")
    train = vectorizer.fit_transform([documents[i] for i in train_idx])
    val = vectorizer.transform([documents[i] for i in val_idx])
    vectorized = time.perf_counter()
    dense_bytes = (train.shape[0] + val.shape[0]) * train.shape[1] * 4
    if psutil.virtual_memory().available < dense_bytes * 1.5 + 512 * 2**20:
        raise MemoryError("RAM disponible insuficiente para densificar TF-IDF en float32.")
    train_x, val_x = torch.from_numpy(train.toarray()), torch.from_numpy(val.toarray())
    end = time.perf_counter()
    # Varianza poblacional por columna, estimada exclusivamente sobre train interno.
    train_mean = np.asarray(train.mean(axis=0)).ravel().astype(np.float64)
    train_square_mean = np.asarray(train.power(2).mean(axis=0)).ravel().astype(np.float64)
    std = np.sqrt(np.maximum(0, train_square_mean - train_mean**2))
    info = {
        "split_sha256": split_hash, "split_seed": SPLIT_SEED, "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(), "classes": list(dataset.target_names), "input_dim": train.shape[1],
        "vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(), "zero_train_fraction": float(np.mean(train.getnnz(axis=1) == 0)),
        "train_col_std_q": np.quantile(std, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        "dense_mib": dense_bytes / 2**20,
        "preparation_s": {"loading_split": loaded - start, "text": processed - loaded,
                          "tfidf": vectorized - processed, "dense": end - vectorized},
    }
    return (train_x, torch.from_numpy(labels[train_idx]), val_x, torch.from_numpy(labels[val_idx])), info


@torch.inference_mode()
def evaluate(model, x, y):
    model.eval()
    loss = 0.0
    correct = 0
    for inputs, targets in zip(x.split(BATCH_SIZE), y.split(BATCH_SIZE)):
        scores = model(inputs)
        loss += F.cross_entropy(scores, targets, reduction="sum").item()
        correct += int((scores.argmax(dim=1) == targets).sum())
    return loss / len(y), correct


def layer_statistics(state):
    statistics = []
    for key, tensor in state.items():
        if tensor.ndim != 2:
            continue
        values = np.abs(tensor.detach().numpy().ravel())
        hist, _ = np.histogram(values, bins=HISTOGRAM_BINS)
        med, p95, p99 = np.quantile(values, (0.50, 0.95, 0.99)).tolist()
        statistics.append({"layer": key, "n": values.size, "median": med, "p95": p95, "p99": p99,
                           "max": float(values.max()), "zero_fraction": float(np.mean(values == 0)),
                           "near_zero_fraction": float(np.mean(values <= 1e-3)),
                           "histogram": hist.tolist(), "overflow": int((values > HISTOGRAM_BINS[-1]).sum())})
    return statistics


def bn_scales(state):
    values = []
    for name, tensor in state.items():
        if name.endswith("running_var"):
            prefix = name.removesuffix("running_var")
            gamma = state[prefix + "weight"].numpy()
            scale = gamma / np.sqrt(tensor.numpy() + 1e-5)
            values.append({"layer": prefix.rstrip("."), "gamma": gamma.tolist(), "eval_scale": scale.tolist()})
    return values


def run_seed(anchor, seed, data, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    history_file = scratch / "corrida.json"
    if history_file.exists():
        saved = read_json(history_file)
        if len(saved["epochs"]) == EPOCHS and "wall_s" in saved:
            return saved
        print(f"Se repite la corrida incompleta {anchor.name}/{seed} desde su semilla.", flush=True)

    train_x, train_y, val_x, val_y = data
    started = time.perf_counter()
    model = MLP(train_x.shape[1], anchor, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=0)
    # La inicialización, el orden de ejemplos y el enmascarado tienen secuencias independientes.
    torch.manual_seed(seed + 10_000)
    order = torch.Generator().manual_seed(seed + 20_000)
    order_digest = hashlib.sha256()
    initial_digest = hashlib.sha256()
    for kernel in model.kernels:
        initial_digest.update(kernel.detach().numpy().tobytes())
    kernels = model.kernels
    bn_samples = []
    diagnostic_s = 0.0

    def record_bn(_module, args):
        nonlocal diagnostic_s
        if len(bn_samples) < 5:
            start = time.perf_counter()
            variance = args[0].detach().var(dim=0, correction=0).numpy()
            bn_samples.append({"batch": len(bn_samples) + 1,
                               "variance_q": np.quantile(variance, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                               "fraction_below_eps": float(np.mean(variance < 1e-5))})
            diagnostic_s += time.perf_counter() - start

    bn_hook = None
    if anchor.bn_momentum is not None:
        bn = next(layer for layer in model.layers if isinstance(layer, nn.BatchNorm1d))
        bn_hook = bn.register_forward_pre_hook(record_bn)

    record = {"anchor": asdict(anchor), "seed": seed, "epochs": [], "bn_batches": bn_samples,
              "initial_kernels_sha256": initial_digest.hexdigest(),
              "initial_weights": layer_statistics(model.state_dict())}
    print(f"Inicia {anchor.name}/{seed} ({EPOCHS} épocas).", flush=True)
    io_s = 0.0
    for epoch in range(1, EPOCHS + 1):
        clock = time.perf_counter()
        diagnostic_before = diagnostic_s
        model.train()
        indices = torch.randperm(len(train_y), generator=order)
        order_digest.update(indices.numpy().tobytes())
        online_ce, online_correct, penalty_sum = 0.0, 0, 0.0
        for batch in indices.split(BATCH_SIZE):
            x, y = train_x[batch], train_y[batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            penalty = ce.new_zeros(())
            if anchor.l1:
                penalty = penalty + anchor.l1 * sum(weight.abs().sum() for weight in kernels)
            if anchor.l2:
                penalty = penalty + anchor.l2 * sum(weight.square().sum() for weight in kernels)
            (ce + penalty).backward()
            optimizer.step()
            online_ce += ce.detach().item() * len(y)
            penalty_sum += penalty.detach().item() * len(y)
            online_correct += int((logits.detach().argmax(dim=1) == y).sum())
        update_s = time.perf_counter() - clock - (diagnostic_s - diagnostic_before)
        if epoch == 1 and bn_hook is not None:
            bn_hook.remove()
        clock = time.perf_counter()
        train_loss, train_correct = evaluate(model, train_x, train_y)
        val_loss, val_correct = evaluate(model, val_x, val_y)
        eval_s = time.perf_counter() - clock
        if not np.isfinite([online_ce, penalty_sum, train_loss, val_loss]).all():
            raise FloatingPointError(f"Pérdida no finita en {anchor.name}/{seed}, época {epoch}.")
        record["epochs"].append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "updates": (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE,
            "train_online_acc": online_correct / len(train_y), "train_online_ce": online_ce / len(train_y),
            "train_eval_acc": train_correct / len(train_y), "train_eval_ce": train_loss,
            "val_acc": val_correct / len(val_y), "val_correct": val_correct, "val_ce": val_loss,
            "penalty": penalty_sum / len(train_y), "update_s": update_s, "eval_s": eval_s,
        })
        clock = time.perf_counter()
        save_model(scratch / f"epoca_{epoch:02}.pt", model, anchor, seed, epoch, train_x.shape[1])
        write_json(history_file, record)
        io_s += time.perf_counter() - clock
        if epoch == 1 or epoch % 5 == 0:
            print(f"{anchor.name}/{seed} e{epoch:02}: val={val_correct / len(val_y):.4f} "
                  f"ajuste+eval={update_s + eval_s:.2f}s", flush=True)

    record.update(order_sha256=order_digest.hexdigest(), bn_instrumentation_s=diagnostic_s,
                  checkpoint_logging_s=io_s, wall_s=time.perf_counter() - started)
    write_json(history_file, record)
    print(f"Completa {anchor.name}/{seed}: {record['wall_s']:.1f}s.", flush=True)
    return record


def run_anchor(anchor, data, output):
    destination = output / "corridas" / anchor.name
    summary = destination / "resumen.json"
    if summary.exists():
        return read_json(summary)
    scratch = output / "temporales" / anchor.name
    records = [run_seed(anchor, seed, data, scratch / str(seed)) for seed in SEEDS]
    correct = np.array([[row["val_correct"] for row in record["epochs"]] for record in records])
    best_epoch = int(np.argmax(correct.sum(axis=0))) + 1
    for record in records:
        seed = record["seed"]
        per_seed = destination / str(seed)
        per_seed.mkdir(parents=True, exist_ok=True)
        source = scratch / str(seed)
        best_path = source / f"epoca_{best_epoch:02}.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"]
        final = torch.load(source / f"epoca_{EPOCHS:02}.pt", map_location="cpu", weights_only=True)["state_dict"]
        record["selected_epoch"] = best_epoch
        record["selected_weights"] = layer_statistics(best)
        record["last_weights"] = layer_statistics(final)
        record["bn_scales"] = {"selected": bn_scales(best), "last": bn_scales(final)}
        shutil.copyfile(best_path, per_seed / "checkpoint.pt")
        write_json(per_seed / "historial.json", record)

    result = {"anchor": anchor.name, "selected_epoch": best_epoch, "seeds": list(SEEDS),
              "val_acc": float(correct[:, best_epoch - 1].sum() / (len(SEEDS) * len(data[3])))}
    write_json(summary, result)
    for seed in SEEDS:
        per_seed = scratch / str(seed)
        for path in per_seed.glob("epoca_*.pt"):
            path.unlink()
        (per_seed / "corrida.json").unlink()
        per_seed.rmdir()
    scratch.rmdir()
    print(f"Ancla completa {anchor.name}: época común {best_epoch}.", flush=True)
    return result


def verify_structure():
    reference = MLP(8, ANCHORS[0], SEEDS[0])
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(17))
    for anchor in ANCHORS:
        model = MLP(8, anchor, SEEDS[0])
        assert all(torch.equal(a, b) for a, b in zip(reference.kernels, model.kernels))
        model.eval()
        assert model(x).shape == (4, NUM_CLASSES)
        if anchor.dropout:
            assert torch.equal(reference(x), model(x))
        if anchor.bn_momentum is not None:
            assert len([layer for layer in model.layers if isinstance(layer, nn.BatchNorm1d)]) == 2


def run_pilot(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify_structure()
    remaining_gib = shutil.disk_usage(output).free / 2**30
    if remaining_gib < 2:
        raise OSError("El piloto requiere 2 GiB libres para checkpoints temporales y definitivos.")
    manifest = {"anchors": [asdict(anchor) for anchor in ANCHORS], "seeds": list(SEEDS),
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "widths": list(WIDTHS), "lr": LR,
                "betas": list(BETAS), "threads": 2, "histogram_bins": HISTOGRAM_BINS.tolist(),
                "packages": {name: version(name) for name in ("torch", "numpy", "scikit-learn", "nltk")},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = output / "protocolo.json"
    if manifest_path.exists() and manifest != read_json(manifest_path):
        raise ValueError("El directorio de resultados corresponde a otro protocolo de ejecución.")
    write_json(manifest_path, manifest)
    print("Preparando el split interno del TP2 y TF-IDF...", flush=True)
    data, info = prepare_data(data_dir)
    data_path = output / "datos.json"
    if data_path.exists():
        previous = read_json(data_path)
        if any(info[key] != previous[key] for key in ("split_sha256", "vocabulary", "idf")):
            raise ValueError("La representación TF-IDF difiere de la corrida anterior.")
    else:
        write_json(data_path, info)
    print(f"Datos listos: {len(data[0])} train, {len(data[2])} validación, {info['input_dim']} términos.", flush=True)
    results = [run_anchor(anchor, data, output) for anchor in ANCHORS]
    if not (len(results) == 6 and all(len(row["seeds"]) == 3 for row in results)):
        raise RuntimeError("Faltan corridas del piloto.")
    write_json(output / "piloto_completo.json", results)
    print("Piloto completo: 18 corridas, 540 épocas.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Caché local sklearn/ y nltk_data/.")
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    run_pilot(options.data_dir.resolve(), options.output.resolve())

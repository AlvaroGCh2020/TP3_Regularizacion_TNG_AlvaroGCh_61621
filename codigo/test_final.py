"""Evalúa en test los checkpoints fijados en validación, sin actualizar sus pesos."""

from __future__ import annotations

import argparse
import hashlib
import time
from importlib.metadata import version
from pathlib import Path

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from busqueda_optuna import write_csv
from confirmacion import digest
from piloto import ANCHORS, BATCH_SIZE, MLP, NUM_CLASSES, SEEDS, TP2_SPLIT_HASH, Anchor, read_json, write_json


def verify_selection(project, selection_path):
    document = read_json(selection_path)
    families = document["families"]
    validation = read_json(selection_path.with_name("verificacion.json"))
    assert document["rule"] == "max_val_correct_sum_then_first_epoch_then_trial_id"
    assert document["seeds"] == list(SEEDS) and document["val_samples"] == 2263
    assert [family["family"] for family in families] == [anchor.name for anchor in ANCHORS]
    assert [family["trial_id"] for family in families] == validation["verified_selected_ids"]
    for family in families:
        assert 1 <= family["epoch"] <= 30 and family["eligible"]
        assert {item["seed"] for item in family["checkpoints"]} == set(SEEDS)
        for item in family["checkpoints"]:
            checkpoint_path = (project / item["path"]).resolve()
            if not checkpoint_path.is_relative_to(project):
                raise ValueError(f"Checkpoint fuera del proyecto: {item['path']}")
            if digest(checkpoint_path) != item["sha256"]:
                raise ValueError(f"Hash de checkpoint distinto al de validación: {item['path']}")
    assert sum(len(family["checkpoints"]) for family in families) == validation["verified_checkpoints"] == 18
    return families


def load_test(data_dir, frozen):
    nltk.data.path.insert(0, str(data_dir / "nltk_data"))
    dataset = fetch_20newsgroups(subset="test", data_home=str(data_dir / "sklearn"),
                                 download_if_missing=False, remove=("headers", "footers", "quotes"))
    y = np.asarray(dataset.target, dtype=np.int64)
    assert list(dataset.target_names) == frozen["classes"] and len(y) == 7532
    english_stopwords = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    documents = (" ".join(lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text.lower())
                          if token.isalpha() and token not in english_stopwords) for text in dataset.data)
    vectorizer = TfidfVectorizer(vocabulary=frozen["vocabulary"], ngram_range=(1, 1), min_df=2,
                                max_features=10_000, norm="l2", dtype=np.float32)
    vectorizer.idf_ = np.asarray(frozen["idf"], dtype=np.float32)
    representation = vectorizer.transform(documents)
    assert representation.shape == (len(y), frozen["input_dim"]) and representation.dtype == np.float32
    return torch.from_numpy(representation.toarray()), y


def save_array(path, array):
    scratch = path.with_suffix(".tmp")
    with scratch.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    scratch.replace(path)


@torch.inference_mode()
def predict(project, checkpoint, family, x):
    values = torch.load(project / checkpoint["path"], weights_only=True, map_location="cpu")
    assert (values["anchor"], values["seed"], values["epoch"], values["input_dim"]) == (
        family["regularization"], checkpoint["seed"], family["epoch"], x.shape[1])
    model = MLP(x.shape[1], Anchor(**family["regularization"]), checkpoint["seed"])
    model.load_state_dict(values["state_dict"], strict=True)
    model.eval()
    return torch.cat([model(batch).argmax(dim=1) for batch in x.split(BATCH_SIZE)]).numpy()


def summarize(output, records):
    reference = {r["seed"]: r for r in records if r["family"] == "referencia"}
    assert set(reference) == set(SEEDS)
    for row in records:
        baseline = reference[row["seed"]]
        row["delta_accuracy_pp"] = (row["correct"] - baseline["correct"]) * 100 / row["samples"]
        row["delta_f1_pp"] = (row["f1_macro"] - baseline["f1_macro"]) * 100

    summary = []
    for anchor in ANCHORS:
        group = [r for r in records if r["family"] == anchor.name]
        assert [row["seed"] for row in group] == list(SEEDS)
        row = {key: group[0][key] for key in ("family", "trial_id", "optimizer", "epoch")}
        for key in ("accuracy", "f1_macro", "delta_accuracy_pp", "delta_f1_pp"):
            values = [record[key] for record in group]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_sd"] = float(np.std(values, ddof=1))
        summary.append(row)
    write_csv(output / "test_por_semilla.csv", records)
    write_csv(output / "test_resumen.csv", summary)
    return summary


def run(data_dir, project, output):
    begin = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predicciones"
    predictions.mkdir(exist_ok=True)
    selection = project / "resultados" / "confirmacion" / "seleccion_validacion.json"
    vectorizer = project / "resultados" / "piloto" / "datos.json"
    families = verify_selection(project, selection)
    frozen = read_json(vectorizer)
    assert frozen["split_sha256"] == TP2_SPLIT_HASH and len(frozen["idf"]) == frozen["input_dim"]
    manifest = {"selection_sha256": digest(selection), "vectorizer_sha256": digest(vectorizer),
                "data_cache_sha256": digest(data_dir / "sklearn" / "20news-bydate_py3.pkz"),
                "runner_sha256": digest(Path(__file__)), "subset": "test", "seeds": list(SEEDS),
                "packages": {name: version(name) for name in ("torch", "numpy", "scikit-learn", "nltk")}}
    manifest_path = output / "protocolo.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("La evaluación guardada pertenece a otra selección o representación.")
    write_json(manifest_path, manifest)
    print("18 hashes de checkpoints verificados. Transformando test con TF-IDF de train interno.", flush=True)

    start = time.perf_counter()
    x, y = load_test(data_dir, frozen)
    prep_s = time.perf_counter() - start
    labels_path = output / "etiquetas.npy"
    if labels_path.exists() and not np.array_equal(np.load(labels_path, allow_pickle=False), y):
        raise ValueError("Cambió el orden de las etiquetas de test.")
    if not labels_path.exists():
        save_array(labels_path, y)
        write_json(output / "clases.json", frozen["classes"])

    records, newly_evaluated = [], 0
    for family in families:
        for checkpoint in sorted(family["checkpoints"], key=lambda item: item["seed"]):
            seed = checkpoint["seed"]
            path = predictions / f"trial_{family['trial_id']:03}_semilla_{seed}.npy"
            receipt_path = path.with_suffix(".json")
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert receipt["checkpoint_sha256"] == checkpoint["sha256"]
                assert receipt["predictions_sha256"] == digest(path)
                pred = np.load(path, allow_pickle=False)
                elapsed = receipt["inference_s"]
            else:
                started = time.perf_counter()
                pred = predict(project, checkpoint, family, x)
                elapsed = time.perf_counter() - started
                save_array(path, pred)
                write_json(receipt_path, {"checkpoint_sha256": checkpoint["sha256"], "predictions_sha256": digest(path),
                                          "inference_s": elapsed})
                newly_evaluated += 1
            assert pred.shape == y.shape and np.all((pred >= 0) & (pred < NUM_CLASSES))
            records.append({"family": family["family"], "trial_id": family["trial_id"],
                            "optimizer": family["optimizer"], "epoch": family["epoch"], "seed": seed,
                            "samples": len(y), "correct": int((pred == y).sum()),
                            "accuracy": float(accuracy_score(y, pred)),
                            "f1_macro": float(f1_score(y, pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)),
                            "inference_s": elapsed, "predictions": path.relative_to(output).as_posix()})
            print(f"{len(records):02}/18 {family['family']} #{family['trial_id']:02}/{seed}: "
                  f"accuracy={records[-1]['accuracy']:.4f}, F1 macro={records[-1]['f1_macro']:.4f}.", flush=True)

    assert len(records) == len(list(predictions.glob("*.npy"))) == 18
    summary = summarize(output, records)
    write_json(output / "test_completo.json", {"evaluations": 18, "newly_evaluated": newly_evaluated,
                                              "samples_per_checkpoint": len(y), "prep_s": prep_s,
                                              "wall_s": time.perf_counter() - begin, "training_updates": 0,
                                              "labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "summary": summary})
    print("Test completo: 18 checkpoints, 7532 ejemplos por checkpoint, cero actualizaciones de pesos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "project", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.project.resolve(), args.output.resolve())

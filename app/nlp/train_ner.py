import random
import logging
import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from collections import defaultdict

from cleaner import (
    ENTITY_LABELS,
    clean_dataturks_records,
    read_dataturks_records,
)

logger = logging.getLogger(__name__)



def convert_dataturks_to_examples(json_path: str, nlp) -> List[Example]:
    """
    Normalize Dataturks records and convert them to spaCy ``Example`` objects.
    """
    records = read_dataturks_records(json_path)
    input_count = len(records)
    records, cleaning_stats = clean_dataturks_records(records)
    examples: List[Example] = []
    skipped_alignment = 0

    for record_index, record in enumerate(records, start=1):
        text = record["content"]
        doc = nlp.make_doc(text)
        entities = []
        for annotation in record["annotation"]:
            point = annotation["points"][0]
            start = point["start"]
            end = point["end"] + 1
            span = doc.char_span(start, end, alignment_mode="strict")
            if span is None:
                skipped_alignment += 1
                continue
            entities.append((span.start_char, span.end_char, annotation["label"][0]))

        if entities:
            examples.append(Example.from_dict(doc, {"entities": entities}))

    logger.info(
        "Normalized %d resumes from %d records; merged %d duplicates, "
        "removed %d overlapping spans, corrected %d span texts",
        len(records),
        input_count,
        cleaning_stats["merged_duplicate_records"],
        cleaning_stats["removed_overlapping_spans"],
        cleaning_stats["corrected_span_text"],
    )
    logger.info(
        "Converted %d resumes (%d spans skipped due to spaCy token alignment)",
        len(examples),
        skipped_alignment,
    )
    return examples



def save_docbin(examples: List[Example], output_path: str):
    """Save ``Example`` reference docs as a spaCy v3 ``DocBin`` binary."""
    doc_bin = DocBin()
    for example in examples:
        doc_bin.add(example.reference)
    doc_bin.to_disk(output_path)
    logger.info("Saved %d examples → %s", len(examples), output_path)



def train_ner_model(
    json_path: str,
    output_dir: str,
    n_iter: int = 50,
    train_split: float = 0.85,
    dropout: float = 0.2,
    early_stop_patience: int = 8,
    seed: int = 42,
) -> Optional[spacy.language.Language]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1.")

    logger.info("=" * 60)
    logger.info("Starting NER model training (spaCy v3)")
    logger.info("Base model: en_core_web_sm (pre-trained English)")
    logger.info("=" * 60)


    try:

        base = spacy.load("en_core_web_sm", exclude=["ner", "parser", "senter", "tok2vec",
                                                       "tagger", "attribute_ruler", "lemmatizer"])


        nlp = spacy.blank("en")
        nlp.vocab = base.vocab
        logger.info("Using en_core_web_sm vocabulary")
    except OSError:
        logger.info("en_core_web_sm not found, using blank vocab. Run: python -m spacy download en_core_web_sm")
        nlp = spacy.blank("en")


    ner = nlp.add_pipe("ner", last=True)


    for label in ENTITY_LABELS:
        ner.add_label(label)

    logger.info("Entity labels: %s", ENTITY_LABELS)


    all_examples = convert_dataturks_to_examples(json_path, nlp)
    if not all_examples:
        logger.error("No training examples found — check the JSON path.")
        return None


    if len(all_examples) < 2:
        raise ValueError("At least two valid resumes are required for train/dev evaluation.")
    random_generator = random.Random(seed)
    random_generator.shuffle(all_examples)
    split_idx = min(
        len(all_examples) - 1,
        max(1, int(len(all_examples) * train_split)),
    )
    train_examples = all_examples[:split_idx]
    dev_examples = all_examples[split_idx:]

    logger.info("Training set: %d | Dev set: %d", len(train_examples), len(dev_examples))


    os.makedirs(output_dir, exist_ok=True)
    data_dir = Path(output_dir).parent / "data"
    data_dir.mkdir(exist_ok=True)
    if train_examples:
        save_docbin(train_examples, str(data_dir / "train.spacy"))
    if dev_examples:
        save_docbin(dev_examples, str(data_dir / "dev.spacy"))


    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()

        best_f = -1.0
        best_iteration = 0
        no_improve = 0

        for iteration in range(n_iter):
            random_generator.shuffle(train_examples)
            losses = {}

            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)

            if dev_examples:
                overall, per_entity = _evaluate_detailed(nlp, dev_examples)
                f_score = overall["f"]

                logger.info(
                    "Iter %02d/%02d | Loss: %7.1f | P: %.4f | R: %.4f | F: %.4f",
                    iteration + 1, n_iter, losses.get("ner", 0),
                    overall["p"], overall["r"], f_score,
                )


                if (iteration + 1) % 5 == 0:
                    logger.info("  Per-entity F-scores:")
                    for label, scores in sorted(per_entity.items()):
                        logger.info("    %-25s P:%.3f  R:%.3f  F:%.3f",
                                    label, scores["p"], scores["r"], scores["f"])


                if f_score > best_f:
                    best_f = f_score
                    best_iteration = iteration + 1
                    no_improve = 0
                    nlp.to_disk(output_dir)
                    logger.info("  → New best model saved (F=%.4f)", best_f)
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        logger.info(
                            "Early stopping at iteration %d "
                            "(no improvement for %d iterations)",
                            iteration + 1, early_stop_patience
                        )
                        break
            else:
                logger.info("Iter %02d/%02d | Loss: %.4f",
                            iteration + 1, n_iter, losses.get("ner", 0))

    if not dev_examples:
        nlp.to_disk(output_dir)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Best overall F-score: %.4f", best_f)
    logger.info("Model saved to: %s", output_dir)
    logger.info("=" * 60)


    best_nlp = spacy.load(output_dir)
    overall, per_entity = _evaluate_detailed(best_nlp, dev_examples)
    metrics = {
        "dataset": str(Path(json_path).resolve()),
        "seed": seed,
        "train_split": train_split,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "iterations_requested": n_iter,
        "best_iteration": best_iteration,
        "overall": overall,
        "per_entity": per_entity,
    }
    metrics_path = Path(output_dir) / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation metrics saved to: %s", metrics_path)
    logger.info("Final per-entity scores (best model):")
    for label, scores in sorted(per_entity.items()):
        logger.info("  %-25s P:%.3f  R:%.3f  F:%.3f",
                    label, scores["p"], scores["r"], scores["f"])

    return best_nlp



def _evaluate_detailed(
    nlp, dev_examples: List[Example]
) -> Tuple[dict, Dict[str, dict]]:

    total_correct = 0
    total_predicted = 0
    total_gold = 0

    entity_correct: Dict[str, int] = defaultdict(int)
    entity_predicted: Dict[str, int] = defaultdict(int)
    entity_gold: Dict[str, int] = defaultdict(int)

    for example in dev_examples:
        doc = nlp(example.reference.text)

        predicted = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold = {(e.start_char, e.end_char, e.label_) for e in example.reference.ents}

        correct = predicted & gold

        total_correct += len(correct)
        total_predicted += len(predicted)
        total_gold += len(gold)

        for _, _, label in predicted:
            entity_predicted[label] += 1
        for _, _, label in gold:
            entity_gold[label] += 1
        for _, _, label in correct:
            entity_correct[label] += 1

    # Overall scores
    p = total_correct / total_predicted if total_predicted > 0 else 0.0
    r = total_correct / total_gold if total_gold > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    overall = {"p": p, "r": r, "f": f}

    per_entity: Dict[str, dict] = {}
    all_labels = set(ENTITY_LABELS) | set(entity_predicted) | set(entity_gold)
    for label in all_labels:
        ep = entity_correct[label] / entity_predicted[label] if entity_predicted[label] > 0 else 0.0
        er = entity_correct[label] / entity_gold[label] if entity_gold[label] > 0 else 0.0
        ef = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
        per_entity[label] = {"p": ep, "r": er, "f": ef}

    return overall, per_entity


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Train a custom spaCy v3 NER model on the Dataturks resume dataset"
    )
    project_root = Path(__file__).resolve().parents[2]
    default_data = project_root / "Entity_Recognition_in_Resumes.json"
    default_output = Path(__file__).resolve().parent / "models" / "spacy_ner"

    ap.add_argument("--data", type=str, default=str(default_data),
                    help="Path to Dataturks JSON file")
    ap.add_argument("--output", type=str, default=str(default_output),
                    help="Output directory for trained model")
    ap.add_argument("--iter", type=int, default=50,
                    help="Max training iterations (default: 50)")
    ap.add_argument("--split", type=float, default=0.85,
                    help="Train/dev split ratio")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout rate")
    ap.add_argument("--patience", type=int, default=8,
                    help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible train/dev split")

    args = ap.parse_args()

    train_ner_model(
        json_path=args.data,
        output_dir=args.output,
        n_iter=args.iter,
        train_split=args.split,
        dropout=args.dropout,
        early_stop_patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

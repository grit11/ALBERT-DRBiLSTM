import argparse
import csv
import datetime
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
#from transformers import AlbertTokenizer
from model.model_albert_bilstm import AlbertBiLSTMClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_timestamp(timestamp_text):
    try:
        return float(datetime.datetime.strptime(timestamp_text, "%Y-%m-%d").toordinal())
    except ValueError:
        return 0.0


def read_data(file_path, num=None):
    texts = []
    labels = []
    metadata = []

    with open(file_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行格式错误，应包含 6 列，实际为 {len(parts)} 列。"
                )

            review, label, max_similarity, suspicion_score, timestamp, sentiment = parts
            texts.append(review)
            labels.append(int(label))
            metadata.append(
                [
                    float(max_similarity),
                    float(suspicion_score),
                    parse_timestamp(timestamp),
                    float(sentiment),
                ]
            )

            if num is not None and len(texts) >= num:
                break

    return texts, labels, np.asarray(metadata, dtype=np.float32)


class MetadataScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, values):
        if self.mean is None or self.std is None:
            raise ValueError("MetadataScaler 尚未 fit。")
        return (values - self.mean) / self.std

    def to_dict(self):
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, payload):
        scaler = cls()
        scaler.mean = np.asarray(payload["mean"], dtype=np.float32)
        scaler.std = np.asarray(payload["std"], dtype=np.float32)
        return scaler


class TextDataset(Dataset):
    def __init__(self, texts, labels, metadata, tfidf_matrix, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.metadata = metadata
        self.tfidf_matrix = tfidf_matrix
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "metadata": torch.tensor(self.metadata[index], dtype=torch.float32),
            "tfidf": torch.tensor(self.tfidf_matrix[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.labels)


def build_dataloader(texts, labels, metadata, tfidf_matrix, tokenizer, max_len, batch_size, shuffle):
    dataset = TextDataset(texts, labels, metadata, tfidf_matrix, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_best_metrics(labels, probs):
    thresholds = np.linspace(0.05, 0.95, 19)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))
        f1_scores.append(f1_score(labels, preds, zero_division=0))

    best_index = int(np.argmax(f1_scores))
    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "best_threshold": float(thresholds[best_index]),
        "best_precision": float(precisions[best_index]),
        "best_recall": float(recalls[best_index]),
        "best_f1": float(f1_scores[best_index]),
    }


def make_json_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: make_json_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]
    return value


def round_nested(value, digits=4):
    if isinstance(value, dict):
        return {key: round_nested(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def dump_json_rounded(payload, file_path, digits=4):
    serializable = make_json_serializable(payload)
    rounded = round_nested(serializable, digits=digits)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rounded, f, ensure_ascii=False, indent=2)


def append_csv_row(csv_path, fieldnames, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    losses = []
    labels = []
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            metadata = batch["metadata"].to(device)
            tfidf = batch["tfidf"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata_features=metadata,
                tfidf_features=tfidf,
                labels=batch_labels,
                threshold=threshold,
            )
            losses.append(outputs["loss"].item())
            labels.extend(batch_labels.cpu().numpy().tolist())
            probs.extend(outputs["probs"].cpu().numpy().tolist())

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "report": classification_report(labels, preds, labels=[0, 1], zero_division=0, digits=4),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = None

    return metrics, labels, probs


def plot_metrics(labels, probs, output_dir, split_name):
    best_metrics = compute_best_metrics(labels, probs)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(best_metrics["thresholds"], best_metrics["precisions"], label="Precision", marker="o")
    plt.plot(best_metrics["thresholds"], best_metrics["recalls"], label="Recall", marker="o")
    plt.plot(best_metrics["thresholds"], best_metrics["f1_scores"], label="F1", marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{split_name} Threshold Sweep")
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    figure_path = output_dir / f"{split_name.lower()}_curves.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    best_metrics["roc_auc"] = float(roc_auc)
    best_metrics["figure_path"] = str(figure_path)
    return best_metrics


def save_experiment_config(args, tfidf_dim, output_dir):
    config = vars(args).copy()
    config["actual_tfidf_dim"] = tfidf_dim
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def save_preprocessing_artifacts(output_dir, tfidf_vectorizer, metadata_scaler):
    with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(output_dir / "metadata_scaler.json", "w", encoding="utf-8") as f:
        json.dump(metadata_scaler.to_dict(), f, ensure_ascii=False, indent=2)


def main(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    history_csv_path = output_dir / "train_history.csv"
    summary_csv_path = project_root / "experiment_summary.csv"

    train_texts, train_labels, train_metadata = read_data(args.train_path, args.max_train_samples)
    dev_texts, dev_labels, dev_metadata = read_data(args.dev_path, args.max_dev_samples)
    test_texts, test_labels, test_metadata = read_data(args.test_path, args.max_test_samples)

    metadata_scaler = MetadataScaler().fit(train_metadata)
    train_metadata = metadata_scaler.transform(train_metadata)
    dev_metadata = metadata_scaler.transform(dev_metadata)
    test_metadata = metadata_scaler.transform(test_metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    #tokenizer = AlbertTokenizer.from_pretrained(args.pretrained_model_name)
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=args.tfidf_max_features,
        min_df=1,
    )
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    dev_tfidf = tfidf_vectorizer.transform(dev_texts).toarray().astype(np.float32)
    test_tfidf = tfidf_vectorizer.transform(test_texts).toarray().astype(np.float32)
    tfidf_dim = train_tfidf.shape[1]

    save_experiment_config(args, tfidf_dim, output_dir)
    save_preprocessing_artifacts(output_dir, tfidf_vectorizer, metadata_scaler)

    train_dataloader = build_dataloader(
        train_texts, train_labels, train_metadata, train_tfidf, tokenizer, args.max_len, args.batch_size, True
    )
    dev_dataloader = build_dataloader(
        dev_texts, dev_labels, dev_metadata, dev_tfidf, tokenizer, args.max_len, args.batch_size, False
    )
    test_dataloader = build_dataloader(
        test_texts, test_labels, test_metadata, test_tfidf, tokenizer, args.max_len, args.batch_size, False
    )

    positive_count = max(sum(train_labels), 1)
    negative_count = max(len(train_labels) - positive_count, 1)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AlbertBiLSTMClassifier(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        tfidf_dim=tfidf_dim,
        metadata_dim=4,
        dropout_rate=args.classifier_dropout,
        lstm_dropout=args.lstm_dropout,
        pos_weight=pos_weight.to(device),
        pretrained_model_name=args.pretrained_model_name,
        fusion_mode=args.fusion_mode,
        use_dilated_conv=args.use_dilated_conv,
        use_residual=args.use_residual_connection,
        use_channel_attention=args.use_channel_attention,
        use_metadata_features=args.use_metadata_features,
        use_tfidf_features=args.use_tfidf_features,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    best_dev_f1 = -1.0
    best_epoch = 0
    trigger_times = 0
    checkpoint_path = output_dir / "best_model.pth"
    best_dev_metrics_snapshot = None

    train_history_fields = [
        "run_name",
        "epoch",
        "lr",
        "train_loss",
        "train_precision",
        "train_recall",
        "train_f1",
        "dev_loss",
        "dev_precision",
        "dev_recall",
        "dev_f1",
        "dev_auc",
        "is_best",
        "patience_counter",
        "seed",
        "fusion_mode",
        "tfidf_dim",
        "batch_size",
        "max_len",
    ]

    for epoch_index in range(args.epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch_index + 1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            metadata = batch["metadata"].to(device)
            tfidf = batch["tfidf"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata_features=metadata,
                tfidf_features=tfidf,
                labels=labels,
            )
            loss = outputs["loss"]
            if torch.isnan(loss):
                raise ValueError("训练过程中出现 NaN loss，请检查数据或学习率设置。")

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs["preds"].detach().cpu().numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())

        scheduler.step()

        train_precision = precision_score(train_targets, train_preds, zero_division=0)
        train_recall = recall_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        print(
            f"Epoch {epoch_index + 1}: "
            f"train_loss={np.mean(train_losses):.4f}, "
            f"train_precision={train_precision:.4f}, "
            f"train_recall={train_recall:.4f}, "
            f"train_f1={train_f1:.4f}"
        )

        dev_metrics, dev_labels_arr, dev_probs = evaluate(model, dev_dataloader, device)
        print(
            f"Epoch {epoch_index + 1}: "
            f"dev_loss={dev_metrics['loss']:.4f}, "
            f"dev_precision={dev_metrics['precision']:.4f}, "
            f"dev_recall={dev_metrics['recall']:.4f}, "
            f"dev_f1={dev_metrics['f1']:.4f}"
        )
        print(dev_metrics["report"])

        is_best = dev_metrics["f1"] > best_dev_f1
        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_epoch = epoch_index + 1
            trigger_times = 0
            torch.save(model.state_dict(), checkpoint_path)

            best_dev_curve = plot_metrics(dev_labels_arr, dev_probs, output_dir, "dev")
            best_dev_metrics_snapshot = {
                "epoch": best_epoch,
                "metrics": dev_metrics,
                "threshold_search": best_dev_curve,
            }
            dump_json_rounded(best_dev_metrics_snapshot, output_dir / "best_dev_metrics.json")
        else:
            trigger_times += 1

        append_csv_row(
            history_csv_path,
            train_history_fields,
            {
                "run_name": output_dir.name,
                "epoch": epoch_index + 1,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 3 if optimizer.param_groups[0]["lr"] >= 1 else 6),
                "train_loss": round(float(np.mean(train_losses)), 4),
                "train_precision": round(float(train_precision), 4),
                "train_recall": round(float(train_recall), 4),
                "train_f1": round(float(train_f1), 4),
                "dev_loss": round(float(dev_metrics["loss"]), 4),
                "dev_precision": round(float(dev_metrics["precision"]), 4),
                "dev_recall": round(float(dev_metrics["recall"]), 4),
                "dev_f1": round(float(dev_metrics["f1"]), 4),
                "dev_auc": round(float(dev_metrics["auc"]), 4) if dev_metrics["auc"] is not None else "",
                "is_best": int(is_best),
                "patience_counter": trigger_times,
                "seed": args.seed,
                "fusion_mode": args.fusion_mode,
                "tfidf_dim": tfidf_dim,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
            },
        )

        if trigger_times >= args.patience:
            print("早停触发。")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics, test_labels_arr, test_probs = evaluate(model, test_dataloader, device)
    test_curve = plot_metrics(test_labels_arr, test_probs, output_dir, "test")

    dump_json_rounded(
        {
            "best_epoch": best_epoch,
            "metrics": test_metrics,
            "threshold_search": test_curve,
        },
        output_dir / "test_metrics.json",
    )

    summary_fields = [
        "run_name",
        "output_dir",
        "seed",
        "best_epoch",
        "best_dev_precision",
        "best_dev_recall",
        "best_dev_f1",
        "best_dev_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "lr",
        "batch_size",
        "max_len",
        "epochs",
        "patience",
        "weight_decay",
        "tfidf_max_features",
        "actual_tfidf_dim",
        "fusion_mode",
    ]
    append_csv_row(
        summary_csv_path,
        summary_fields,
        {
            "run_name": output_dir.name,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_dev_precision": round(float(best_dev_metrics_snapshot["metrics"]["precision"]), 4)
            if best_dev_metrics_snapshot
            else "",
            "best_dev_recall": round(float(best_dev_metrics_snapshot["metrics"]["recall"]), 4)
            if best_dev_metrics_snapshot
            else "",
            "best_dev_f1": round(float(best_dev_f1), 4),
            "best_dev_auc": round(float(best_dev_metrics_snapshot["metrics"]["auc"]), 4)
            if best_dev_metrics_snapshot and best_dev_metrics_snapshot["metrics"]["auc"] is not None
            else "",
            "test_precision": round(float(test_metrics["precision"]), 4),
            "test_recall": round(float(test_metrics["recall"]), 4),
            "test_f1": round(float(test_metrics["f1"]), 4),
            "test_auc": round(float(test_metrics["auc"]), 4) if test_metrics["auc"] is not None else "",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "tfidf_max_features": args.tfidf_max_features,
            "actual_tfidf_dim": tfidf_dim,
            "fusion_mode": args.fusion_mode,
        },
    )

    print("Test report:")
    print(test_metrics["report"])
    print(f"最佳开发集 F1: {best_dev_f1:.4f}")
    print(f"测试集 F1: {test_metrics['f1']:.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="ALBERT-DRBiLSTM fake review detection trainer")
    parser.add_argument("--train-path", default="data/1.txt")
    parser.add_argument("--dev-path", default="data/3.txt")
    parser.add_argument("--test-path", default="data/2.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pretrained-model-name", default="albert-base-chinese")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--lr-step-size", type=int, default=3)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--tfidf-max-features", type=int, default=1000)
    parser.add_argument("--lstm-dropout", type=float, default=0.6)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--fusion-mode", choices=["sf", "c", "ws", "aensf"], default="aensf")
    parser.add_argument("--use-dilated-conv", action="store_true", default=True)
    parser.add_argument("--no-use-dilated-conv", dest="use_dilated_conv", action="store_false")
    parser.add_argument("--use-residual-connection", action="store_true", default=True)
    parser.add_argument("--no-use-residual-connection", dest="use_residual_connection", action="store_false")
    parser.add_argument("--use-channel-attention", action="store_true", default=True)
    parser.add_argument("--no-use-channel-attention", dest="use_channel_attention", action="store_false")
    parser.add_argument("--use-metadata-features", action="store_true", default=True)
    parser.add_argument("--no-use-metadata-features", dest="use_metadata_features", action="store_false")
    parser.add_argument("--use-tfidf-features", action="store_true", default=True)
    parser.add_argument("--no-use-tfidf-features", dest="use_tfidf_features", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())




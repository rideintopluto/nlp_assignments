r"""四组 CBOW 模型的统一相似词评估入口，放在项目根目录。

运行：.\.venv\Scripts\python.exe evaluate.py
导出全部模型的词向量：.\.venv\Scripts\python.exe evaluate.py --export-vectors
只比较中文：.\.venv\Scripts\python.exe evaluate.py --lang zh
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


EXPECTED_GROUPS = ((10, 100), (20, 100), (30, 100), (30, 200))
CONTROL_KEYS = (
    "window_size", "batch_size", "learning_rate", "optimizer", "loss_function",
    "seed", "sample_count", "sentence_count", "pytorch_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一比较四组 CBOW 模型的相似词结果")
    parser.add_argument("--lang", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="默认 cuda；可显式指定 cpu 或 cuda:0")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--queries-file", type=Path, default=Path("evaluation_queries.json"))
    parser.add_argument("--export-vectors", action="store_true", help="同时导出每组模型的完整词向量")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须大于等于 1。")
    return args


def project_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_queries(path: Path, languages: tuple[str, ...]) -> dict[str, list[str]]:
    with path.open(encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("查询词文件必须是含 zh/en 键的 JSON 对象。")
    for language in languages:
        words = payload.get(language)
        if not isinstance(words, list) or not words:
            raise ValueError(f"查询词文件缺少 {language} 的非空词列表。")
        if not all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words):
            raise ValueError(f"{language} 的查询词应为无空白符的独立词。")
        if len(set(words)) != len(words):
            raise ValueError(f"{language} 的查询词存在重复。")
    return {language: payload[language] for language in languages}


def discover_checkpoints(root: Path, languages: tuple[str, ...]) -> list[tuple[Path, dict]]:
    """根据每个模型旁的 config.json 识别语言与实验组；重复运行全部保留。"""
    if not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{root}")
    selected = []
    for path in sorted(root.rglob("model.pt")):
        config_path = path.with_name("config.json")
        if not config_path.is_file():
            raise ValueError(f"模型旁缺少 config.json：{path}")
        with config_path.open(encoding="utf-8-sig") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"配置不是 JSON 对象：{config_path}")
        language = config.get("language")
        group = (config.get("epochs"), config.get("embedding_dim"))
        if language in languages and group in EXPECTED_GROUPS:
            selected.append((path, config))

    missing = []
    for language in languages:
        for epochs, dim in EXPECTED_GROUPS:
            matches = [p for p, c in selected if (c.get("language"), c.get("epochs"), c.get("embedding_dim")) == (language, epochs, dim)]
            if not matches:
                missing.append(f"{language}: epoch={epochs}, dim={dim}")
            elif len(matches) > 1:
                print(f"提示：{language} epoch={epochs}, dim={dim} 有 {len(matches)} 次运行，将分别保留结果。")
    if missing:
        raise ValueError("未找到以下实验组的完整 model.pt 与 config.json：\n" + "\n".join(missing))
    return sorted(selected, key=lambda item: (
        item[1]["language"], item[1]["embedding_dim"], item[1]["epochs"], str(item[0])
    ))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    # UTF-8 BOM 方便在 Windows Excel 中直接打开中文结果。
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_text(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    languages = ("zh", "en") if args.lang == "both" else (args.lang,)
    queries = read_queries(project_path(root, args.queries_file), languages)
    selected = discover_checkpoints(project_path(root, args.outputs_root), languages)

    import torch
    from cbow.evaluation import export_word_vectors, load_word_vectors, most_similar

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用之前训练成功的解释器；需要 CPU 时显式传入 --device cpu。")
    print(f"评估设备：{device}；找到 {len(selected)} 个模型。", flush=True)

    # 全部模型共用同一组查询词；语言分别比较，不混合中英文向量空间。
    models = []
    references = {}
    for path, companion_config in selected:
        model = load_word_vectors(path, device)
        for key in ("language", "epochs", "embedding_dim", *CONTROL_KEYS):
            if companion_config.get(key) != model.training_config.get(key):
                raise ValueError(f"模型内部配置与旁边的 config.json 不一致（{key}）：{path}")
        if model.language in references:
            reference = references[model.language]
            if model.word_to_id != reference.word_to_id:
                raise ValueError(f"同一语言的模型词表不同，不能直接作为当前受控对比：{path}")
            differences = [key for key in CONTROL_KEYS if model.training_config.get(key) != reference.training_config.get(key)]
            if differences:
                raise ValueError(f"同一语言的模型还有其他训练条件不同：{differences}。请核对 {path}")
        else:
            references[model.language] = model
        models.append(model)

    # 可选的词频帮助解释低频词表现；它来自当前语料，不是评估得分。
    frequencies = {}
    for language in languages:
        corpus_path = root / "data" / f"{language}.txt"
        frequencies[language] = Counter(corpus_path.read_text(encoding="utf-8-sig").split()) if corpus_path.is_file() else None

    summary_rows = []
    neighbor_rows = []
    query_rows = []
    report = [
        "# CBOW 相似词对比", "",
        "本报告比较固定查询词的余弦近邻，属于定性分析，不包含准确率、独立验证损失或测试集指标。",
        "余弦相似度不是概率；不同模型的相似度数值不可直接当作质量排名。",
        "查询词自身和零向量不参与近邻排名；保留停用词、标点、数字和所有其他有效词。",
        "词频来自运行评估时的 data/zh.txt、data/en.txt；缺少语料时留空。", "",
    ]
    results = {}
    for model_number, model in enumerate(models, start=1):
        model_id = f"{model.language}_e{model.epochs}_d{model.dim}_{model_number:02d}"
        present = 0
        for query in queries[model.language]:
            frequency = frequencies[model.language][query] if frequencies[model.language] is not None else ""
            if query not in model.word_to_id:
                status, neighbors = "oov", []
            elif not model.nonzero_rows[model.word_to_id[query]].item():
                status, neighbors = "zero_vector", []
            else:
                status, neighbors = "ok", most_similar(model, query, args.top_k)
                present += 1
            results[(model_id, query)] = (status, neighbors)
            query_rows.append({"model_id": model_id, "language": model.language, "query": query, "status": status, "corpus_frequency": frequency})
            for rank, (neighbor, similarity) in enumerate(neighbors, start=1):
                neighbor_rows.append({
                    "model_id": model_id, "language": model.language,
                    "epochs": model.epochs, "embedding_dim": model.dim,
                    "query": query, "rank": rank, "neighbor": neighbor,
                    "cosine_similarity": similarity,
                })
        summary_rows.append({
            "model_id": model_id, "language": model.language, "epochs": model.epochs,
            "embedding_dim": model.dim, "vocab_size": len(model.id_to_word),
            "final_epoch_train_loss": model.loss_history[-1],
            "query_count": len(queries[model.language]), "valid_queries": present,
            "run_id": model.run_id, "checkpoint_path": str(model.checkpoint_path),
        })
        print(f"{model_id} | 有效查询 {present}/{len(queries[model.language])} | {model.run_id}", flush=True)

    for language in languages:
        report.extend([f"## {'中文' if language == 'zh' else '英文'}", ""])
        language_summaries = [row for row in summary_rows if row["language"] == language]
        for query in queries[language]:
            frequency = frequencies[language][query] if frequencies[language] is not None else "未知"
            report.extend([
                f"### 查询词：{md_text(query)}（当前语料词频：{frequency}）", "",
                "| 模型 | 相似词（括号内为余弦相似度） |", "|---|---|",
            ])
            for row in language_summaries:
                status, neighbors = results[(row["model_id"], query)]
                if status == "oov":
                    description = "OOV：查询词不在该模型词表中"
                elif status == "zero_vector":
                    description = "查询词为零向量，无法计算余弦相似度"
                else:
                    description = "；".join(f"{md_text(word)} ({score:.4f})" for word, score in neighbors) or "无可用候选词"
                report.append(f"| {row['model_id']} | {description} |")
            report.append("")

    now = datetime.now().astimezone()
    output_dir = root / "evaluations" / now.strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "model_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output_dir / "neighbors.csv", neighbor_rows, [
        "model_id", "language", "epochs", "embedding_dim", "query", "rank", "neighbor", "cosine_similarity",
    ])
    write_csv(output_dir / "query_status.csv", query_rows, list(query_rows[0]))
    (output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    settings = {
        "created_at": now.isoformat(timespec="seconds"), "device": str(device),
        "top_k": args.top_k, "queries": queries, "embedding_source": "embedding.weight",
        "export_vectors": args.export_vectors,
        "checkpoints": [str(model.checkpoint_path) for model in models],
        "note": "相似词定性分析；训练语料未作为独立验证集使用。",
    }
    (output_dir / "evaluation_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.export_vectors:
        for row, model in zip(summary_rows, models):
            export_word_vectors(model, output_dir / "vectors" / f"{row['model_id']}.txt")
    print(f"评估完成：{output_dir}")
    print("先查看 comparison.md；模型与实验组的对应关系见 model_summary.csv。")


if __name__ == "__main__":
    main()

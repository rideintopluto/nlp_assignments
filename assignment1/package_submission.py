"""Create the code-only ZIP required by the course."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "artifacts",
    "data",
}
EXCLUDED_SUFFIXES = {".ckpt", ".pdf", ".pt", ".pth", ".pyc", ".zip"}


def safe_component(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("学号和姓名不能为空")
    return re.sub(r'[<>:"/\\|?*]', "_", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成课程要求的代码提交压缩包")
    parser.add_argument("--student-id", required=True, help="学号")
    parser.add_argument("--student-name", required=True, help="姓名")
    parser.add_argument("--assignment", default="1", help="作业编号")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    student_id = safe_component(args.student_id)
    student_name = safe_component(args.student_name)
    output_name = f"{student_id}_{student_name}_作业{safe_component(args.assignment)}.zip"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (args.output_dir / output_name).resolve()

    included = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if should_include(path, root) and path.resolve() != output_path:
                arcname = Path(root.name) / path.relative_to(root)
                archive.write(path, arcname)
                included.append(arcname.as_posix())

    if not included:
        raise RuntimeError("没有找到可打包的代码文件")
    print(f"代码提交包已生成：{output_path}")
    print(f"共包含 {len(included)} 个文件；未包含数据集、模型权重和训练输出。")


if __name__ == "__main__":
    main()

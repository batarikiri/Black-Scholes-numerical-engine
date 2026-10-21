"""
run_all.py
==========
Запускает генерацию всех 4 фигур для статьи.

Использование:
    python scripts/run_all.py
"""

import sys
from pathlib import Path

# Добавляем src/ в sys.path, чтобы работали absolute import'ы
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import generate_all_figures  # noqa: E402


if __name__ == "__main__":
    outdir = ROOT / "figures"
    generate_all_figures(outdir=str(outdir))
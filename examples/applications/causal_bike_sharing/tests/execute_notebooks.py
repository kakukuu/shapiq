"""Execute notebooks in fresh kernels and save outputs to the runtime directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tests" / ".runtime"
for variable, name in (
    ("JUPYTER_CONFIG_DIR", "jupyter"),
    ("JUPYTER_RUNTIME_DIR", "runtime"),
    ("IPYTHONDIR", "ipython"),
    ("MPLCONFIGDIR", "matplotlib"),
):
    directory = RUNTIME / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


class VisibleClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream" and (
            cell_index == 1 or "ensure_targeted_cases(" in cell.source
        ):
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def execute(arms):
    report_path = RUNTIME / "notebook_execution.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for arm in arms:
        path = ROOT / f"experiment_{arm}.ipynb"
        nb = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]

        notebook_cell_count = len(nb.cells)

        def completed(
            cell, cell_index, *, notebook_name=path.name, cell_count=notebook_cell_count, **kwargs
        ):
            if cell.cell_type == "code":
                print(f"{notebook_name}: cell {cell_index + 1}/{cell_count} complete", flush=True)

        client = VisibleClient(
            nb,
            km=manager,
            timeout=7200,
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_executed=completed,
        )
        try:
            with client.setup_kernel(cleanup_kc=True):
                client.execute()
        except Exception:
            nbformat.write(nb, RUNTIME / f"{path.stem}_failed.ipynb")
            raise
        nbformat.validate(nb)
        errors = [
            out
            for c in nb.cells
            if c.cell_type == "code"
            for out in c.outputs
            if out.output_type == "error"
        ]
        assert not errors
        nb.metadata.setdefault("experiment_workflow", {})["outputs"] = "executed_verified"
        output_path = RUNTIME / path.name
        nbformat.write(nb, output_path)
        report[arm] = {
            "code_cells": sum(c.cell_type == "code" for c in nb.cells),
            "errors": len(errors),
            "kernel": sys.executable,
        }
        (RUNTIME / "notebook_execution.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"SAVED {output_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="*", default=["a", "b1", "b2", "b3", "lag"])
    execute(parser.parse_args().arms)

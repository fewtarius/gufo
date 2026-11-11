"""bench.json loading and table expansion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "gufo-model-bench/1"
TARGETS = ("gufo", "reference")


@dataclass(frozen=True)
class TableSpec:
    id: str
    base: str
    variant: str | None
    spec: dict[str, Any]

    @property
    def kind(self) -> str:
        # A table may carry a device suffix (loading-handheld), so match the
        # family by prefix rather than by an exact base name.
        for family in ("single", "multi", "loading", "memory", "image-encoder"):
            if self.base == family or self.base.startswith(family + "-"):
                return family
        return self.base

    @property
    def speculative(self) -> bool:
        return bool(self.spec.get("speculative"))

    def workload_tables(self) -> list[TableSpec]:
        """Keep each workload's measurements under its existing artifact identity."""
        if self.kind not in ("single", "multi"):
            return []
        common = {k: v for k, v in self.spec.items() if k != "workloads"}
        return [
            TableSpec(f"{base}-{self.variant}" if self.variant else base, base,
                      self.variant, {**common, **workload})
            for base, workload in self.spec.get("workloads", {}).items()
        ]


@dataclass
class BenchConfig:
    model: str
    root: Path
    data: dict[str, Any]
    files: dict[str, dict[str, Path]] = field(default_factory=dict)
    artifacts_override: Path | None = None

    @property
    def model_dir(self) -> Path:
        return self.root / "docs" / "internal" / "models" / self.model

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_override or (self.model_dir / "artifacts")

    @property
    def benchmarks_path(self) -> Path:
        return self.model_dir / "BENCHMARKS.md"

    @property
    def category(self) -> str:
        return self.data["category"]

    @property
    def variants(self) -> dict[str, dict[str, Any]]:
        return self.data["variants"]

    @property
    def single_variant(self) -> bool:
        return list(self.variants) == ["default"]

    @property
    def reference_name(self) -> str:
        reference = self.data.get("reference")
        return reference["name"] if reference else None

    @property
    def gufo_only(self) -> bool:
        """A page with no reference entry measures Gufo alone.

        Rendering then drops the reference and gain columns instead of filling
        them with TODO, so a model we do not compare stays readable.
        """
        return self.reference_name is None

    @property
    def speculative(self) -> dict[str, Any]:
        return self.data.get("speculative") or {}

    @property
    def reference_speculative(self) -> list[str] | None:
        """Reference server arguments for the equivalent speculative mode, if any."""
        reference = self.speculative.get("reference")
        if isinstance(reference, dict) and reference.get("args"):
            return list(reference["args"])
        return None

    def substitute(self, args: list[str], variant: str | None) -> list[str]:
        """Replace `{role}` placeholders with the variant's file paths."""
        out = []
        for part in args:
            if part.startswith("{") and part.endswith("}"):
                part = str(self.file(part[1:-1], variant))
            out.append(part)
        return out

    def tables(self) -> list[TableSpec]:
        result: list[TableSpec] = []
        for base, spec in self.data["tables"].items():
            if spec.get("per_variant"):
                for variant in self.variants:
                    result.append(TableSpec(f"{base}-{variant}", base, variant, spec))
            else:
                result.append(TableSpec(base, base, None, spec))
        return result

    def table(self, table_id: str) -> TableSpec:
        for table in self.tables():
            if table.id == table_id:
                return table
        raise KeyError(f"unknown table {table_id!r} for {self.model}")

    def variant_label(self, variant: str) -> str:
        return self.variants[variant]["label"]

    def _resolve_variant(self, variant: str | None) -> str:
        """A table without a variant measures the model's only variant.

        A config may name its single variant anything (q4 for a page that once
        carried two), so falling back to the literal "default" raises for it.
        """
        if variant:
            return variant
        if len(self.variants) == 1:
            return next(iter(self.variants))
        return "default"

    def file(self, role: str, variant: str | None) -> Path:
        variant = self._resolve_variant(variant)
        by_variant = self.files.get(role, {})
        path = by_variant.get(variant) or by_variant.get("*")
        if path is None:
            description = self.data["files"].get(role, role)
            raise SystemExit(
                f"missing --{role} for variant {variant!r}: {description}"
            )
        return path

    def require_files(self, variant: str | None) -> None:
        variant = self._resolve_variant(variant)
        for role in self.variants[variant].get("requires", []):
            self.file(role, variant)


def load_config(root: Path, model: str, path: Path | None = None) -> BenchConfig:
    path = path or root / "docs" / "internal" / "models" / model / "artifacts" / "bench.json"
    if not path.exists():
        raise SystemExit(f"no bench.json for {model}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: expected schema {SCHEMA}")
    return BenchConfig(model=model, root=root, data=data)


def parse_file_args(config: BenchConfig, values: dict[str, list[str]]) -> None:
    """Map `--gguf q4=/path` / `--gguf /path` arguments onto file roles."""
    for role, entries in values.items():
        for entry in entries:
            variant, separator, raw = entry.partition("=")
            if not separator:
                variant, raw = ("default" if config.single_variant else "*"), entry
            path = Path(raw).expanduser()
            if not path.exists():
                raise SystemExit(f"--{role} {entry}: {path} does not exist")
            config.files.setdefault(role, {})[variant] = path.absolute()

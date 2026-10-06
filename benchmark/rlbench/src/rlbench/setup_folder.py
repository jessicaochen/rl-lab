"""Load and validate a setup folder, and render its manifests.

A setup folder follows the convention documented in benchmark/README.md:

    my-setup/
      setup/          # k8s manifests applied before the job
      job.yaml        # template for the training Job (the completion signal)
      config/         # optional run configs passed with --config
      hooks/          # optional: pre-setup.sh, post-run.sh
      provision.sh    # optional one-time cluster prep, never run by rlbench
      features/<n>/   # optional opt-in features (``--feature <n>``), see FeatureFolder

Manifests may reference ``${VARS}``. Values come from the process environment
plus rlbench built-ins (RUN_ID, RUN_NAME). Unresolved variables are an error:
setup folders must stay portable, so anything cluster-specific has to arrive
via the environment and end up recorded only in the run folder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RUN_LABEL = "rlbench/run"


class SetupFolderError(Exception):
    pass


@dataclass
class SetupFolder:
    root: Path
    setup_manifests: list[Path] = field(default_factory=list)
    job_manifest: Path | None = None

    @classmethod
    def load(cls, root: str | Path) -> "SetupFolder":
        root = Path(root).resolve()
        if not root.is_dir():
            raise SetupFolderError(f"setup folder not found: {root}")
        setup_dir = root / "setup"
        job = root / "job.yaml"
        if not setup_dir.is_dir():
            raise SetupFolderError(f"missing required '{setup_dir}' directory")
        if not job.is_file():
            raise SetupFolderError(f"missing required '{job}'")
        manifests = sorted(
            p for p in setup_dir.iterdir() if p.suffix in (".yaml", ".yml")
        )
        if not manifests:
            raise SetupFolderError(f"no manifests in {setup_dir}")
        return cls(root=root, setup_manifests=manifests, job_manifest=job)

    def scrape_targets(self) -> list[tuple[str, str, str, str]]:
        """Optional ``scrape-targets.txt``: ``pods <namespace> <label-selector> <port> <path>`` per line."""
        return _parse_scrape_targets(self.root / "scrape-targets.txt")

    def hook(self, name: str) -> Path | None:
        p = self.root / "hooks" / name
        return p if p.is_file() else None

    def feature(self, name: str) -> "FeatureFolder":
        """Load ``features/<name>/`` (``--feature <name>``); unknown names fail before the cluster is touched."""
        return FeatureFolder.load(self.root, name)


@dataclass
class FeatureFolder:
    """An opt-in feature of a setup: ``<setup>/features/<name>/``.

        features/<name>/
          README.md            # what it is, requirements, how to verify it was active
          vars.env             # KEY=VALUE render-time variables (env < vars.env < --var)
          setup/*.yaml         # optional extra manifests, applied after the setup's
          config/*             # optional files copied (flat) into the run config dir;
                               #   convention: feature-<name>.sh fragment sourced by the job
          hooks/{pre-setup,post-run}.sh   # optional, run after the setup's hooks
          scrape-targets.txt   # optional, merged into the setup's targets

    Features flip ``${VAR:-default}`` knobs of the base manifests and may add
    objects; they never patch base objects. rlbench also sets the built-ins
    ``RLBENCH_FEATURES`` (comma-joined) and ``FEATURE_<NAME>=1`` per feature.
    """

    name: str
    root: Path
    vars: dict[str, str] = field(default_factory=dict)
    setup_manifests: list[Path] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)

    _NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

    @classmethod
    def load(cls, setup_root: Path, name: str) -> "FeatureFolder":
        if not cls._NAME_RE.match(name):
            raise SetupFolderError(f"feature name must be lowercase [a-z0-9-]: {name!r}")
        root = Path(setup_root) / "features" / name
        if not root.is_dir():
            available = sorted(p.name for p in (Path(setup_root) / "features").glob("*") if p.is_dir())
            raise SetupFolderError(
                f"unknown feature {name!r}: {root} not found"
                + (f" (available: {', '.join(available)})" if available else "")
            )
        setup_dir = root / "setup"
        config_dir = root / "config"
        return cls(
            name=name,
            root=root,
            vars=_parse_vars_env(root / "vars.env"),
            setup_manifests=sorted(p for p in setup_dir.iterdir() if p.suffix in (".yaml", ".yml"))
            if setup_dir.is_dir() else [],
            config_files=sorted(p for p in config_dir.iterdir() if p.is_file()) if config_dir.is_dir() else [],
        )

    @property
    def env_name(self) -> str:
        """``FEATURE_<NAME>`` built-in: upper-case, dashes to underscores."""
        return "FEATURE_" + self.name.upper().replace("-", "_")

    def scrape_targets(self) -> list[tuple[str, str, str, str]]:
        return _parse_scrape_targets(self.root / "scrape-targets.txt")

    def hook(self, name: str) -> Path | None:
        p = self.root / "hooks" / name
        return p if p.is_file() else None


def feature_builtins(features: list["FeatureFolder"]) -> dict[str, str]:
    """Render-time built-ins describing the enabled features (always set, so
    ``${RLBENCH_FEATURES}`` renders to "" when no feature is enabled)."""
    out = {"RLBENCH_FEATURES": ",".join(f.name for f in features)}
    for f in features:
        out[f.env_name] = "1"
    return out


def feature_variables(features: list["FeatureFolder"]) -> dict[str, str]:
    """Merged ``vars.env`` of all enabled features (later features win)."""
    out: dict[str, str] = {}
    for f in features:
        out.update(f.vars)
    return out


def _parse_vars_env(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` per line, ``#`` comments and blank lines ignored; values are
    taken verbatim (no quoting rules, no shell expansion)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for n, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", stripped):
            raise SetupFolderError(f"{path}:{n}: expected KEY=VALUE, got {line!r}")
        k, v = stripped.split("=", 1)
        out[k] = v
    return out


def _parse_scrape_targets(f: Path) -> list[tuple[str, str, str, str]]:
    targets = []
    if f.is_file():
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] != "pods" or len(parts) != 5:
                raise SetupFolderError(f"{f}: expected 'pods <namespace> <selector> <port> <path>', got: {line!r}")
            targets.append((parts[1], parts[2], parts[3], parts[4]))
    return targets


# Render-time variables are BRACED ONLY: ``${NAME}`` or ``${NAME:-default}``.
# Bare ``$name`` is never touched, so shell scripts used as run configs keep
# their own variables. Strict mode (manifests) errors on an unknown ``${NAME}``
# without a default and honours ``$$`` -> ``$`` escapes; lenient mode (run
# configs) leaves unknowns intact and leaves ``$$`` alone (shell PID syntax).
_VAR_RE = re.compile(
    r"\$(?:(?P<escaped>\$)|\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\})"
)


def _render(path: Path, variables: dict[str, str], strict: bool) -> str:
    def repl(m: re.Match) -> str:
        if m.group("escaped"):
            return "$" if strict else m.group(0)
        name = m.group("name")
        if name in variables:
            return str(variables[name])
        if m.group("default") is not None:
            return m.group("default")
        if strict:
            raise SetupFolderError(
                f"{path}: unresolved variable ${{{name}}} — pass it via the environment or --var"
            )
        return m.group(0)

    return _VAR_RE.sub(repl, path.read_text())


def render_text(path: Path, variables: dict[str, str]) -> str:
    """Strict rendering (manifests): unknown ``${NAME}`` without a default is an error."""
    return _render(path, variables, strict=True)


def render_text_lenient(path: Path, variables: dict[str, str]) -> str:
    """Lenient rendering (run configs): known vars and ``${NAME:-default}`` are
    substituted and baked into the run folder copy; anything else is untouched."""
    return _render(path, variables, strict=False)


def render_manifest(path: Path, variables: dict[str, str]) -> list[dict]:
    """Substitute ${VARS} and return the parsed YAML documents (label-injected)."""
    docs = [d for d in yaml.safe_load_all(render_text(path, variables)) if d]
    for doc in docs:
        _inject_run_label(doc, variables["RUN_ID"])
    return docs


def _inject_run_label(doc: dict, run_id: str) -> None:
    """Label the object (and its pod template, if any) so every resource and
    pod belonging to a run is discoverable and deletable by label alone."""
    meta = doc.setdefault("metadata", {})
    meta.setdefault("labels", {})[RUN_LABEL] = run_id
    template = doc.get("spec", {}).get("template")
    if isinstance(template, dict):
        tmeta = template.setdefault("metadata", {})
        if isinstance(tmeta, dict):
            tmeta.setdefault("labels", {})[RUN_LABEL] = run_id


def namespace_of(docs: list[dict]) -> str:
    """The run's namespace: an explicit Namespace object wins, else the first
    namespaced object's metadata.namespace."""
    for d in docs:
        if d.get("kind") == "Namespace":
            return d["metadata"]["name"]
    for d in docs:
        ns = d.get("metadata", {}).get("namespace")
        if ns:
            return ns
    raise SetupFolderError(
        "could not determine namespace: add a Namespace manifest or set metadata.namespace"
    )

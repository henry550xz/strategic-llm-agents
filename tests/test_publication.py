from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_local_links_exist():
    missing = []
    for document in ROOT.rglob("*.md"):
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            if "://" in target or target.startswith("#"): continue
            clean = target.split("#", 1)[0]
            if clean and not (document.parent / clean).resolve().exists(): missing.append((document, target))
    assert not missing


def test_no_large_public_files():
    assert not [(path, path.stat().st_size) for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 1_000_000]


def test_no_raw_dataset_or_model_artifacts():
    forbidden = {".safetensors", ".pt", ".pth", ".jsonl"}
    assert not [path for path in ROOT.rglob("*") if path.suffix in forbidden]


def test_release_metadata_has_no_placeholders():
    text = "\n".join(path.read_text() for path in [ROOT / "README.md", ROOT / "CITATION.cff"])
    assert "OWNER" not in text and "PLACEHOLDER" not in text and "TODO" not in text


def test_readme_identity_and_release_verdict():
    readme = (ROOT / "README.md").read_text()
    assert readme.startswith("# Strategic LLM Agents\n\n### Learning, simulation, and evaluation for strategic multi-agent policies")
    assert "PUBLIC_RELEASE_READY" in (ROOT / "PUBLIC_RELEASE_AUDIT.md").read_text()

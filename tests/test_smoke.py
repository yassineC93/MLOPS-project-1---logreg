import subprocess
import sys
from pathlib import Path


def test_main_creates_run_artifacts(tmp_path: Path):
    """
    Smoke test:
    - lance main.py dans un dossier temporaire
    - vérifie que les artefacts attendus sont créés
    """
    # On exécute depuis un répertoire temporaire => test reproductible, pas de pollution.
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    output_dir = tmp_path / "ci_outputs"

    cmd = [
        sys.executable,
        str(main_py),
        "--output-dir",
        str(output_dir),
        "--test-size",
        "0.2",
        "--seed",
        "42",
    ]

    res = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    assert res.returncode == 0, f"main.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # output_dir doit exister
    assert output_dir.exists(), "output dir not created"

    # un run_id (sous-dossier) doit exister
    run_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
    assert run_dirs, "no run folder created"

    run_dir = sorted(run_dirs)[-1]

    expected = ["model.joblib", "metrics.json", "predictions.csv", "run.log", "run_config.json"]
    missing = [f for f in expected if not (run_dir / f).exists()]
    assert not missing, f"missing files: {missing}"

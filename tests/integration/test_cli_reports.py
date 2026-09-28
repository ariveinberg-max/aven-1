from __future__ import annotations

from pathlib import Path

import pytest

from neurolayer.cli import main

pytest.importorskip("pyriemann")


def _config(tmp_path: Path, decoder: str) -> Path:
    path = tmp_path / f"{decoder}.yaml"
    path.write_text(
        f"""name: cli-{decoder.replace("_", "-")}
purpose: benchmark
datasets: [synthetic_mi]
synthetic: {{n_subjects: 4, n_trials_per_class: 30}}
decoder: {{name: {decoder}}}
protocol: {{ks: [0, 10], n_unlabeled: 5, n_folds: null, n_bootstrap: 100}}
"""
    )
    return path


def test_report_probe_and_control_commands(
    catalog_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--catalog", str(catalog_dir), "--output", str(tmp_path / "runs")]
    for decoder in ("chance", "logvar_logreg"):
        assert main([*base, "run", str(_config(tmp_path, decoder))]) == 0
    runs = sorted(str(p) for p in (tmp_path / "runs").iterdir())
    out = tmp_path / "report.md"
    assert (
        main(
            [
                *base,
                "report",
                "compare",
                *runs,
                "--reference",
                runs[0],
                "--out",
                str(out),
                "--plot",
                str(tmp_path / "cec.svg"),
            ]
        )
        == 0
    )
    assert "Paired vs reference" in out.read_text()
    assert (tmp_path / "cec.svg").exists()

    assert (
        main([*base, "probe", str(_config(tmp_path, "chance")), "--encoders", "log_variance"]) == 0
    )
    assert "| log_variance | subject |" in capsys.readouterr().out

    code = main([*base, "control", str(_config(tmp_path, "logvar_logreg")), "--seeds", "3"])
    assert code in (0, 1)
    assert "label-shuffle control" in capsys.readouterr().out


def test_gate0_refuses_unverified_license(
    unverified_catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--catalog", str(unverified_catalog), "gate0", "physionet_mi"]) == 2
    assert "license gate" in capsys.readouterr().err

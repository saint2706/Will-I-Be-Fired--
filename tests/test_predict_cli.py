import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import predict_cli  # noqa: E402


def test_main_processes_multiple_records(tmp_path, monkeypatch, capsys):
    records = [
        {
            "Department": "IT",
            "Salary": "60000",
            "DateofHire": "2020-01-01",
        },
        {
            "Department": "HR",
            "Salary": "70000",
            "DateofHire": "2019-06-15",
        },
    ]
    employee_file = tmp_path / "employees.json"
    employee_file.write_text(json.dumps(records))

    args = argparse.Namespace(
        employee_json=employee_file,
        model=Path("dummy.joblib"),
        horizons=None,
        calibrate=False,
        policy_config=Path("dummy_policy.yaml"),
    )

    monkeypatch.setattr(predict_cli, "parse_args", lambda: args)
    monkeypatch.setattr(predict_cli, "configure_logging", lambda: None)
    monkeypatch.setattr(predict_cli, "set_global_seed", lambda x: None)

    seen_records = []

    def fake_predict(record, *, horizons, model_path):
        seen_records.append(record)
        assert tuple(horizons) == (1.0, 2.0, 5.0)
        assert model_path == args.model
        return [
            predict_cli.TenureRisk(
                tenure_years=1.0,
                termination_probability=0.25,
                confidence=0.75,
            )
        ]

    monkeypatch.setattr(predict_cli, "predict_tenure_risk", fake_predict)

    predict_cli.main()

    assert len(seen_records) == 2
    assert seen_records[0]["Salary"] == 60000.0
    assert seen_records[1]["Salary"] == 70000.0

    output = capsys.readouterr().out
    assert "=== Employee record 1 ===" in output
    assert "=== Employee record 2 ===" in output
    assert output.count("--- Predicted Termination Risk ---") == 2

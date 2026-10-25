"""Train the demonstration project and export local reports/models."""
import json
from pathlib import Path
import joblib
from src.data import make_demo_data
from src.evaluate import evaluate_models


def main():
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    metrics, models, _, calibration_residuals = evaluate_models(make_demo_data())
    metrics.to_csv(output_dir / "demo_metrics.csv", index=False)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, output_dir / filename)

    report = {
        "dataset": "synthetic demo data; not real-world evidence",
        "rows": 1200,
        "target": "response_time_minutes",
        "selection_rule": "lowest validation MAE",
        "split": metrics.attrs.get(
            "split_proportions",
            "approximately 60% train, 10% validation, 10% calibration, 20% test",
        ),
        "split_method": metrics.attrs.get("split_method", "random"),
        "selected_model": metrics.attrs.get("selected_model"),
        "metrics": metrics.to_dict(orient="records"),
        "calibration_residuals": calibration_residuals,
    }
    (output_dir / "demo_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nSaved local artifacts. Metrics apply only to synthetic data.")


if __name__ == "__main__":
    main()

"""Train demo models and save local artifacts."""
import json
from pathlib import Path
import joblib
from src.data import make_demo_data
from src.evaluate import evaluate_models


def main():
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    metrics, models, _ = evaluate_models(make_demo_data())
    metrics.to_csv(out / "demo_metrics.csv", index=False)
    for name, model in models.items():
        joblib.dump(model, out / (name.lower().replace(" ", "_") + ".joblib"))
    report = {"dataset": "synthetic demo data; not real-world evidence", "rows": 1200,
              "target": "response_time_minutes", "metrics": metrics.to_dict(orient="records")}
    (out / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print("\nSaved local artifacts. Metrics apply only to synthetic data.")


if __name__ == "__main__":
    main()

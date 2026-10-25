# Data

The built-in app generates synthetic demonstration data in memory. No private or personal data is committed.

For CSV import testing, try the [Datanemics Customer Support Tickets CSV](https://datanemics.com/datasets/data/support-tickets.csv). Its publisher describes it as a synthetic 4,000-row dataset under CC0 public-domain terms. The file includes `first_response_min`, `opened_at`, `queue`, `priority`, and `channel`; the app can normalize those columns and use a chronological split.

This sample demonstrates the CSV workflow but cannot establish real-world accuracy. Blank first-response values indicate tickets that were never answered in this sample and are excluded from regression; that introduces survivorship bias. Never treat unanswered tickets as zero-minute responses.

Local CSV and Parquet files are ignored by Git. Before using institutional data, verify permissions, privacy, target definition, time zones, missing-ticket handling, feature availability at prediction time, and data quality.

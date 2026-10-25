# Data

No personal or private data is committed. The default app creates synthetic demonstration data in memory.

For CSV import testing, the app supports the [Datanemics Customer Support Tickets CSV](https://datanemics.com/datasets/data/support-tickets.csv), which its publisher describes as synthetic and CC0. It contains `first_response_min`, `opened_at`, `queue`, `priority`, and `channel`. The app maps these to its supported schema and uses a chronological split if timestamps are valid.

This sample is only a demonstration dataset—not evidence of real-world performance. Rows with blank first-response time are excluded from regression; if they represent never-answered tickets, this introduces survivorship bias. Never treat unanswered tickets as zero-minute responses. Check the publisher's data description, license, and limitations before using any source.

Local CSV and Parquet files are ignored by Git. Before using institutional data, verify permissions, privacy, target definition, missing-ticket handling, time zones, and data quality.

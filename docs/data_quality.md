# Data-quality review

The labeled file contains 48,000 rows from January through October 2025. The final validation file contains 12,000 rows from November through December 2025.

## Findings

| Check | Labeled data | Final validation | Treatment |
| --- | ---: | ---: | --- |
| Duplicate rows | 0 | 0 | No action |
| Duplicate `load_id` | 0 | 0 | Enforced as an error |
| Missing weight | 300 | 165 | Missing flag plus native CatBoost handling |
| Negative weight | 292 | 145 | Absolute value plus anomaly flag |
| Missing market index | 374 | 249 | Missing flag plus native CatBoost handling |
| Unseen cities | - | 8 | Geography and numeric coordinates support generalization |
| Unseen lanes | - | 736 | Lane is supplementary; distance, geography, and other features remain available |

The target is right-skewed and includes rare values above $10,000. Those rows are retained because the brief does not identify them as errors. Modeling the logarithm of the target reduces their leverage and improves typical-load error.

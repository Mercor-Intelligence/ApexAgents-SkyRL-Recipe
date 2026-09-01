| **Run** | **Loss agg** | **IS variant** | **Prompt nudge** | **OLF** | **ALP (assistant/environment penalty coefficient)** | **Mean Reward (%)** | **Pass@1(%)** | **Avg turns** | **Avg tokens** | **Tokens / turn** | **Tool call success (%)** | **# ctx len exceed** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **4. prompt_mean** | prompt_mean | / | / | / | / | 32.54 ± 1.19 | 16.74 ± 0.79 | 32.12 ± 0.64 | 95,792 ± 581 | 837.8 ± 8.2 | 95.88 ± 0.02 | 1 / 2 / 3 |
| **10. DPPO + prompt_mean + nudge** | prompt_mean | DPPO | True | / | / | 31.81 ± 0.81 | 16.11 ± 0.43 | 35.99 ± 0.44 | 101,441 ± 790 | 775.3 ± 8.4 | 97.74 ± 0.14 | 0 / 3 / 4 |
| **5. Nudge** | / | / | True | / | / | 31.64 ± 0.68 | 15.69 ± 1.18 | 38.31 ± 0.64 | 96,297 ± 1,144 | 685.8 ± 5.8 | 97.31 ± 0.18 | 0 / 1 / 2 |
| **13. DPPO + prompt_mean + nudge + ALP 0.25, 0.1** | prompt_mean | DPPO | True | / | 0.25 / 0.1 | 31.60 ± 1.54 | 15.07 ± 1.27 | 44.96 ± 1.16 | 110,082 ± 1,029 | 761.2 ± 5.3 | 98.41 ± 0.11 | 2 / 3 / 1 |
| **12. DPPO + prompt_mean + nudge + ALP 0.1, 0.1** | prompt_mean | DPPO | True | / | 0.1 / 0.1 | 30.93 ± 0.22 | 16.18 ± 1.18 | 43.77 ± 0.30 | 95,546 ± 796 | 549.8 ± 6.5 | 97.78 ± 0.17 | 1 / 0 / 0 |
| **9. prompt_mean + ALP** | prompt_mean | / | / | / | 0.25 / 0.25 | 30.85 ± 0.99 | 16.11 ± 0.79 | 35.80 ± 1.10 | 88,535 ± 1,509 | 620.3 ± 8.4 | 97.90 ± 0.58 | 1 / 0 / 0 |
| **2. ResetKV** | / | / | / | / | / | 30.57 ± 1.75 | 14.86 ± 1.22 | 25.19 ± 0.51 | 86,522 ± 156 | 852.4 ± 13.9 | 98.40 ± 0.03 | 1 / 0 / 0 |
| **11. DPPO + prompt_mean + nudge + OLF** | prompt_mean | DPPO | True | True | / | 30.30 ± 0.50 | 14.72 ± 0.43 | 32.25 ± 0.74 | 103,034 ± 1,541 | 794.1 ± 22.7 | 94.60 ± 0.27 | 0 / 2 / 2 |
| **7. DPPO + ALP + OLF** | / | DPPO | / | True | 0.25 / 0.25 | 30.00 ± 1.20 | 15.49 ± 0.87 | 25.77 ± 0.46 | 95,683 ± 663 | 844.3 ± 10.6 | 96.34 ± 0.14 | 2 / 1 / 0 |
| **3. DPPO (ep1, gs88)** | / | DPPO | / | / | / | 29.03 ± 0.45 | 13.96 ± 0.21 | 32.40 ± 0.66 | 87,124 ± 876 | 587.5 ± 13.1 | 97.16 ± 0.22 | 0 / 1 / 0 |
| **1. Baseline (ep1, gs87)** | token_mean | rollout_is | False | False | False | 28.69 ± 0.80 | 13.12 ± 1.63 | 21.21 ± 0.27 | 77,462 ± 873 | 834.0 ± 14.7 | 98.37 ± 0.32 | 0 / 0 / 0 |
| **6. DPPO + ALP** | / | DPPO | / | / | 0.25 / 0.25 | 28.12 ± 0.75 | 13.26 ± 0.48 | 24.99 ± 0.70 | 83,829 ± 391 | 765.2 ± 18.6 | 97.53 ± 0.06 | 0 / 0 / 1 |
| **8. DPPO + ALP + Prompt nudge** | / | DPPO | True | / | 0.25 / 0.25 | 27.24 ± 0.59 | 13.26 ± 0.52 | 25.20 ± 0.15 | 83,999 ± 1,427 | 813.8 ± 4.2 | 97.51 ± 0.18 | 0 / 1 / 1 |
| **Untrained model** | — | — | — | — | — | 22.74 ± 1.04 | 9.10 ± 1.26 | 16.80 ± 0.13 | 87,703 ± 1,233 | 882.9 ± 19.0 | 96.60 ± 0.28 | 4 / 4 / 3 |

Table: Algorithm ablations on Qwen3.6-35B-A3B. Every run is evaluated at its epoch-1 checkpoint, 3 passes over the held-out 480 tasks (mean ± std); rows ranked by Mean Reward. "/" means same as baseline (token_mean, GLM-5 loss, no nudge / OLF / ALP). ResetKV = baseline + clearing the prefix cache on every weight sync. "# ctx len exceed" lists, for each of the 3 passes, how many rollouts overran the context window.

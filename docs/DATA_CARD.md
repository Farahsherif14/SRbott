# Data Card: service-request classifier (category + urgency)

Draft of 2026-11-01. Items marked **[FILL IN]** need a team member's input. Do not delete the "Known limitations" section.

## 1. Purpose

Training and evaluating a classifier for resident maintenance requests sent to a Telegram bot (English, formal Arabic, Egyptian Arabic).

- Categories: `plumbing`, `electricity`, `ac`, `carpentry`, `pest control` (written `pest_control` in the labeling guide and test files), `general`.
- Urgency: `low`, `medium`, `high` in training data. The labeling guide also has `emergency`.
- Success criterion: category macro-F1 of at least 0.85 on a held-out, hand-written test set.
- Only public or synthetic data is used. No real resident messages.

## 2. Sources

| # | Dataset | What it is | Rows | Used for |
|---|---|---|---|---|
| 1 | NYC 311 Service Requests from 2020 to Present | Public government data, NYC Open Data (dataset `erm2-nwe9`) | 995 rows in `train_pool.csv` | Category mapping only. **Not** in the final model (see below) |
| 2 | Synthetic English requests | Written by a local LLM (`qwen3:4b` through Ollama) with `generate_synthetic.py` | 1,343 rows after cleaning | Training |
| 3 | Arabic and Egyptian check set | Written by Claude (Anthropic's AI assistant), `arabic_check.csv` | 300 rows (150 groups) | Measurement and tuning only, never as labelled training data |
| 4 | Hand-written test set | Written by our team | **[FILL IN]** rows | Final evaluation only |

### 2.1 NYC 311

- Source: NYC Open Data, "311 Service Requests from 2020 to Present", https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data
- Our export: `311_Service_Requests_from_2020_to_Present_20261022.csv` (exported 2026-10-22). We read the first 2,000,000 rows (`--nrows`).
- The portal describes the data as public and as containing no personally identifying information about the requester. The catalog listing shows no license, and the dataset is covered by the NYC Open Data Terms of Use, not Data.gov's. **[FILL IN: confirm the terms with your supervisor and record the date checked.]**
- The 2020-present dataset renamed `Complaint Type` to `Problem` and `Descriptor` to `Problem Detail`. `clean_311.py` handles both names.
- The 311 data has only categorical fields, so the "text" is templated from `Problem`, `Problem Detail` and `Additional Details`. It is not natural language.
- `mapping.py` maps the fields to our categories and gives a rule-based (weak) urgency. `HEAT/HOT WATER` is split by `Additional Details`: "NO HEAT" becomes `ac`, "NO HOT WATER" becomes `plumbing`.
- Because the text is templated, validation scores on it are meaningless (about 0.96 to 1.0). The final model is trained **without** these rows (`category_model.py train`, flag `--with-311` to include them).

### 2.2 Synthetic English

- Generator: `ml/generate_synthetic.py`, model `qwen3:4b` through Ollama, temperature 1.0, thinking off, 5 requests per call.
- Prompt: per-category scenario lists, an urgency calibration guide, varied unit references from `mock_units.csv` (960 invented units, no real addresses), and an instruction to describe some problems without naming the appliance.
- Arabic and Egyptian output from this model was unusable (garbled), so English only was generated.

| File | Rows as used | Notes |
|---|---|---|
| `synthetic_pool_en_3.csv` | 810 | Generated 860. See cleaning below |
| `synthetic_pool_en_4.csv` | 419 | **[FILL IN: which prompt version generated it]** |
| `synthetic_pool_en_test2.csv` | 114 | Generated with the corrected prompt |

Cleaning applied to `synthetic_pool_en_3.csv`:
1. Generator drift: 23 AC requests labeled as other categories were relabeled to `ac`, and 6 broken rows were dropped (rows that named the wrong category in the text, ambiguous, or too short).
2. The repeated template "AC not working in unit X" was trimmed to 25 rows.
3. `audit_labels.py` (grouped cross-validation): 27 rows confirmed as wrong labels were removed, then 2 more found by a keyword search. 6 flagged rows were kept after review.
4. Backups of each change are kept as `.bak`, `.bak2`, `.bak3` files.

The prompt later gained a keyword guard that rejects non-AC rows that mention AC and rows where the category name leaks into the text. Multi-issue requests were removed from the training prompt and belong in the test set.

### 2.3 Arabic and Egyptian check set

- File: `data/processed/arabic_check.csv`. 25 situations per category, each written twice with a shared `group_id`: formal Arabic (`source = claude_seed_msa`) and Egyptian colloquial (`claude_seed_eg`). Language column is `ar` for both.
- Written by Claude, not machine-translated from the English data, and not taken from real people. Categories follow `LABELING_GUIDE.md`.
- It is **never** used as labelled training data. Its text (without labels) gives the Arabic mean embedding used for centering. Its labels were used to compare model setups and to tune the review threshold, so its scores are optimistic.
- **[FILL IN: a native Egyptian speaker on the team should review the Egyptian rows. Record the reviewer and date.]**

### 2.4 Hand-written test set

- Written by team members in `data/test/raw_en.txt`, `raw_msa.txt`, `raw_eg.txt` (`category | urgency | message`), built into `testset_*.csv` by `ml/build_testset.py`.
- Target: 200 to 300 messages across English, formal Arabic and Egyptian, at least 6 per category in each file.
- Mix: about 20% without the trade or appliance named, about 10% with two problems, about 10% vague, with typos and different lengths.
- Writers and counts: **[FILL IN: names, number of messages each]**
- Rules: written in the writer's own words, not copied from generated data or from the check set. Kept out of all training and tuning. `category_model.py test` checks for exact overlaps with training data.

## 3. Preprocessing

`text_utils.clean_text` masks phone numbers and email addresses, normalizes Arabic characters and removes zero-width characters. It is applied to training text and at prediction time. Language is detected with `detect_lang` and decides which language mean is subtracted.

## 4. Labels

`LABELING_GUIDE.md` (v3) defines the categories, urgency levels and tricky cases (e.g. AC named as the cause goes to `ac`, hot water goes to `plumbing`, shared doors and gates go to `general`).

Mismatch to be aware of: the synthetic training urgency follows the generator's rules (AC not working is medium, door that will not lock is medium) while the guide is stricter (AC dead is high, door that cannot be locked is emergency). The guide also has four urgency levels, training has three. For evaluation, `emergency` is counted as `high`.

## 5. Model (for context)

- Embeddings: `qwen3-embedding:4b` through Ollama, with the instruction prefix "Instruct: Classify a resident maintenance request into a service category".
- Per-language mean centering (English mean, Arabic mean), then logistic regression (C = 10, balanced classes). Category and urgency each have their own classifier.
- Trained on English only. Arabic and Egyptian are handled through the multilingual embeddings.
- Review flag: predictions with confidence below 0.75 go to human review (provisional, tuned on the check set).

## 6. Results so far

| Measure | Result | Trust |
|---|---|---|
| Category macro-F1, Arabic/Egyptian check set | 0.870 overall, 0.940 formal Arabic, 0.799 Egyptian | Optimistic: this set was used to choose the setup |
| Automatic answers at threshold 0.75 | about 96% correct, about 20% flagged | Provisional |
| Category macro-F1, English validation | about 0.99 | Not meaningful (templated and near-duplicate text) |
| Category macro-F1, hand-written test set | **[FILL IN]** | The result that counts |
| Urgency macro-F1 | **[FILL IN]** | Noisy labels; see section 4 |
| ML vs LLM zero-shot (accuracy, latency, cost) | **[FILL IN from `data/test/comparison_summary.md`]** | |

Earlier setups on the check set (macro-F1 overall): bge-m3 0.757, qwen3-embedding 0.6b 0.628, qwen3-embedding 4b 0.816, 4b with instruction prefix 0.887 (0.870 after final text cleaning).

## 7. Known limitations

- The English training text comes from a small 4B model. It is repetitive, and it often names the appliance ("AC") in AC requests. Requests that only describe symptoms ("the room is boiling") are the weakest case.
- Egyptian Arabic is the weakest language. The model never sees Arabic during training.
- The check set is small (150 rows per style), written by one author, and was used for tuning. Differences of a few points between runs are within noise.
- Urgency labels are noisy (cross-validated accuracy of about 0.79 on synthetic data) and follow different rules from the labeling guide.
- NYC 311 data describes New York complaints and heating, not an Egyptian compound with cooling needs.
- The model was not tested on real resident messages.

## 8. Privacy and ethics

No real personal data is used. NYC 311 data is public and has no requester identity. Units are invented. Phone numbers and emails are masked in cleaning. Synthetic and check-set texts are generated by AI models and written by team members, not collected from people.

## 9. How to reproduce

```powershell
python ml\clean_311.py
python ml\generate_synthetic.py --provider ollama --model qwen3:4b --languages en --groups-per-category 150 --per-call 5 --seed 13 --output data\processed\synthetic_pool_en_N.csv
python ml\audit_labels.py
python ml\category_model.py train
python -u ml\build_testset.py
python ml\category_model.py test        # once, at the gate
python ml\compare_llm.py --price-in X --price-out Y
```

## 10. Change log

- 2026-11-01: first draft.

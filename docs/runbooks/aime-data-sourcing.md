# AIME data sourcing runbook

`AIMEMultiYearDataset` reads three JSON files from a directory passed
as `data_dir`. It does not fetch from the network. This runbook
explains how to populate that directory.

## Expected files and schema

`<data_dir>/aime_{year}.json` for `year` in `{2024, 2025, 2026}`:

```json
{
  "year": 2026,
  "questions": [
    {"id": "2026-01", "problem": "...", "answer": "42"},
    ...
  ]
}
```

Each year file MUST contain exactly 30 questions. The loader raises
`ValueError` otherwise.

- `id` is for traceability (we don't use it directly; the loader
  rebuilds qids as `aime-<year>-q<idx>`).
- `problem` is the full problem text.
- `answer` is a string (AIME answers are integers 0..999; storing
  as string sidesteps leading-zero ambiguity).

## Sources

| Year | Status (2026-04) | Where to fetch |
|---|---|---|
| 2024 | available | Most public Hugging Face mirrors of AMC datasets ship AIME 2024 — e.g. `Maxwell-Jia/AIME_2024`. |
| 2025 | available | AIME I + AIME II contests took place Feb 2025; problems and answers are public on AoPS (`https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions`). |
| 2026 | available | AIME I + AIME II Feb 2026 contests; AoPS wiki + MAA archive. |

For each year, AIME I has 15 problems and AIME II has 15 problems,
so 30 total per year. Pool both halves of the same year into one
`questions` array; the order does not matter (we sample
deterministically with seed 42 for 2024 only; 2025/2026 are loaded
in full).

## Where to put the files

The default config field is `Config.code_data_dir =
artifacts/datasets/`. The `artifacts/` tree is `.gitignored`, so
populating it is per-machine.

```
artifacts/datasets/
├── aime_2024.json
├── aime_2025.json
└── aime_2026.json
```

## Sanity check after fetching

```bash
python -c "
from pathlib import Path
from agentdiet.eval.datasets import AIMEMultiYearDataset
ds = AIMEMultiYearDataset(data_dir=Path('artifacts/datasets'))
qs = ds.load()
print(f'loaded {len(qs)} questions')
print(f'first qid: {qs[0].qid}')
"
```

Expected output: `loaded 80 questions`, `first qid: aime-2026-q00`.

## Why we don't auto-fetch

- AIME 2026 is post-cutoff for any frozen public dataset release.
- Manual curation gives us the chance to spot OCR / formatting issues
  in the problem text.
- Re-running the experiments doesn't re-hit the data source.

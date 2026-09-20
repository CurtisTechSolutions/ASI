# Rust against Go: the count / reward model, measured

Written by `bench/compare.py`; rerun it to replace this file.  Every row
trained on the same corpus and predicted the same prefixes, and the Go and
Rust runs were checked against each other before being timed (same graph,
same transitions, same loss, same prediction at the same cost).

The last two columns are against **Go's own default counting at the same
worker count** - the row above each group of three.

- **when** 2026-09-20 00:59 UTC
- **machine** Intel(R) Xeon(R) Processor @ 2.10GHz, Linux x86_64
- **go** go version go1.24.7 linux/amd64
- **rust** rustc 1.94.1 (e408947bf 2026-03-25)
- **corpus** 62,757 texts / 2,000,013 characters, 3 epochs, 3,000 predictions of 918,919 expansions
- **graph** 894 nodes, 4,575 edges, 1,048 trigrams, compression 1.1762
- **blame** every 7th text punished before the predictions (8,965 texts, 3,692 edges)
- **repeats** 3 runs per row, median reported
- **the two searches** differ on 596 of 3,000 continuations (19.9%)

| build | traversal | workers | train s | transitions/s | predict s | predictions/s | expansions | train vs Go | predict vs Go |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Go (racy, its default) | reward | 1 | 0.8266 | 6,212,717 | 0.7353 | 4,080 | 918,919 | 1.00x | 1.00x |
| Go (exact) | reward | 1 | 0.9406 | 5,459,494 | 0.7513 | 3,993 | 918,919 | 0.88x | 0.98x |
| Rust | reward | 1 | 0.3611 | 14,219,552 | 0.1916 | 15,657 | 918,919 | 2.29x | 3.84x |
| Go (racy, its default) | reward | all cores | 0.7543 | 6,808,141 | 0.8834 | 3,396 | 918,919 | 1.00x | 1.00x |
| Go (exact) | reward | all cores | 0.7634 | 6,726,917 | 0.8738 | 3,433 | 918,919 | 0.99x | 1.01x |
| Rust | reward | all cores | 0.2613 | 19,651,727 | 0.1914 | 15,676 | 918,919 | 2.89x | 4.62x |
| Go (racy, its default) | least-punished | 1 | 0.8340 | 6,157,089 | 0.0736 | 40,748 | 60,068 | 1.00x | 1.00x |
| Go (exact) | least-punished | 1 | 0.9552 | 5,376,051 | 0.0750 | 39,981 | 60,068 | 0.87x | 0.98x |
| Rust | least-punished | 1 | 0.3618 | 14,191,842 | 0.0113 | 266,194 | 60,068 | 2.30x | 6.53x |
| Go (racy, its default) | least-punished | all cores | 0.7128 | 7,204,648 | 0.0744 | 40,330 | 60,068 | 1.00x | 1.00x |
| Go (exact) | least-punished | all cores | 0.7740 | 6,634,617 | 0.0733 | 40,930 | 60,068 | 0.92x | 1.01x |
| Rust | least-punished | all cores | 0.2613 | 19,648,996 | 0.0120 | 250,423 | 60,068 | 2.73x | 6.21x |

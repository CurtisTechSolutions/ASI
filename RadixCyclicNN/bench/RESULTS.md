# Rust against Go: the count / reward model, measured

Written by `bench/compare.py`; rerun it to replace this file.  Every row
trained on the same corpus and predicted the same prefixes, and the Go and
Rust runs were checked against each other before being timed (same graph,
same transitions, same loss, same prediction at the same cost).

The last two columns are against **Go's own default counting at the same
worker count** - the row above each group of three.

- **when** 2026-09-20 03:04 UTC
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
| Go (racy, its default) | reward | 1 | 0.7838 | 6,551,689 | 0.7169 | 4,185 | 918,919 | 1.00x | 1.00x |
| Go (exact) | reward | 1 | 0.8661 | 5,928,926 | 0.7048 | 4,256 | 918,919 | 0.90x | 1.02x |
| Rust | reward | 1 | 0.3579 | 14,348,494 | 0.1865 | 16,087 | 918,919 | 2.19x | 3.84x |
| Go (racy, its default) | reward | all cores | 0.6906 | 7,435,905 | 0.7853 | 3,820 | 918,919 | 1.00x | 1.00x |
| Go (exact) | reward | all cores | 0.7260 | 7,073,650 | 0.8074 | 3,716 | 918,919 | 0.95x | 0.97x |
| Rust | reward | all cores | 0.2488 | 20,638,692 | 0.1869 | 16,054 | 918,919 | 2.78x | 4.20x |
| Go (racy, its default) | least-punished | 1 | 0.7710 | 6,660,382 | 0.0681 | 44,077 | 60,068 | 1.00x | 1.00x |
| Go (exact) | least-punished | 1 | 0.9029 | 5,687,671 | 0.0732 | 40,969 | 60,068 | 0.85x | 0.93x |
| Rust | least-punished | 1 | 0.3588 | 14,312,614 | 0.0109 | 274,601 | 60,068 | 2.15x | 6.23x |
| Go (racy, its default) | least-punished | all cores | 0.7054 | 7,279,337 | 0.0702 | 42,716 | 60,068 | 1.00x | 1.00x |
| Go (exact) | least-punished | all cores | 0.6798 | 7,553,548 | 0.0731 | 41,021 | 60,068 | 1.04x | 0.96x |
| Rust | least-punished | all cores | 0.2513 | 20,433,258 | 0.0122 | 245,934 | 60,068 | 2.81x | 5.76x |

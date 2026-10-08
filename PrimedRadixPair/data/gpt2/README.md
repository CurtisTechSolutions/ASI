# data/gpt2

Put GPT-2's two vocabulary files here and the `gpt2` codec reads them:

* `encoder.json` and `vocab.bpe` — the original release
  (`https://openaipublic.blob.core.windows.net/gpt-2/models/124M/`), or
* `vocab.json` and `merges.txt` — Hugging Face's copy of the same content.

`make gpt2-files` fetches the first pair when the network allows it. The files
are not committed: they are OpenAI's (MIT-licensed), about 1.5 MB, and the
tests that need them are skipped when they are absent — the tokenizer's
algorithm is tested on a synthetic vocabulary either way.

# cmd

The commands of this module, one directory per binary — the standard Go layout.
Everything importable lives in `../radixnet/` and `../server/`; nothing in here
is imported by anything else.

| directory | binary |
|---|---|
| `radixnet-count/` | `radixnet-count` — the CLI of the count / reward model |

```bash
cd ..                                   # the go/ directory
go build -o bin/radixnet-count ./cmd/radixnet-count
```

Or, from `../..`: `make go-build`.

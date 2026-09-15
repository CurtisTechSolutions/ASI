# docker

Container plumbing for RadixCyclicNN. The image itself is built from
`../Dockerfile`; this directory holds only what has to run *inside* it.

## Contents

| file | what it is |
|---|---|
| `entrypoint.sh` | the container entrypoint — optionally restores the newest checkpoint, then execs the `radixnet` CLI with whatever arguments the container was given |

## What `entrypoint.sh` does

`exec radixnet "$@"`, with one thing first: when `RADIXNET_RESUME=1` it restores
`<checkpoint dir>/latest.json` into the model file, so a restarted container
carries on from where a long `train` or `evolve` job left off instead of from
the last explicit save. A failed restore is a warning, not a stop — the container
continues with whatever model file it has.

| variable | default | meaning |
|---|---|---|
| `RADIXNET_RESUME` | `0` | `1` restores the latest checkpoint into the model file before starting |
| `RADIXNET_MODEL` | `/data/model.json` | the model file |
| `RADIXNET_CHECKPOINT_DIR` | `/data/checkpoints` | where checkpoints are rotated |

## Running the stack

From `..`:

```bash
docker compose up api                  # the HTTP API and the prebuilt frontend
docker compose up evolve               # the self-upgrading loop
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api   # with a GPU
```

`../.env.example` lists every setting the compose files read, `../README.md`
§ *Docker Compose* explains the services, and `../.dockerignore` decides what
reaches the build context.

Note that `api` and `evolve` are stopped with `SIGINT` rather than `SIGTERM`, so
they save their model on `docker stop` instead of being killed mid-write.

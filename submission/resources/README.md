# Model weights

The five ensemble checkpoints are not tracked in this repository because of their size
(~330 MB each). Produce them with:

```
./reproduce_runs.sh
```

which writes `dinov2_lora_attnmil_neoplasia_seed{42..46}.pt` into this directory. The container
loads every file matching that pattern, so the ensemble size follows from what is present here.

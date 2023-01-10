#!/usr/bin/env bash
fairseq-train \
    data-bin/wikitext-103 \
    --user-dir spade-modules \
    --arch spade_lm_wiki103 --task spade_language_modeling \
    --attention-type local --local-radius 511 \
    --no-token-positional-embeddings \
    --use-relative-attention --relative-attention-num-buckets 32 --relative-attention-max-distance 128 \
    --s4-every-n-layers 100 --s4-local-combine concat \
    --s4-state-dim 64 --s4-channels 1 --s4-n-ssm 1024 --s4-dt-min 0.001 --s4-dt-max 0.1 \
    --s4-lr '{"A": 0.001, "B": 0.001, "dt": null}' \
    --lr 1.0 --lr-scheduler cosine --min-lr 0.0001 --lr-period-updates 270000 \
    --t-mult 2 --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 \
    --max-update 286000 --stop-min-lr 1e-09 \
    --optimizer nag --clip-norm 0.1 \
    --criterion adaptive_loss \
    --max-tokens 3072 --tokens-per-sample 3072 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test \
    --log-format json --log-interval 10 \
    --num-workers 0 \
    --ddp-backend legacy_ddp \
    --update-freq 3 \
    --seed 1 \

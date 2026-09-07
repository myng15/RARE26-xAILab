# RARE26 Challenge — xAILab

Code for the xAILab Bamberg team to the [RARE26 challenge (MICCAI 2026)](https://rare26.grand-challenge.org/): early-stage, low-prevalence cancer detection (Barrett’s Esophagus neoplasia) in endoscopy images.

The repository covers the complete method: data splitting, training, validation, and the inference container submitted to the challenge platform. The container layout follows the
[example submission repository](https://github.com/TUE-ARIA/RARE25-Submission) provided by the challenge organizers.

<!-- ## Method

A **DINOv2 ViT-B/14** backbone with 4 register tokens, self-supervised pretrained on gastrointestinal
endoscopy imagery, adapted with **LoRA** on the attention `qkv` projections and fitted with **focal loss**.

The design choice that matters most is reading the backbone's **patch tokens** rather than its single
whole-image CLS token. Neoplasia is often a small, localized region of an otherwise normal-looking
frame, and a whole-image summary dilutes it. The patch tokens are combined into a frame embedding by
**attention-based multiple-instance pooling** (Ilse et al., ICML 2018, Eq. 8): a small side-network scores
every patch, the scores are softmax-normalized across patches into weights, and the embedding is their weighted sum. The pooling is trained from image-level labels only; no lesion annotations are used.

The final prediction averages the per-frame probabilities of **five models trained with different random
seeds**. Every frame is scored independently, so the output does not depend on how the platform batches
the test set. -->

## 1. Environment

```bash
conda create -n rare26 python=3.12
conda activate rare26
pip install -r requirements.txt
```

Clone the DINOv2 reference implementation, which supplies the backbone definition. The same checkout is used for training and the container build:

```bash
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
cp -r third_party inference/third_party    # the container build copies this into the image
```

## 2. Data

Place the challenge training set under `data/train`, organized as `<center>/<class>/<image>.png`, with
class directories named `ndbe` (non-dysplastic Barrett's esophagus) and `neo` (neoplasia).

The backbone is initialized from [GastroNet-5M](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) self-supervised weights; access must be requested there. Place the checkpoint in `resources/` and pass it
via `--backbone_weights`. 

## 3. Data splitting

```bash
python data_splitting/create_splits.py
```

writes four CSVs to `data/splits`:

| file | purpose |
|---|---|
| `center1_train_center2_test.csv` | train on center 1, evaluate on center 2 |
| `center2_train_center1_test.csv` | train on center 2, evaluate on center 1 |
| `pooled_holdout.csv` | both centers pooled, held-out slice stratified by center and label |
| `5fold_cv.csv` | stratified 5-fold cross-validation |

The cross-center splits measure transfer to an unseen center. The pooled split matches the condition the deployed model is trained under (the final model is fitted on all available data from both centers) and is the split used to
confirm whether a change is adopted. A change can help on one cross-center direction and not survive pooled training, so both are reported.

The 5-fold split is reported alongside the other splits for reference, but is not used to decide whether a candidate change is adopted.

## 4. Training

```bash
# the cross-center generalizability evaluation
python -m training.run_training --split cross_center \
    --split_csv data/splits/center2_train_center1_test.csv --seed 42

# the pooled held-out evaluation
python -m training.run_training --split pooled --seed 42

# a deployable checkpoint, trained on every labeled image
python -m training.run_training --split final --seed 42 \
    --export "inference/resources/dinov2_lora_attnmil_neoplasia_seed42.pt"
```

Each evaluating run writes per-frame predictions, point-estimate metrics, and a bootstrap interval to `results/`. The bootstrap follows the same resampling scheme the challenge platform uses to score the leaderboard (see `validation/metrics.py`, §5 below), so these local intervals are comparable to it.

## 5. Validation

```bash
python -m validation.run_validation results/predictions_pooled_attention_seed*.csv
```

scores one prediction file, or averages several into an ensemble and scores that.

The challenge scores the median, over bootstrap resamples at ~1% prevalence, of the positive predictive value at 90% recall (PPV@90%Recall). `validation/metrics.py` reproduces that resampling scheme so local numbers are comparable to the leaderboard, and additionally reports AUROC, AUPRC, the false positive rate at 90% recall, and a partial AUC over a band of the ROC curve near the operating point (pAUC@90%Recall).

To reproduce every run behind the submission:

```bash
BACKBONE_WEIGHTS=resources/gastronet_dinov2_vitb.pth ./reproduce_runs.sh
```

## 6. Submission container

```bash
cd inference
# needs inference/third_party/dinov2 and the checkpoints in inference/resources
./do_build.sh        # build the image
./do_save.sh         # write the tarball to upload
```

## Repository layout

```
data_splitting/create_splits.py   split definitions
model/architecture.py             backbone, LoRA adaptation, pooling, checkpoint export
model/config.py                   hyperparameters of the submitted configuration
training/data.py                  datasets, transforms, class-balanced sampling
training/losses.py                focal loss, MixUp
training/trainer.py               training and prediction loops
training/run_training.py          train one model under one evaluation protocol
validation/metrics.py             challenge metrics and bootstrap evaluation
validation/run_validation.py      score predictions, single or ensembled
inference/                        submission container
reproduce_runs.sh                 reproduce every run behind the submission
```

`model/architecture.py` also contains an unweighted mean-pooling head, selectable with `--pooling mean`, which was used during development as a reference point for the attention head.

## License

See `LICENSE`.

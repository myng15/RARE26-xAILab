# RARE26 Challenge — xAILab

Code for the xAILab entry to the [RARE26 challenge](https://rare26.grand-challenge.org/): per-frame
detection of Barrett's neoplasia in endoscopy images.

The repository covers the complete method — data splitting, training, evaluation, and the inference
container that is submitted to the challenge platform. The submission container layout follows the
[example submission repository](https://github.com/TUE-VCA/RARE25-Submission) provided by the organisers.

## Method

A DINOv2 ViT-B/14 backbone with 4 register tokens, self-supervised pretrained on gastrointestinal
endoscopy imagery, adapted with LoRA on the attention `qkv` projections and fitted with focal loss.

The design choice that matters most for this task is reading the backbone's **patch tokens** rather than
its single whole-image summary token, and pooling them into a frame embedding. Neoplasia is often a small,
localised region of an otherwise unremarkable frame, and a whole-image summary dilutes it. Two pooling
rules are provided and both were submitted to the leaderboard:

- `attention` — attention-based multiple-instance pooling (Ilse et al., ICML 2018, Eq. 8). A small
  side-network scores each patch, the scores are softmax-normalised into weights, and the embedding is
  their weighted sum. Trained from image-level labels only.
- `mean` — an unweighted mean over patch tokens, with no learned pooling parameters.

The final prediction averages the per-frame probabilities of five models trained with different random
seeds. Every frame is scored independently, so the output does not depend on how the platform batches the
test set.

## 1. Environment

```bash
conda create -n rare26 python=3.12
conda activate rare26
pip install -r requirements.txt
```

Clone the DINOv2 reference implementation, which supplies the backbone definition. It is not vendored
here; the same checkout is used for training and for the container build:

```bash
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
cp -r third_party submission/third_party    # the container build copies this into the image
```

## 2. Data

Place the challenge training set under `data/train`, organised as `<center>/<class>/<image>.png` with
class directories named `ndbe` (non-dysplastic Barrett's oesophagus) and `neo` (neoplasia).

The backbone is initialised from [GastroNet-5M](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights)
self-supervised weights; access must be requested from that repository. Place the checkpoint in
`resources/` and point `--backbone_weights` at it. Training also runs from a randomly initialised or
generically pretrained backbone, at reduced accuracy.

## 3. Splits

```bash
python data_splitting/create_splits.py
```

writes four CSVs to `data/splits`:

| file | purpose |
|---|---|
| `center1_train_center2_test.csv` | train on centre 1, evaluate on centre 2 |
| `center2_train_center1_test.csv` | train on centre 2, evaluate on centre 1 |
| `pooled_holdout.csv` | both centres pooled, held-out slice stratified by centre and label |
| `5fold_cv.csv` | stratified 5-fold cross-validation |

The cross-centre splits measure transfer to an unseen centre. The pooled split matches the condition the
deployed model is trained under — the final model is fitted on both centres — and is the split used to
decide whether a change is adopted. A change can help on one cross-centre direction and not survive
pooled training, so both are reported.

## 4. Training and evaluation

```bash
# transfer to an unseen centre
python train.py --split cross_center --split_csv data/splits/center2_train_center1_test.csv --seed 42

# the pooled held-out evaluation
python train.py --split pooled --seed 42

# a deployable checkpoint, trained on every labelled image
python train.py --split final --seed 42 --export submission/resources/dinov2_lora_attnmil_neoplasia_seed42.pt
```

Add `--pooling mean` for the mean-pooling variant. Each evaluating run writes per-frame predictions,
point-estimate metrics and a bootstrap interval to `results/`.

`python evaluate.py results/predictions_*.csv` scores one prediction file, or averages several into an
ensemble and scores that.

To reproduce every run behind the submission:

```bash
BACKBONE_WEIGHTS=resources/gastronet_dinov2_vitb.pth ./reproduce_runs.sh
```

### Metrics

The challenge scores the median, over bootstrap resamples at roughly 1% prevalence, of the positive
predictive value at 90% recall. `rare26/metrics.py` reproduces that resampling scheme so local numbers are
comparable to the leaderboard, and additionally reports AUROC, average precision, the false-positive rate
at 90% recall, and a partial AUC over a band of the ROC curve near the operating point. The partial AUC is
reported over a narrow band as well as the conventional `[0.90, 1.00]`: with few positives in an
evaluation set, the top of that band is determined by the handful of hardest positives and tracks the
challenge metric poorly.

## 5. Submission container

```bash
cd submission
# needs submission/third_party/dinov2 and the checkpoints in submission/resources
./do_build.sh        # build the image
./do_test_run.sh     # run it on the bundled example input
./do_save.sh         # write the tarball to upload
```

`submission/resources/` must contain the exported checkpoints; see the README there.

## Repository layout

```
data_splitting/create_splits.py   split definitions
rare26/config.py                  hyperparameters of the submitted configuration
rare26/data.py                    datasets, transforms, class-balanced sampling
rare26/model.py                   backbone, LoRA adaptation, pooling heads, checkpoint export
rare26/losses.py                  focal loss, MixUp
rare26/train.py                   training and prediction loops
rare26/metrics.py                 challenge metrics and bootstrap evaluation
train.py                          train one model under one protocol
evaluate.py                       score predictions, single or ensembled
reproduce_runs.sh                 every run behind the submission
submission/                       inference container
```

## License

See `LICENSE`.

#!/bin/bash
# Gene2Image main experiment + ablation runner.
# 6 variants x 3 datasets x 3 seeds (single-cell, img_channels=4), per implementation.md 2.1/2.2.
#
# Variants (three orthogonal switches vs Gene2Image):
#   gene2image : real mask  + learnable + transformer   (main method)
#   geneflow   : encoder_type=rna                         (SOTA baseline / lower bound)
#   randpath   : rand mask  + learnable + transformer    (RQ2 mechanism)
#   pathprior  : real mask  + frozen ssGSEA + transformer (RQ3, --no_learnable_pathway)
#   notrans    : real mask  + learnable + NO transformer (component)
#   nomask     : none mask  + learnable + transformer    (component)
#
# Usage:
#   bash scripts/run_experiments.sh <variant> <dataset> <seed> [extra args...]
#   e.g. bash scripts/run_experiments.sh gene2image c1 42
# Or loop everything (sequential; each full run ~hours on a single GPU):
#   bash scripts/run_experiments.sh all
#
# Adjust BATCH_SIZE/EPOCHS for your GPU (V100 32GB: keep batch modest, use AMP).
set -e

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data/processed_data}
MASK_DIR=${MASK_DIR:-data/pathway_masks}
OUT_ROOT=${OUT_ROOT:-results}
DB=${DB:-hallmark}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-100}
GEN_STEPS=${GEN_STEPS:-100}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-"--use_amp"}

# Map short dataset id -> processed_data folder.
dataset_dir() {
  case "$1" in
    c1) echo "Xenium_V1_hSkin_Melanoma_Base_FFPE" ;;
    c2) echo "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE" ;;
    p1) echo "Xenium_Prime_Human_Skin_FFPE" ;;
    *)  echo "UNKNOWN" ;;
  esac
}

run_one() {
  local variant=$1 ds=$2 seed=$3; shift 3
  local folder; folder=$(dataset_dir "$ds")
  if [ "$folder" = "UNKNOWN" ]; then echo "Unknown dataset: $ds"; exit 1; fi

  local adata="$DATA_ROOT/$folder/adata.h5ad"
  local imgpaths="$DATA_ROOT/$folder/cell_patch_256_aux/input/cell_image_paths_local.json"
  local out="$OUT_ROOT/${variant}_${ds}_seed${seed}"

  # Ensure local image paths exist (remap once if missing).
  if [ ! -f "$imgpaths" ]; then
    echo "Remapping image paths for $ds ..."
    $PY scripts/fix_image_paths.py \
      --json "$DATA_ROOT/$folder/cell_patch_256_aux/input/cell_image_paths.json" \
      --local_root "$DATA_ROOT"
  fi

  # Per-variant encoder flags.
  local enc_args=""
  case "$variant" in
    gene2image) enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz" ;;
    geneflow)   enc_args="--encoder_type rna" ;;
    randpath)   enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_rand.npz" ;;
    pathprior)  enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz --no_learnable_pathway" ;;
    notrans)    enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_real.npz --no_pathway_transformer" ;;
    nomask)     enc_args="--encoder_type pathway --pathway_mask $MASK_DIR/${ds}_${DB}_none.npz" ;;
    *) echo "Unknown variant: $variant"; exit 1 ;;
  esac

  echo "=== $variant | $ds | seed=$seed -> $out ==="
  $PY rectified/rectified_main.py \
    --model_type single --img_size 256 --img_channels 4 \
    --adata "$adata" --image_paths "$imgpaths" \
    --output_dir "$out" \
    --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --gen_steps "$GEN_STEPS" \
    --num_dataloader_workers "$WORKERS" --seed "$seed" \
    --pathway_db "$DB" $enc_args $EXTRA "$@"
}

if [ "$1" = "all" ]; then
  for variant in gene2image geneflow randpath pathprior notrans nomask; do
    for ds in c1 c2 p1; do
      for seed in 42 43 44; do
        run_one "$variant" "$ds" "$seed"
      done
    done
  done
else
  run_one "$@"
fi

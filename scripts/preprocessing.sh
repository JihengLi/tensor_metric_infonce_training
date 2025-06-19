#!/usr/bin/env bash
# run_4ch_metrics.sh
# Usage: ./run_4ch_metrics.sh <input_dir> <output_dir>

set -euo pipefail
IFS=$'\n\t'

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

input_dir=$1
output_dir=$2

echo "=== Starting 4-channel DTI metric extraction ==="
echo "Input directory: $input_dir"
echo "Output directory: $output_dir"

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MRTRIX_TMPFILE_DIR=/tmp

mkdir -p "$output_dir"
sudo chmod 777 "$output_dir"

find "$input_dir" -type f -name "*.mha" | while read -r dwi_mha; do
    subj=$(basename "${dwi_mha%.*}")
    json_file="${dwi_mha%.mha}.json"
    if [[ ! -f $json_file ]]; then
        echo "ERROR: Missing JSON file for subject $subj -> $json_file"
        exit 1
    fi

    echo
    echo "=== Processing subject: $subj ==="
    tmp_dir="$output_dir/${subj}_tmp"
    subj_out="$output_dir/$subj"
    mkdir -p "$tmp_dir" "$subj_out"

    echo "--> Converting JSON to bval/bvec"
    python3 convert_json_to_bvalbvec.py \
        "$json_file" \
        "$tmp_dir/${subj}.bval" \
        "$tmp_dir/${subj}.bvec"
    bval="$tmp_dir/${subj}.bval"
    bvec="$tmp_dir/${subj}.bvec"

    echo "--> Converting MHA to NIfTI"
    python3 convert_mha_to_nifti.py \
        "$dwi_mha" \
        "$tmp_dir/${subj}.nii.gz"
    nifti="$tmp_dir/${subj}.nii.gz"

    echo "--> Performing skull stripping with BET (f=0.2)"
    bet "$nifti" "$tmp_dir/bet_stripped" -m -f 0.2
    mask="$tmp_dir/bet_stripped_mask.nii.gz"

    echo "--> Applying mask to DWI volume"
    mrcalc "$nifti" "$mask" -mult "$tmp_dir/dwi_brain.nii.gz"
    dwi_for_tensor="$tmp_dir/dwi_brain.nii.gz"

    echo "--> Fitting diffusion tensors"
    dwi2tensor "$dwi_for_tensor" \
        "$tmp_dir/tensor.mif" \
        -mask "$mask" \
        -fslgrad "$bvec" "$bval" \
        -nthreads $OMP_NUM_THREADS

    echo "--> Converting tensor.mif to NIfTI with 6 tensor components"
    mrconvert "$tmp_dir/tensor.mif" "$subj_out/metrics_tensor_raw.nii.gz"
    echo "--> Saved 6-channel raw tensor: $subj_out/metrics_tensor_raw.nii.gz"

    echo "--> Cleaning up temporary files for $subj"
    rm -rf "$tmp_dir"
done

echo
echo "=== All subjects processed. Metrics available in $output_dir ==="
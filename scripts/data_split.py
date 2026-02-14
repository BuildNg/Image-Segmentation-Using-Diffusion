"""
Split LIDC-IDRI-slices into training (~13000 slices) and testing sets.

Produces the same directory structure as data_example/:
    data/training/<slice_id>/
        image_<slice_id>.png
        label0_<slice_id>.png
        label1_<slice_id>.png
        label2_<slice_id>.png
        label3_<slice_id>.png
    data/testing/<slice_id>/
        (same layout)

The split is done at the *patient* level to avoid data leakage
(no patient appears in both training and testing).
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image


def collect_slices(lidc_dir):
    """
    Walk the LIDC-IDRI-slices directory and collect all slices.
    Returns a dict: {patient_id: [(image_path, {mask_name: mask_path}, slice_name), ...]}
    """
    patients = {}
    lidc_dir = Path(lidc_dir)

    for patient_dir in sorted(lidc_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name  # e.g. LIDC-IDRI-0001
        patient_slices = []

        for nodule_dir in sorted(patient_dir.iterdir()):
            if not nodule_dir.is_dir():
                continue
            images_dir = nodule_dir / "images"
            if not images_dir.is_dir():
                continue

            # Collect mask directories
            mask_dirs = sorted([
                d for d in nodule_dir.iterdir()
                if d.is_dir() and d.name.startswith("mask")
            ])

            # Each image slice maps to corresponding masks
            for img_file in sorted(images_dir.iterdir()):
                if not img_file.is_file():
                    continue
                slice_name = img_file.stem  # e.g. "slice-0"
                masks = {}
                for i, mask_dir in enumerate(mask_dirs):
                    mask_file = mask_dir / img_file.name
                    if mask_file.is_file():
                        masks[f"label{i}"] = mask_file
                
                if len(masks) == 4:  # only include fully annotated slices
                    patient_slices.append((img_file, masks, f"{patient_id}_{nodule_dir.name}_{slice_name}"))

        if patient_slices:
            patients[patient_id] = patient_slices

    return patients


def split_patients(patients, target_train_slices=13000, seed=42):
    """
    Split patients into train/test sets such that training has ~target_train_slices.
    Split is at the patient level to prevent data leakage.
    """
    random.seed(seed)
    patient_ids = list(patients.keys())
    random.shuffle(patient_ids)

    train_patients = []
    test_patients = []
    train_count = 0

    for pid in patient_ids:
        n_slices = len(patients[pid])
        if train_count < target_train_slices:
            train_patients.append(pid)
            train_count += n_slices
        else:
            test_patients.append(pid)

    return sorted(train_patients), sorted(test_patients)


def copy_slices(patients, patient_ids, output_dir):
    """
    Copy slices into the output directory with the expected structure:
        output_dir/<global_idx>/
            image_<global_idx>.png
            label0_<global_idx>.png
            label1_<global_idx>.png
            label2_<global_idx>.png
            label3_<global_idx>.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    for pid in sorted(patient_ids):
        for img_path, masks, slice_id in patients[pid]:
            slice_dir = output_dir / str(global_idx)
            slice_dir.mkdir(parents=True, exist_ok=True)

            # Convert image to .jpg
            img = Image.open(img_path)
            img.save(slice_dir / f"image_{global_idx}.jpg")

            # Convert masks to .jpg as label0, label1, label2, label3
            for label_name, mask_path in sorted(masks.items()):
                mask = Image.open(mask_path)
                mask.save(slice_dir / f"{label_name}_{global_idx}.jpg")

            global_idx += 1

    return global_idx


def main():
    parser = argparse.ArgumentParser(description="Split LIDC-IDRI-slices into train/test sets")
    parser.add_argument("--source", type=str, default="./LIDC-IDRI-slices",
                        help="Path to LIDC-IDRI-slices directory")
    parser.add_argument("--output", type=str, default="./data",
                        help="Output directory (will contain training/ and testing/)")
    parser.add_argument("--train_slices", type=int, default=13000,
                        help="Approximate number of training slices (default: 13000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Collecting slices from {args.source}...")
    patients = collect_slices(args.source)
    total_slices = sum(len(v) for v in patients.values())
    print(f"Found {len(patients)} patients with {total_slices} total slices")

    print(f"\nSplitting patients (target ~{args.train_slices} training slices)...")
    train_patients, test_patients = split_patients(patients, args.train_slices, args.seed)

    train_slices = sum(len(patients[p]) for p in train_patients)
    test_slices = sum(len(patients[p]) for p in test_patients)
    print(f"  Training: {len(train_patients)} patients, {train_slices} slices")
    print(f"  Testing:  {len(test_patients)} patients, {test_slices} slices")

    train_dir = os.path.join(args.output, "training")
    test_dir = os.path.join(args.output, "testing")

    print(f"\nCopying training slices to {train_dir}...")
    n_train = copy_slices(patients, train_patients, train_dir)
    print(f"  Copied {n_train} slices")

    print(f"\nCopying testing slices to {test_dir}...")
    n_test = copy_slices(patients, test_patients, test_dir)
    print(f"  Copied {n_test} slices")

    print(f"\nDone! Data split saved to {args.output}/")
    print(f"  {train_dir}/ ({n_train} slices)")
    print(f"  {test_dir}/ ({n_test} slices)")


if __name__ == "__main__":
    main()

"""
Automated dataset acquisition and structure verification for the OCT2017 dataset.
Fetches from Hugging Face or provides instructions for Kaggle download.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.DataAcquisition")


def scan_dataset_distribution(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Scans and counts image files across classes and splits."""
    stats = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            stats[split] = {}
            for cls in config.classes:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    count = len([f for f in cls_dir.iterdir() if f.suffix.lower() in valid_extensions])
                    stats[split][cls] = count
                else:
                    stats[split][cls] = 0
                    
    # Also check if flat class structure exists
    flat_stats = {}
    for cls in config.classes:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            count = len([f for f in cls_dir.iterdir() if f.suffix.lower() in valid_extensions])
            flat_stats[cls] = count
            
    if flat_stats and any(flat_stats.values()):
        stats["flat_root"] = flat_stats
        
    return stats


def print_dataset_summary(stats: Dict[str, Dict[str, int]]):
    """Prints a clear summary table of dataset statistics."""
    print("\n" + "=" * 65)
    print("           OCT2017 DATASET DISTRIBUTION SUMMARY")
    print("=" * 65)
    
    header = f"{'Split':<12} | " + " | ".join([f"{cls:<10}" for cls in config.classes]) + " | Total"
    print(header)
    print("-" * 65)
    
    grand_total = 0
    for split, counts in stats.items():
        split_total = sum(counts.values())
        grand_total += split_total
        row = f"{split:<12} | " + " | ".join([f"{counts.get(cls, 0):<10}" for cls in config.classes]) + f" | {split_total:<6}"
        print(row)
        
    print("-" * 65)
    print(f"Grand Total Images Indexed: {grand_total}")
    print("=" * 65 + "\n")


def download_from_huggingface(output_dir: Path):
    """Downloads dataset from Hugging Face via the datasets library."""
    try:
        from datasets import load_dataset
        logger.info("Connecting to Hugging Face Hub (keremberke/retinal-oct-classification)...")
        hf_ds = load_dataset("keremberke/retinal-oct-classification", name="full")
        
        for split_name in ["train", "validation", "test"]:
            target_split = "val" if split_name == "validation" else split_name
            split_data = hf_ds[split_name]
            logger.info(f"Extracting {len(split_data)} images for split: {target_split}")
            
            for item in split_data:
                img = item["image"]
                label_idx = item["label"]
                # HF label mapping
                label_name = config.classes[label_idx] if label_idx < len(config.classes) else "UNKNOWN"
                
                dest_dir = output_dir / target_split / label_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Save as JPEG
                img_id = item.get("image_id", hash(img.tobytes()) % 10000000)
                img.save(str(dest_dir / f"{label_name.lower()}_{img_id}.jpeg"))
                
        logger.info(f"Dataset successfully downloaded and unpacked to {output_dir}")
    except Exception as e:
        logger.error(f"Hugging Face download failed: {e}")
        print("\nAlternative: To use the Kaggle dataset, run:")
        print("  kaggle datasets download -d paultimothymooney/kermany2018")
        print(f"  unzip kermany2018.zip -d {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCT2017 Dataset Acquisition & Audit Tool")
    parser.add_argument("--download", action="store_true", help="Download full dataset from Hugging Face")
    parser.add_argument("--path", type=str, default=str(config.data_dir), help="Target data directory")
    args = parser.parse_args()
    
    target_path = Path(args.path)
    
    if args.download:
        download_from_huggingface(target_path)
        
    stats = scan_dataset_distribution(target_path)
    print_dataset_summary(stats)

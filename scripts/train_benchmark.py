"""
Training and Benchmarking Orchestrator.
Trains and compares all 5 CNN architectures:
1. Custom CNN
2. ResNet-18
3. VGG-16
4. MobileNet-V2
5. DenseNet-121

Generates:
- Checkpoints in models/checkpoints/
- Learning curves in reports/
- Confusion matrices in reports/
- Comprehensive comparison metrics in reports/model_comparison.json & .csv
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

from src.config import config
from src.dataset import build_dataloaders
from src.engine.evaluator import ModelEvaluator
from src.engine.trainer import ModelTrainer
from src.models.model_factory import build_model, count_parameters
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything

logger = setup_logger("RDD-DL.Benchmark")


def measure_inference_latency(model: nn.Module, device: torch.device, repetitions: int = 50) -> float:
    """Measures single-image inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    start = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_input)
    total_time = time.time() - start
    avg_latency_ms = (total_time / repetitions) * 1000.0
    return round(avg_latency_ms, 2)


def run_benchmark(
    data_dir: Path = config.samples_dir,
    epochs: int = 5,
    models_to_run: List[str] = None
):
    seed_everything(config.seed)
    models_to_run = models_to_run or ["custom_cnn", "resnet18", "vgg16", "mobilenetv2", "densenet121"]
    
    logger.info(f"Initiating benchmark on dataset: {data_dir} for models: {models_to_run}")
    
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        num_workers=0
    )
    
    results = []
    
    for model_name in models_to_run:
        logger.info("\n" + "=" * 55)
        logger.info(f"Starting Training & Evaluation for: {model_name.upper()}")
        logger.info("=" * 55)
        
        checkpoint_path = config.checkpoints_dir / f"best_{model_name}.pt"
        curves_path = config.reports_dir / f"learning_curves_{model_name}.png"
        cm_path = config.reports_dir / f"confusion_matrix_{model_name}.png"
        
        # Build model
        model, _ = build_model(model_name, num_classes=len(classes), pretrained=True)
        param_stats = count_parameters(model)
        
        # Train
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            checkpoint_path=checkpoint_path,
            device=config.device
        )
        history = trainer.fit()
        trainer.plot_history(curves_path)
        
        # Evaluate
        evaluator = ModelEvaluator(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=config.device
        )
        metrics, y_true, y_pred, y_probs = evaluator.evaluate()
        evaluator.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # Inference Latency
        latency_ms = measure_inference_latency(model, config.device)
        
        result_record = {
            "Model": model_name,
            "Parameters (M)": round(param_stats["total_parameters"] / 1e6, 2),
            "Trainable Params (M)": round(param_stats["trainable_parameters"] / 1e6, 2),
            "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
            "Precision (Weighted)": round(metrics["precision_weighted"], 4),
            "Recall (Weighted)": round(metrics["recall_weighted"], 4),
            "F1-Score (Weighted)": round(metrics["f1_weighted"], 4),
            "Latency (ms/img)": latency_ms,
            "Checkpoint": str(checkpoint_path.name)
        }
        results.append(result_record)
        logger.info(f"Completed {model_name}: Acc={result_record['Accuracy (%)']}%, F1={result_record['F1-Score (Weighted)']}, Latency={latency_ms}ms")

    # Export comparison reports
    df_results = pd.DataFrame(results)
    csv_path = config.reports_dir / "model_comparison.csv"
    json_path = config.reports_dir / "model_comparison.json"
    
    df_results.to_csv(str(csv_path), index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Saved benchmark results to {csv_path} and {json_path}")
    print("\n" + "=" * 80)
    print("                     FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Model Benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per model")
    parser.add_argument("--data_dir", type=str, default=str(config.samples_dir), help="Path to dataset")
    parser.add_argument("--models", nargs="+", default=["custom_cnn", "resnet18", "vgg16", "mobilenetv2", "densenet121"])
    args = parser.parse_args()
    
    run_benchmark(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        models_to_run=args.models
    )

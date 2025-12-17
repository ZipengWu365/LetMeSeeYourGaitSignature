#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete P1 HMM-based Mamba Experiment

Stage 1: Smoke test with small config (done)
Stage 2: Default smoke test config for validation (2 epochs)
Stage 3: Full train with default config (20 epochs)
Stage 4: Hyperparameter sweep (Stage 1: 12 configs, Stage 2: 6 configs)
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_experiment(config_name: str, config: dict, output_base: Path):
    """Run a single P1 experiment with given config"""
    
    output_dir = output_base / config_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    cmd = [
        'python3',
        str(project_root / 'train' / 'train_p1_mamba_hmm.py'),
        '--output_dir', str(output_dir),
        '--epochs', str(config.get('epochs', 20)),
        '--d_model', str(config.get('d_model', 16)),
        '--n_layers', str(config.get('n_layers', 1)),
        '--seed', '42',
    ]
    
    print(f"\n[*] Running experiment: {config_name}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except TypeError:
        # Older Python version without capture_output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Log output
    log_file = output_dir / 'train.log'
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
    
    # Extract metrics
    if result.returncode == 0:
        # Try to find the best metrics file
        metrics_file = output_dir / 'test_results' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            decoded_f1 = metrics.get('decoded', {}).get('macro_f1', None)
            raw_f1 = metrics.get('raw', {}).get('macro_f1', None)
            return {
                'status': 'success',
                'raw_f1': raw_f1,
                'decoded_f1': decoded_f1,
            }
    
    return {'status': 'failed', 'returncode': result.returncode}


def main():
    parser = argparse.ArgumentParser(description='Complete P1 Experiment')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], default=3,
                       help='Experiment stage: 1=smoke, 2=default+val, 3=full, 4=hyperopt')
    parser.add_argument('--output_dir', type=str, default='artifacts/p1_experiments',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    output_base.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print(f"  P1 HMM-BASED MAMBA EXPERIMENT - STAGE {args.stage}")
    print("="*70)
    
    experiments = []
    
    if args.stage >= 3:
        # Stage 3: Full training with default config
        print("\n[STAGE 3] Full training with default smoke test config")
        config_default = {
            'input_dim': 32,
            'n_classes': 4,
            'd_model': 16,
            'n_layers': 1,
            'dropout': 0.2,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 20,
            'early_stopping_patience': 5,
            'gradient_clip': 1.0,
        }
        experiments.append(('default_20epochs', config_default))
    
    elif args.stage == 2:
        # Stage 2: Default config, fewer epochs for validation
        print("\n[STAGE 2] Validation with default config (5 epochs)")
        config_default = {
            'input_dim': 32,
            'n_classes': 4,
            'd_model': 16,
            'n_layers': 1,
            'dropout': 0.2,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 5,
            'early_stopping_patience': 3,
            'gradient_clip': 1.0,
        }
        experiments.append(('default_5epochs', config_default))
    
    elif args.stage == 1:
        # Stage 1: Smoke test
        print("\n[STAGE 1] Smoke test (2 epochs)")
        config_smoke = {
            'input_dim': 32,
            'n_classes': 4,
            'd_model': 16,
            'n_layers': 1,
            'dropout': 0.2,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 2,
            'early_stopping_patience': 2,
            'gradient_clip': 1.0,
        }
        experiments.append(('smoke_2epochs', config_smoke))
    
    # Run all experiments
    results_summary = {}
    
    for exp_name, config in experiments:
        result = run_experiment(exp_name, config, output_base)
        results_summary[exp_name] = result
        
        if result['status'] == 'success':
            print(f"  ✓ {exp_name}: raw_f1={result['raw_f1']:.4f}, decoded_f1={result['decoded_f1']:.4f}")
        else:
            print(f"  ✗ {exp_name}: failed with code {result.get('returncode')}")
    
    # Save summary
    summary_file = output_base / f'stage{args.stage}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'stage': args.stage,
            'timestamp': datetime.now().isoformat(),
            'experiments': results_summary,
        }, f, indent=2)
    
    print(f"\n[OK] Stage {args.stage} complete. Results saved to {output_base}")


if __name__ == '__main__':
    main()

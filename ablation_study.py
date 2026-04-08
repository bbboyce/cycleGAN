"""
CycleGAN Ablation Study Framework
=================================
Systematic ablation experiments to understand the contribution of each loss component.

Losses in CycleGAN:
1. GAN Loss (G_A, G_B)
2. Cycle Consistency Loss (cycle_A, cycle_B)
3. Identity Loss (idt_A, idt_B)

Usage:
    python ablation_study.py --config ablation_configs.json
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


class AblationStudy:
    """Manage ablation study experiments"""

    def __init__(self, config_path="ablation_configs.json"):
        self.config_path = config_path
        self.results = OrderedDict()
        self.load_config()

    def load_config(self):
        """Load ablation study configuration"""
        if not os.path.exists(self.config_path):
            print(f"❌ Config file not found: {self.config_path}")
            print("Creating default configuration...")
            self.create_default_config()

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def create_default_config(self):
        """Create default ablation study configuration"""
        config = {
            "base_experiment": {
                "name": "Full CycleGAN (Baseline)",
                "description": "All loss components enabled",
                "lambda_A": 10.0,
                "lambda_B": 10.0,
                "lambda_identity": 0.5,
                "lambda_gan_a": 1.0,
                "lambda_gan_b": 1.0,
                "lambda_cycle_a": 1.0,
                "lambda_cycle_b": 1.0
            },
            "experiments": [
                {
                    "name": "No Cycle Loss",
                    "description": "Remove cycle consistency loss (lambda_A=0, lambda_B=0)",
                    "lambda_A": 0.0,
                    "lambda_B": 0.0,
                    "lambda_identity": 0.5,
                    "lambda_gan_a": 1.0,
                    "lambda_gan_b": 1.0,
                    "lambda_cycle_a": 0.0,
                    "lambda_cycle_b": 0.0
                },
                {
                    "name": "No Identity Loss",
                    "description": "Remove identity loss (lambda_identity=0)",
                    "lambda_A": 10.0,
                    "lambda_B": 10.0,
                    "lambda_identity": 0.0,
                    "lambda_gan_a": 1.0,
                    "lambda_gan_b": 1.0,
                    "lambda_cycle_a": 1.0,
                    "lambda_cycle_b": 1.0
                },
                {
                    "name": "Cycle Loss Only",
                    "description": "Only cycle loss, no GAN loss or identity loss",
                    "lambda_A": 10.0,
                    "lambda_B": 10.0,
                    "lambda_identity": 0.0,
                    "lambda_gan_a": 0.0,
                    "lambda_gan_b": 0.0,
                    "lambda_cycle_a": 1.0,
                    "lambda_cycle_b": 1.0
                },
                {
                    "name": "Half Cycle Weight",
                    "description": "Reduce cycle loss weight (lambda_A=5, lambda_B=5)",
                    "lambda_A": 5.0,
                    "lambda_B": 5.0,
                    "lambda_identity": 0.5,
                    "lambda_gan_a": 1.0,
                    "lambda_gan_b": 1.0,
                    "lambda_cycle_a": 0.5,
                    "lambda_cycle_b": 0.5
                },
                {
                    "name": "Double Cycle Weight",
                    "description": "Increase cycle loss weight (lambda_A=20, lambda_B=20)",
                    "lambda_A": 20.0,
                    "lambda_B": 20.0,
                    "lambda_identity": 0.5,
                    "lambda_gan_a": 1.0,
                    "lambda_gan_b": 1.0,
                    "lambda_cycle_a": 2.0,
                    "lambda_cycle_b": 2.0
                }
            ],
            "training_config": {
                "dataset_mode": "unaligned",
                "model": "cycle_gan",
                "netG": "resnet_9blocks",
                "netD": "basic",
                "gan_mode": "lsgan",
                "pool_size": 50,
                "lr": 0.0002,
                "beta1": 0.5,
                "num_threads": 4,
                "batch_size": 1,
                "load_size": 286,
                "crop_size": 256,
                "preprocess": "resize_and_crop",
                "no_flip": False,
                "display_freq": 400,
                "save_latest_freq": 5000,
                "save_epoch_freq": 5,
                "n_epochs": 100,
                "n_epochs_decay": 100,
                "epoch_count": 1,
                "checkpoints_dir": "./checkpoints",
                "results_dir": "./results_ablation"
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created default config: {self.config_path}")

    def run_experiment(self, exp_name, exp_config, dataset_path="./datasets/handbags"):
        """
        Run a single ablation experiment

        Args:
            exp_name: Name of the experiment
            exp_config: Configuration dictionary
            dataset_path: Path to dataset
        """
        print(f"\n{'='*70}")
        print(f"🧪 Running: {exp_name}")
        print(f"{'='*70}")

        # Create experiment directory
        results_dir = self.config["training_config"]["results_dir"]
        exp_dir = os.path.join(results_dir, exp_name.replace(" ", "_").lower())
        os.makedirs(exp_dir, exist_ok=True)

        # Build command
        cmd = [
            "python", "train.py",
            f"--dataroot={dataset_path}",
            f"--name={exp_name.replace(' ', '_').lower()}",
            f"--checkpoints_dir={self.config['training_config']['checkpoints_dir']}",
            "--model=cycle_gan",
            "--dataset_mode=unaligned",
            f"--lambda_A={exp_config['lambda_A']}",
            f"--lambda_B={exp_config['lambda_B']}",
            f"--lambda_identity={exp_config['lambda_identity']}",
            f"--n_epochs={self.config['training_config']['n_epochs']}",
            f"--n_epochs_decay={self.config['training_config']['n_epochs_decay']}",
            f"--batch_size={self.config['training_config']['batch_size']}",
            "--continue_train",  # Enable to continue from checkpoint if exists
        ]

        print(f"📝 Command: {' '.join(cmd)}")
        print(f"📊 Config: {json.dumps(exp_config, indent=2)}")

        # Save experiment config
        config_file = os.path.join(exp_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(exp_config, f, indent=2)

        # Run training
        try:
            result = subprocess.run(cmd, cwd=os.getcwd())
            if result.returncode == 0:
                print(f"✅ {exp_name} completed successfully!")
                self.results[exp_name] = {"status": "success", "config": exp_config}
            else:
                print(f"❌ {exp_name} failed with return code {result.returncode}")
                self.results[exp_name] = {"status": "failed", "config": exp_config}
        except Exception as e:
            print(f"❌ Error running experiment: {e}")
            self.results[exp_name] = {"status": "error", "config": exp_config, "error": str(e)}

    def run_all_experiments(self, dataset_path="./datasets/handbags"):
        """Run all ablation experiments"""
        print("\n" + "="*70)
        print("🎬 Starting Ablation Study")
        print("="*70)

        # Run baseline
        print("\n📊 Running BASELINE experiment...")
        self.run_experiment(
            self.config["base_experiment"]["name"],
            self.config["base_experiment"],
            dataset_path
        )

        # Run ablation experiments
        for exp in self.config["experiments"]:
            self.run_experiment(exp["name"], exp, dataset_path)

    def generate_report(self):
        """Generate ablation study report"""
        report_path = os.path.join(
            self.config["training_config"]["results_dir"],
            f"ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("# CycleGAN Ablation Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Experiment Summary\n\n")
            f.write("| Experiment | Status | lambda_A | lambda_B | lambda_identity |\n")
            f.write("|----------|--------|----------|----------|----------------|\n")

            for exp_name, exp_result in self.results.items():
                config = exp_result.get("config", {})
                status = exp_result.get("status", "unknown")
                f.write(
                    f"| {exp_name} | {status} | "
                    f"{config.get('lambda_A', 'N/A')} | "
                    f"{config.get('lambda_B', 'N/A')} | "
                    f"{config.get('lambda_identity', 'N/A')} |\n"
                )

            f.write("\n## Analysis\n\n")
            f.write("### Key Findings\n\n")
            f.write("- Compare model outputs across experiments\n")
            f.write("- Analyze loss curves for each configuration\n")
            f.write("- Measure image quality metrics (FID, IS, LPIPS)\n\n")

            f.write("### Recommendations\n\n")
            f.write("Based on the ablation study results:\n")
            f.write("1. Document which loss components are critical for convergence\n")
            f.write("2. Identify which components contribute most to image quality\n")
            f.write("3. Recommend optimal hyperparameter settings\n")

        print(f"\n📄 Report saved to: {report_path}")
        return report_path

    def visualize_results(self):
        """Create visualization of ablation results"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Lambda values
            exp_names = list(self.results.keys())
            lambda_a_values = [self.results[e]["config"].get("lambda_A", 0) for e in exp_names]
            lambda_b_values = [self.results[e]["config"].get("lambda_B", 0) for e in exp_names]

            x = np.arange(len(exp_names))
            width = 0.35

            axes[0].bar(x - width/2, lambda_a_values, width, label="lambda_A", alpha=0.8)
            axes[0].bar(x + width/2, lambda_b_values, width, label="lambda_B", alpha=0.8)
            axes[0].set_xlabel("Experiment")
            axes[0].set_ylabel("Lambda Value")
            axes[0].set_title("Cycle Loss Weights Across Experiments")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([e.replace(" ", "\n") for e in exp_names], fontsize=8)
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.3)

            # Plot 2: Identity loss
            lambda_idt_values = [self.results[e]["config"].get("lambda_identity", 0) for e in exp_names]

            axes[1].bar(exp_names, lambda_idt_values, alpha=0.8, color="orange")
            axes[1].set_xlabel("Experiment")
            axes[1].set_ylabel("Identity Loss Weight")
            axes[1].set_title("Identity Loss Weights Across Experiments")
            axes[1].set_xticklabels([e.replace(" ", "\n") for e in exp_names], fontsize=8)
            axes[1].grid(axis="y", alpha=0.3)

            plt.tight_layout()

            viz_path = os.path.join(
                self.config["training_config"]["results_dir"],
                f"ablation_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {viz_path}")

        except ImportError:
            print("⚠️  matplotlib not available for visualization")


def main():
    parser = argparse.ArgumentParser(description="CycleGAN Ablation Study")
    parser.add_argument(
        "--config",
        type=str,
        default="ablation_configs.json",
        help="Path to ablation study configuration file"
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="./datasets/handbags",
        help="Path to dataset"
    )
    parser.add_argument(
        "--gen-config",
        action="store_true",
        help="Generate default configuration and exit"
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all experiments in config"
    )

    args = parser.parse_args()

    # Initialize ablation study
    study = AblationStudy(args.config)

    if args.gen_config:
        print("✅ Configuration generated!")
        return

    if args.list_experiments:
        print("\n📋 Experiments in configuration:\n")
        print(f"✅ {study.config['base_experiment']['name']}")
        print(f"   {study.config['base_experiment']['description']}\n")
        for i, exp in enumerate(study.config["experiments"], 1):
            print(f"{i}. {exp['name']}")
            print(f"   {exp['description']}\n")
        return

    # Run ablation study
    print("\n🚀 Starting Ablation Study Framework\n")
    study.run_all_experiments(args.dataroot)

    # Generate report
    study.generate_report()

    # Visualize results
    study.visualize_results()

    print("\n✅ Ablation study completed!")


if __name__ == "__main__":
    main()

"""
Visualization Utilities for CNN Models
Create beautiful plots and charts for model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CNNVisualizer:
    """Visualization toolkit for CNN models"""
    
    def __init__(self, save_dir: str = './plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(
        self,
        history: Dict,
        model_name: str,
        save: bool = True
    ):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history: Dict with 'train_losses', 'val_losses', 'train_accs', 'val_accs'
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'{model_name}_training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'📊 Plot saved to {save_path}')
        
        plt.show()
    
    def plot_model_comparison(
        self,
        comparison_results: Dict,
        metric: str = 'parameters',
        save: bool = True
    ):
        """
        Plot comparison of different models
        
        Args:
            comparison_results: Results from ModelPerformanceAnalyzer.compare_models()
            metric: Which metric to compare ('parameters', 'memory', 'inference_time', 'flops')
            save: Whether to save the plot
        """
        # Extract data
        models = []
        values = []
        
        metric_mapping = {
            'parameters': ('parameters', 'total'),
            'memory': ('memory', 'total_mb'),
            'inference_time': ('inference_time', 'mean_ms'),
            'flops': ('flops', None)
        }
        
        if metric not in metric_mapping:
            raise ValueError(f"Metric must be one of {list(metric_mapping.keys())}")
        
        primary_key, secondary_key = metric_mapping[metric]
        
        for model_name, data in comparison_results.items():
            if model_name == 'rankings':
                continue
            models.append(model_name)
            if secondary_key:
                values.append(data[primary_key][secondary_key])
            else:
                values.append(data[primary_key])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by values
        sorted_pairs = sorted(zip(models, values), key=lambda x: x[1])
        models_sorted, values_sorted = zip(*sorted_pairs)
        
        # Create bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(models_sorted)))
        bars = ax.barh(models_sorted, values_sorted, color=colors, edgecolor='black', linewidth=1.5)
        
        # Labels
        metric_labels = {
            'parameters': 'Parameters (M)',
            'memory': 'Memory (MB)',
            'inference_time': 'Inference Time (ms)',
            'flops': 'FLOPs (G)'
        }
        
        # Scale values for display
        if metric == 'parameters':
            values_display = [v/1e6 for v in values_sorted]
            xlabel = metric_labels[metric]
        elif metric == 'flops':
            values_display = [v/1e9 for v in values_sorted]
            xlabel = metric_labels[metric]
        else:
            values_display = values_sorted
            xlabel = metric_labels[metric]
        
        # Add value labels on bars
        for bar, value in zip(bars, values_display):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(f'CNN Models Comparison - {metric_labels[metric]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'comparison_{metric}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'📊 Plot saved to {save_path}')
        
        plt.show()
    
    def plot_radar_comparison(
        self,
        comparison_results: Dict,
        models_to_compare: Optional[List[str]] = None,
        save: bool = True
    ):
        """
        Create radar chart comparing multiple models on different metrics
        
        Args:
            comparison_results: Results from ModelPerformanceAnalyzer.compare_models()
            models_to_compare: List of model names to compare (max 5)
            save: Whether to save the plot
        """
        if models_to_compare is None:
            models_to_compare = [m for m in comparison_results.keys() if m != 'rankings'][:5]
        
        # Normalize metrics (lower is better, so we invert)
        metrics = ['Parameters', 'Memory', 'Inference Time', 'FLOPs']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_to_compare)))
        
        for idx, model_name in enumerate(models_to_compare):
            if model_name not in comparison_results:
                continue
            
            data = comparison_results[model_name]
            
            # Get values and normalize (inverse because lower is better)
            values = [
                1.0 / (data['parameters']['total'] / 1e6),  # Smaller params = better
                1.0 / data['memory']['total_mb'],  # Less memory = better
                1.0 / data['inference_time']['mean_ms'],  # Faster = better
                1.0 / (data['flops'] / 1e9)  # Fewer FLOPs = better
            ]
            
            # Normalize to 0-1 scale
            values = [v / max(values) for v in values]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('CNN Models - Multi-Metric Comparison\n(Larger area = Better overall)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'radar_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'📊 Plot saved to {save_path}')
        
        plt.show()
    
    def plot_efficiency_frontier(
        self,
        comparison_results: Dict,
        save: bool = True
    ):
        """
        Plot accuracy vs efficiency trade-off (Pareto frontier)
        
        Args:
            comparison_results: Results from ModelPerformanceAnalyzer.compare_models()
            save: Whether to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = []
        params = []
        memory = []
        inference = []
        flops = []
        
        for model_name, data in comparison_results.items():
            if model_name == 'rankings':
                continue
            models.append(model_name)
            params.append(data['parameters']['total'] / 1e6)
            memory.append(data['memory']['total_mb'])
            inference.append(data['inference_time']['mean_ms'])
            flops.append(data['flops'] / 1e9)
        
        # Plot 1: Parameters vs Inference Time
        ax1.scatter(params, inference, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            ax1.annotate(model, (params[i], inference[i]), fontsize=9, ha='right')
        ax1.set_xlabel('Parameters (M)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Model Size vs Speed', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory vs Inference Time
        ax2.scatter(memory, inference, s=200, alpha=0.6, c=range(len(models)), cmap='plasma')
        for i, model in enumerate(models):
            ax2.annotate(model, (memory[i], inference[i]), fontsize=9, ha='right')
        ax2.set_xlabel('Memory (MB)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Memory vs Speed', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FLOPs vs Parameters
        ax3.scatter(flops, params, s=200, alpha=0.6, c=range(len(models)), cmap='coolwarm')
        for i, model in enumerate(models):
            ax3.annotate(model, (flops[i], params[i]), fontsize=9, ha='right')
        ax3.set_xlabel('FLOPs (G)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Parameters (M)', fontsize=11, fontweight='bold')
        ax3.set_title('Computation vs Model Size', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall efficiency score
        # Efficiency = 1 / (normalized_params + normalized_inference + normalized_memory)
        max_params = max(params)
        max_inference = max(inference)
        max_memory = max(memory)
        
        efficiency = [
            1.0 / (p/max_params + i/max_inference + m/max_memory)
            for p, i, m in zip(params, inference, memory)
        ]
        
        sorted_pairs = sorted(zip(models, efficiency), key=lambda x: x[1], reverse=True)
        models_sorted, efficiency_sorted = zip(*sorted_pairs)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models_sorted)))
        bars = ax4.barh(models_sorted, efficiency_sorted, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, efficiency_sorted):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('Efficiency Score (Higher is Better)', fontsize=11, fontweight='bold')
        ax4.set_title('Overall Efficiency Ranking', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('CNN Models - Efficiency Trade-offs', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'efficiency_frontier.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'📊 Plot saved to {save_path}')
        
        plt.show()
    
    def plot_architecture_timeline(
        self,
        comparison_results: Dict,
        save: bool = True
    ):
        """
        Plot CNN architectures on a timeline showing evolution
        
        Args:
            comparison_results: Results from ModelPerformanceAnalyzer.compare_models()
            save: Whether to save the plot
        """
        # Model years (from literature)
        model_years = {
            'LeNet-5': 1998, 'AlexNet': 2012, 'VGG-16': 2014,
            'GoogLeNet': 2014, 'Inception-v3': 2015, 'ResNet-50': 2015,
            'ResNet-18': 2015, 'MobileNet': 2017, 'DenseNet-121': 2017,
            'EfficientNet-B0': 2019, 'ViT-Base': 2020
        }
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        models = []
        years = []
        params = []
        
        for model_name, data in comparison_results.items():
            if model_name == 'rankings':
                continue
            if 'model_info' in data and 'name' in data['model_info']:
                display_name = data['model_info']['name']
            else:
                display_name = model_name
            
            if display_name in model_years:
                models.append(display_name)
                years.append(model_years[display_name])
                params.append(data['parameters']['total'] / 1e6)
        
        # Create scatter plot
        sizes = [min(p * 5, 1000) for p in params]  # Scale for visibility
        scatter = ax.scatter(years, params, s=sizes, alpha=0.6, c=years,
                           cmap='viridis', edgecolors='black', linewidth=2)
        
        # Add labels
        for model, year, param in zip(models, years, params):
            ax.annotate(model, (year, param), fontsize=10, ha='center',
                       xytext=(0, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax.set_ylabel('Parameters (Millions)', fontsize=13, fontweight='bold')
        ax.set_title('Evolution of CNN Architectures (1998-2020)',
                    fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'architecture_timeline.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'📊 Plot saved to {save_path}')
        
        plt.show()
    
    def create_comparison_table(
        self,
        comparison_results: Dict,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Create a formatted comparison table
        
        Args:
            comparison_results: Results from ModelPerformanceAnalyzer.compare_models()
            save: Whether to save as CSV
        
        Returns:
            pandas DataFrame with comparison data
        """
        data = []
        
        for model_name, result in comparison_results.items():
            if model_name == 'rankings':
                continue
            
            row = {
                'Model': model_name,
                'Parameters (M)': result['parameters']['total'] / 1e6,
                'Memory (MB)': result['memory']['total_mb'],
                'Inference (ms)': result['inference_time']['mean_ms'],
                'Inference Std (ms)': result['inference_time']['std_ms'],
                'FLOPs (G)': result['flops'] / 1e9,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.round(2)
        df = df.sort_values('Inference (ms)')
        
        if save:
            save_path = os.path.join(self.save_dir, 'comparison_table.csv')
            df.to_csv(save_path, index=False)
            print(f'📊 Table saved to {save_path}')
        
        return df


# Example usage
if __name__ == '__main__':
    print("CNN Visualization Utilities")
    print("Usage examples:")
    print("""
    from visualization_utils import CNNVisualizer
    import json
    
    # Load comparison results
    with open('comparison_results.json', 'r') as f:
        results = json.load(f)
    
    # Create visualizer
    viz = CNNVisualizer(save_dir='./plots')
    
    # Plot comparisons
    viz.plot_model_comparison(results, metric='parameters')
    viz.plot_efficiency_frontier(results)
    viz.plot_radar_comparison(results)
    viz.plot_architecture_timeline(results)
    
    # Create table
    df = viz.create_comparison_table(results)
    print(df)
    """)
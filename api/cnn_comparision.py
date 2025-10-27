"""
CNN Models Comparison API
Unified interface to compare performance of different CNN architectures
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from collections import OrderedDict

# Import all model factories (assuming they're in separate files or same file)
# from lenet import create_lenet
# from alexnet import create_alexnet
# ... etc

class CNNModelRegistry:
    """Registry for all available CNN models"""
    
    def __init__(self):
        self.models = OrderedDict()
        self._register_models()
    
    def _register_models(self):
        """Register all available models"""
        # Note: In practice, import these from individual modules
        # Here we're referencing the factory functions
        self.models['lenet5'] = {
            'factory': 'create_lenet',
            'default_input_size': (1, 32, 32),
            'description': 'LeNet-5 (1998) - First successful CNN'
        }
        self.models['alexnet'] = {
            'factory': 'create_alexnet',
            'default_input_size': (3, 224, 224),
            'description': 'AlexNet (2012) - ImageNet winner'
        }
        self.models['vgg16'] = {
            'factory': 'create_vgg16',
            'default_input_size': (3, 224, 224),
            'description': 'VGG-16 (2014) - Deep uniform architecture'
        }
        self.models['googlenet'] = {
            'factory': 'create_googlenet',
            'default_input_size': (3, 224, 224),
            'description': 'GoogLeNet (2014) - Inception modules'
        }
        self.models['resnet50'] = {
            'factory': 'create_resnet50',
            'default_input_size': (3, 224, 224),
            'description': 'ResNet-50 (2015) - Residual connections'
        }
        self.models['mobilenet'] = {
            'factory': 'create_mobilenet',
            'default_input_size': (3, 224, 224),
            'description': 'MobileNet (2017) - Efficient mobile architecture'
        }
        self.models['densenet121'] = {
            'factory': 'create_densenet121',
            'default_input_size': (3, 224, 224),
            'description': 'DenseNet-121 (2017) - Dense connections'
        }
        self.models['efficientnet_b0'] = {
            'factory': 'create_efficientnet_b0',
            'default_input_size': (3, 224, 224),
            'description': 'EfficientNet-B0 (2019) - Compound scaling'
        }
        self.models['inceptionv3'] = {
            'factory': 'create_inceptionv3',
            'default_input_size': (3, 299, 299),
            'description': 'Inception-v3 (2015) - Factorized convolutions'
        }
        self.models['vit_base'] = {
            'factory': 'create_vit_base',
            'default_input_size': (3, 224, 224),
            'description': 'Vision Transformer (2020) - Pure transformer'
        }
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]


class ModelPerformanceAnalyzer:
    """Analyze and compare CNN model performance"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.registry = CNNModelRegistry()
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def estimate_memory(self, model: nn.Module, input_size: Tuple) -> Dict[str, float]:
        """Estimate memory usage in MB"""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
        
        # Estimate activation memory (rough approximation)
        dummy_input = torch.randn(1, *input_size).to(self.device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        return {
            'parameters_mb': param_memory,
            'buffers_mb': buffer_memory,
            'total_mb': param_memory + buffer_memory
        }
    
    def measure_inference_time(self, model: nn.Module, input_size: Tuple, 
                              num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
        """Measure inference time"""
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Measure
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }
    
    def estimate_flops(self, model: nn.Module, input_size: Tuple) -> int:
        """Estimate FLOPs (rough approximation)"""
        # This is a simplified FLOP counter
        # For accurate counting, use libraries like thop or fvcore
        total_flops = 0
        
        def count_conv2d(m, x, y):
            nonlocal total_flops
            batch_size = x[0].shape[0]
            output_height, output_width = y.shape[2:4]
            kernel_height, kernel_width = m.kernel_size
            in_channels = m.in_channels
            out_channels = m.out_channels
            groups = m.groups
            
            # FLOPs per output element
            flops_per_element = (kernel_height * kernel_width * in_channels / groups) * 2
            total_flops += flops_per_element * output_height * output_width * out_channels * batch_size
        
        def count_linear(m, x, y):
            nonlocal total_flops
            batch_size = x[0].shape[0]
            total_flops += m.in_features * m.out_features * batch_size * 2
        
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(count_conv2d))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(count_linear))
        
        model.eval()
        dummy_input = torch.randn(1, *input_size).to(self.device)
        with torch.no_grad():
            model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    def analyze_model(self, model: nn.Module, input_size: Tuple, 
                     model_name: str = "Unknown") -> Dict:
        """Comprehensive model analysis"""
        print(f"\nAnalyzing {model_name}...")
        
        results = {
            'model_name': model_name,
            'parameters': self.count_parameters(model),
            'memory': self.estimate_memory(model, input_size),
            'inference_time': self.measure_inference_time(model, input_size),
            'flops': self.estimate_flops(model, input_size),
            'input_size': input_size,
            'device': self.device
        }
        
        # Get model-specific info if available
        if hasattr(model, 'get_model_info'):
            results['model_info'] = model.get_model_info()
        
        return results
    
    def compare_models(self, models_dict: Dict[str, Tuple[nn.Module, Tuple]], 
                      save_path: Optional[str] = None) -> Dict:
        """
        Compare multiple models
        
        Args:
            models_dict: Dictionary of {model_name: (model, input_size)}
            save_path: Optional path to save results as JSON
        
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for model_name, (model, input_size) in models_dict.items():
            comparison_results[model_name] = self.analyze_model(model, input_size, model_name)
        
        # Add ranking
        comparison_results['rankings'] = self._compute_rankings(comparison_results)
        
        if save_path:
            self._save_results(comparison_results, save_path)
        
        return comparison_results
    
    def _compute_rankings(self, results: Dict) -> Dict:
        """Compute rankings for different metrics"""
        rankings = {
            'fastest_inference': [],
            'fewest_parameters': [],
            'lowest_memory': [],
            'lowest_flops': []
        }
        
        models_data = [(name, data) for name, data in results.items() 
                      if name != 'rankings']
        
        # Fastest inference
        sorted_by_speed = sorted(models_data, 
                                key=lambda x: x[1]['inference_time']['mean_ms'])
        rankings['fastest_inference'] = [(name, data['inference_time']['mean_ms']) 
                                         for name, data in sorted_by_speed]
        
        # Fewest parameters
        sorted_by_params = sorted(models_data,
                                 key=lambda x: x[1]['parameters']['total'])
        rankings['fewest_parameters'] = [(name, data['parameters']['total'])
                                         for name, data in sorted_by_params]
        
        # Lowest memory
        sorted_by_memory = sorted(models_data,
                                 key=lambda x: x[1]['memory']['total_mb'])
        rankings['lowest_memory'] = [(name, data['memory']['total_mb'])
                                     for name, data in sorted_by_memory]
        
        # Lowest FLOPs
        sorted_by_flops = sorted(models_data,
                                key=lambda x: x[1]['flops'])
        rankings['lowest_flops'] = [(name, data['flops'])
                                    for name, data in sorted_by_flops]
        
        return rankings
    
    def _save_results(self, results: Dict, save_path: str):
        """Save results to JSON file"""
        # Convert to JSON-serializable format
        json_results = json.dumps(results, indent=2, default=str)
        with open(save_path, 'w') as f:
            f.write(json_results)
        print(f"\nResults saved to {save_path}")
    
    def print_comparison_table(self, results: Dict):
        """Print formatted comparison table"""
        print("\n" + "="*100)
        print("CNN MODELS PERFORMANCE COMPARISON")
        print("="*100)
        
        # Header
        header = f"{'Model':<20} {'Params (M)':<15} {'Memory (MB)':<15} {'Inference (ms)':<18} {'FLOPs (G)':<15}"
        print(header)
        print("-"*100)
        
        # Data rows
        for model_name, data in results.items():
            if model_name == 'rankings':
                continue
            
            params_m = data['parameters']['total'] / 1e6
            memory_mb = data['memory']['total_mb']
            inference_ms = data['inference_time']['mean_ms']
            flops_g = data['flops'] / 1e9
            
            row = f"{model_name:<20} {params_m:<15.2f} {memory_mb:<15.2f} {inference_ms:<18.3f} {flops_g:<15.2f}"
            print(row)
        
        print("="*100)
        
        # Rankings
        if 'rankings' in results:
            print("\nRANKINGS:")
            print("-"*100)
            
            rankings = results['rankings']
            
            print("\nFastest Inference:")
            for i, (name, time) in enumerate(rankings['fastest_inference'][:5], 1):
                print(f"  {i}. {name}: {time:.3f} ms")
            
            print("\nFewest Parameters:")
            for i, (name, params) in enumerate(rankings['fewest_parameters'][:5], 1):
                print(f"  {i}. {name}: {params/1e6:.2f} M")
            
            print("\nLowest Memory:")
            for i, (name, mem) in enumerate(rankings['lowest_memory'][:5], 1):
                print(f"  {i}. {name}: {mem:.2f} MB")
            
            print("\nLowest FLOPs:")
            for i, (name, flops) in enumerate(rankings['lowest_flops'][:5], 1):
                print(f"  {i}. {name}: {flops/1e9:.2f} G")


# Example usage function
def compare_all_cnn_models(num_classes: int = 1000):
    """
    Compare all available CNN models
    
    Usage:
        results = compare_all_cnn_models(num_classes=10)
    """
    # Import model creation functions (you need to import from individual modules)
    # This is a template - adjust imports based on your file structure
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Dictionary of models to compare
    # Format: {name: (model_instance, input_size)}
    models_to_compare = {
        # 'lenet5': (create_lenet(num_classes=num_classes), (1, 32, 32)),
        # 'alexnet': (create_alexnet(num_classes=num_classes), (3, 224, 224)),
        # 'vgg16': (create_vgg16(num_classes=num_classes), (3, 224, 224)),
        # 'googlenet': (create_googlenet(num_classes=num_classes), (3, 224, 224)),
        # 'resnet50': (create_resnet50(num_classes=num_classes), (3, 224, 224)),
        # 'mobilenet': (create_mobilenet(num_classes=num_classes), (3, 224, 224)),
        # 'densenet121': (create_densenet121(num_classes=num_classes), (3, 224, 224)),
        # 'efficientnet_b0': (create_efficientnet_b0(num_classes=num_classes), (3, 224, 224)),
        # 'inceptionv3': (create_inceptionv3(num_classes=num_classes), (3, 299, 299)),
        # 'vit_base': (create_vit_base(num_classes=num_classes), (3, 224, 224)),
    }
    
    # Run comparison
    results = analyzer.compare_models(models_to_compare, save_path='cnn_comparison_results.json')
    
    # Print results
    analyzer.print_comparison_table(results)
    
    return results


if __name__ == "__main__":
    # Example: Compare all models
    print("CNN Models Comparison API")
    print("To use, uncomment model imports and run: compare_all_cnn_models()")
"""
Complete Usage Example for CNN Models Comparison
This file demonstrates how to use all the models and the comparison API
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import all model creation functions
# Assuming all modules are in the same directory or properly installed
from lenet import create_lenet
from alexnet import create_alexnet
from vggnet import create_vgg16
from googlenet import create_googlenet
from resnet import create_resnet50, create_resnet18
from mobilenet import create_mobilenet
from densenet import create_densenet121
from efficientnet import create_efficientnet_b0
from inceptionv3 import create_inceptionv3
from vit import create_vit_base, create_vit_small

from cnn_comparison_api import ModelPerformanceAnalyzer, CNNModelRegistry


def example_1_single_model_analysis():
    """Example 1: Analyze a single model"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Model Analysis")
    print("="*80)
    
    # Create a ResNet-50 model
    model = create_resnet50(num_classes=10)  # For CIFAR-10
    input_size = (3, 224, 224)
    
    # Analyze the model
    analyzer = ModelPerformanceAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')
    results = analyzer.analyze_model(model, input_size, model_name='ResNet-50')
    
    # Print results
    print(f"\nModel: {results['model_name']}")
    print(f"Total Parameters: {results['parameters']['total']:,}")
    print(f"Memory Usage: {results['memory']['total_mb']:.2f} MB")
    print(f"Inference Time: {results['inference_time']['mean_ms']:.3f} ± {results['inference_time']['std_ms']:.3f} ms")
    print(f"FLOPs: {results['flops']/1e9:.2f} G")


def example_2_compare_all_models():
    """Example 2: Compare all CNN models"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare All CNN Models")
    print("="*80)
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Create all models with the same number of classes
    num_classes = 1000  # ImageNet
    
    models_to_compare = {
        'LeNet-5': (create_lenet(num_classes=num_classes, input_channels=3), (3, 32, 32)),
        'AlexNet': (create_alexnet(num_classes=num_classes), (3, 224, 224)),
        'VGG-16': (create_vgg16(num_classes=num_classes), (3, 224, 224)),
        'GoogLeNet': (create_googlenet(num_classes=num_classes), (3, 224, 224)),
        'ResNet-50': (create_resnet50(num_classes=num_classes), (3, 224, 224)),
        'ResNet-18': (create_resnet18(num_classes=num_classes), (3, 224, 224)),
        'MobileNet': (create_mobilenet(num_classes=num_classes), (3, 224, 224)),
        'DenseNet-121': (create_densenet121(num_classes=num_classes), (3, 224, 224)),
        'EfficientNet-B0': (create_efficientnet_b0(num_classes=num_classes), (3, 224, 224)),
        'Inception-v3': (create_inceptionv3(num_classes=num_classes), (3, 299, 299)),
    }
    
    # Run comparison
    results = analyzer.compare_models(models_to_compare, save_path='full_comparison_results.json')
    
    # Print formatted table
    analyzer.print_comparison_table(results)
    
    return results


def example_3_lightweight_models_comparison():
    """Example 3: Compare lightweight models for mobile deployment"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Lightweight Models for Mobile Deployment")
    print("="*80)
    
    analyzer = ModelPerformanceAnalyzer()
    num_classes = 10  # CIFAR-10
    
    # Compare lightweight models
    lightweight_models = {
        'MobileNet-1.0': (create_mobilenet(num_classes=num_classes, width_multiplier=1.0), (3, 224, 224)),
        'MobileNet-0.75': (create_mobilenet(num_classes=num_classes, width_multiplier=0.75), (3, 224, 224)),
        'MobileNet-0.5': (create_mobilenet(num_classes=num_classes, width_multiplier=0.5), (3, 224, 224)),
        'EfficientNet-B0': (create_efficientnet_b0(num_classes=num_classes), (3, 224, 224)),
        'ResNet-18': (create_resnet18(num_classes=num_classes), (3, 224, 224)),
    }
    
    results = analyzer.compare_models(lightweight_models, save_path='lightweight_comparison.json')
    analyzer.print_comparison_table(results)
    
    print("\n📱 Recommendation for Mobile Deployment:")
    fastest = results['rankings']['fastest_inference'][0]
    smallest = results['rankings']['fewest_parameters'][0]
    print(f"  - Fastest: {fastest[0]} ({fastest[1]:.3f} ms)")
    print(f"  - Smallest: {smallest[0]} ({smallest[1]/1e6:.2f}M params)")


def example_4_transfer_learning_setup():
    """Example 4: Setup models for transfer learning"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Transfer Learning Setup")
    print("="*80)
    
    # Load pre-trained ResNet-50 (in practice, load actual weights)
    model = create_resnet50(num_classes=1000)
    
    # Modify for transfer learning with new dataset (e.g., 10 classes)
    num_classes_new = 10
    
    # Replace the final layer
    if hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes_new)
    
    # Freeze all layers except the final layer
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    print(f"Model prepared for transfer learning")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")


def example_5_ensemble_models():
    """Example 5: Create an ensemble of models"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Ensemble of CNN Models")
    print("="*80)
    
    num_classes = 10
    
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super(EnsembleModel, self).__init__()
            self.models = nn.ModuleList(models)
        
        def forward(self, x):
            outputs = [model(x) for model in self.models]
            # Average predictions
            return torch.mean(torch.stack(outputs), dim=0)
    
    # Create ensemble
    models = [
        create_resnet18(num_classes=num_classes),
        create_mobilenet(num_classes=num_classes),
        create_efficientnet_b0(num_classes=num_classes)
    ]
    
    ensemble = EnsembleModel(models)
    
    analyzer = ModelPerformanceAnalyzer()
    results = analyzer.analyze_model(ensemble, (3, 224, 224), 'Ensemble-3-Models')
    
    print(f"Ensemble Performance:")
    print(f"  Total Parameters: {results['parameters']['total']:,}")
    print(f"  Inference Time: {results['inference_time']['mean_ms']:.3f} ms")


def example_6_custom_training_loop():
    """Example 6: Training example with one model"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Training Loop Template")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_resnet18(num_classes=10).to(device)
    
    # Dummy data for demonstration
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training setup complete:")
    print(f"  Device: {device}")
    print(f"  Model: ResNet-18")
    print(f"  Loss: Cross Entropy")
    print(f"  Optimizer: Adam (lr=0.001)")
    
    # Training loop template
    print("\nTraining loop template:")
    print("""
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    """)


def example_7_model_registry():
    """Example 7: Using the model registry"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Model Registry")
    print("="*80)
    
    registry = CNNModelRegistry()
    
    print("Available models:")
    for i, model_name in enumerate(registry.list_models(), 1):
        info = registry.get_model_info(model_name)
        print(f"{i}. {model_name}: {info['description']}")


def example_8_accuracy_vs_efficiency_tradeoff():
    """Example 8: Analyze accuracy vs efficiency tradeoff"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Accuracy vs Efficiency Trade-off Analysis")
    print("="*80)
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Models representing different trade-offs
    tradeoff_models = {
        'Tiny (MobileNet-0.5)': (create_mobilenet(num_classes=1000, width_multiplier=0.5), (3, 224, 224)),
        'Small (MobileNet-1.0)': (create_mobilenet(num_classes=1000, width_multiplier=1.0), (3, 224, 224)),
        'Medium (ResNet-18)': (create_resnet18(num_classes=1000), (3, 224, 224)),
        'Large (ResNet-50)': (create_resnet50(num_classes=1000), (3, 224, 224)),
        'Extra Large (VGG-16)': (create_vgg16(num_classes=1000), (3, 224, 224)),
    }
    
    results = analyzer.compare_models(tradeoff_models)
    analyzer.print_comparison_table(results)
    
    print("\n💡 Insights:")
    print("  - Tiny models: Best for mobile/edge devices")
    print("  - Small models: Good balance for most applications")
    print("  - Medium models: Higher accuracy, acceptable speed")
    print("  - Large models: Maximum accuracy, resource-intensive")


def main():
    """Run all examples"""
    print("\n")
    print("█" * 80)
    print("  CNN MODELS LIBRARY - COMPREHENSIVE EXAMPLES")
    print("█" * 80)
    
    examples = [
        ("Single Model Analysis", example_1_single_model_analysis),
        ("Compare All Models", example_2_compare_all_models),
        ("Lightweight Models Comparison", example_3_lightweight_models_comparison),
        ("Transfer Learning Setup", example_4_transfer_learning_setup),
        ("Ensemble Models", example_5_ensemble_models),
        ("Custom Training Loop", example_6_custom_training_loop),
        ("Model Registry", example_7_model_registry),
        ("Accuracy vs Efficiency", example_8_accuracy_vs_efficiency_tradeoff),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "="*80)
    choice = input("\nEnter example number (1-8) or 'all' to run all examples: ").strip()
    
    if choice.lower() == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {str(e)}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice)-1][1]()
    else:
        print("Invalid choice. Running example 1 by default.")
        example_1_single_model_analysis()


if __name__ == "__main__":
    # Quick test - uncomment the example you want to run
    # example_1_single_model_analysis()
    # example_2_compare_all_models()
    # example_7_model_registry()
    
    # Or run the interactive menu
    main()
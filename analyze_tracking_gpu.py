# analyze_gpu_tracking.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_tracking_data(tracking_dir="gpu_tracking"):
    """Analyze GPU tracking data and generate readable report."""
    
    print("\n=== GPU Tracking Analysis ===\n")
    
    # Memory Analysis
    memory_df = pd.read_csv(os.path.join(tracking_dir, "memory_tracking.csv"))
    print("Memory Usage Summary:")
    print(f"Peak allocated memory: {memory_df['allocated_mb'].max():.2f} MB")
    print(f"Peak reserved memory: {memory_df['reserved_mb'].max():.2f} MB")
    print(f"Steps tracked: {len(memory_df)}\n")
    
    # Memory by step type
    print("Average Memory by Step Type:")
    step_avg = memory_df.groupby('tag')['allocated_mb'].mean()
    print(step_avg.to_string())
    print()
    
    # Gradient Analysis
    grad_df = pd.read_csv(os.path.join(tracking_dir, "gradient_tracking.csv"))
    print("\nGradient Analysis:")
    print(f"Parameters tracked: {len(grad_df['param_name'].unique())}")
    
    # Find largest gradients
    large_grads = grad_df[abs(grad_df['max']) > 1.0]
    if not large_grads.empty:
        print("\nLarge Gradients Detected:")
        for _, row in large_grads.iterrows():
            print(f"Parameter: {row['param_name']}")
            print(f"  Max value: {row['max']:.4f}")
            print(f"  At step: {row['tag']}")
    
    # Value Analysis
    value_df = pd.read_csv(os.path.join(tracking_dir, "value_tracking.csv"))
    print("\nValue Analysis:")
    for var_name in value_df['var_name'].unique():
        var_data = value_df[value_df['var_name'] == var_name]
        print(f"\n{var_name}:")
        print(f"  Shape: {var_data['shape'].iloc[0]}")
        print(f"  Mean range: [{var_data['mean'].min():.4f}, {var_data['mean'].max():.4f}]")
        print(f"  Max value: {var_data['max'].max():.4f}")
    
    # Generate plots
    plt.figure(figsize=(12, 6))
    plt.plot(memory_df['allocated_mb'], label='Allocated')
    plt.plot(memory_df['reserved_mb'], label='Reserved')
    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Step')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.savefig(os.path.join(tracking_dir, 'memory_usage.png'))
    plt.close()
    
    if not grad_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=grad_df, x='param_name', y='norm')
        plt.xticks(rotation=45)
        plt.title('Gradient Norms by Parameter')
        plt.tight_layout()
        plt.savefig(os.path.join(tracking_dir, 'gradient_norms.png'))
        plt.close()

if __name__ == "__main__":
    analyze_tracking_data()
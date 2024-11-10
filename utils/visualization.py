# utils/visualization.py
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_counterfactual(original_img, counterfactual_img, difference):
    """Plot original, counterfactual, and their difference."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_img.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(counterfactual_img.permute(1, 2, 0))
    ax2.set_title('Counterfactual Image')
    ax2.axis('off')
    
    # Amplify differences for visibility
    diff_img = (difference * 3).clamp(0, 1)
    ax3.imshow(diff_img.permute(1, 2, 0))
    ax3.set_title('Differences (Amplified)')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig
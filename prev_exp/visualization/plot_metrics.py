import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_fid_vs_clipscore(results, save_path=None):
    """Create a comprehensive FID vs CLIP-Score plot"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract data
    guidance_types = [r["guidance"] for r in results]
    fids = [r["fid"] for r in results]
    clip_scores = [r["clip_score"] for r in results]
    diversities = [r["diversity"] for r in results]

    # Normalize diversity for color mapping
    norm = Normalize(vmin=min(diversities), vmax=max(diversities))
    cmap = plt.get_cmap('viridis')

    # Create scatter plot with color representing diversity
    sc = ax.scatter(
        fids, clip_scores,
        c=diversities,
        cmap=cmap,
        s=150,
        edgecolors='w',
        linewidths=1,
        alpha=0.8
    )

    # Add annotations for guidance types
    for i, (guidance, fid, cs) in enumerate(zip(guidance_types, fids, clip_scores)):
        ax.annotate(
            guidance,
            (fid, cs),
            textcoords="offset points",
            xytext=(10, 5 if i % 2 else -15),
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.6)
        )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Diversity', fontsize=12)

    # Labels and title
    ax.set_xlabel('Fréchet Inception Distance (FID) - Lower is better', fontsize=12)
    ax.set_ylabel('CLIP-Score - Higher is better', fontsize=12)
    ax.set_title('Stable Diffusion Guidance Scheduler Comparison\nFID vs CLIP-Score (Colored by Diversity)',
                 fontsize=14, pad=20)

    # Grid and inverted x-axis
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.invert_xaxis()

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_all_metrics(results, save_path=None):
    """Create a comprehensive plot showing all metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    guidance_types = [r["guidance"] for r in results]
    metrics = ['fid', 'clip_score', 'diversity']
    titles = ['FID (Lower is better)', 'CLIP-Score (Higher is better)', 'Diversity (Higher is better)']
    colors = plt.cm.viridis(np.linspace(0, 1, len(guidance_types)))

    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in results]
        bars = ax.bar(guidance_types, values, color=colors)
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    plt.suptitle('Stable Diffusion Guidance Scheduler Performance Comparison', fontsize=14, y=1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

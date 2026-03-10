import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def _fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    # Don't add data:image/png;base64, here, frontend will do it or we can do it here. Let's do it here for consistency if needed, but existing code does not. The existing code: `result.lesionOverlay` is just the base64 string.
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def get_roc_curve():
    """Generate a mock ROC curve representing the CNN's performance."""
    # Create fake ROC data
    fpr = np.array([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0, 1.0, 1.0])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Customize the plot
    ax.plot(fpr, tpr, color='#0ea5e9', lw=2, label='CNN Model (AUC = 0.978)')
    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#0ea5e9')
    
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return _fig_to_base64(fig)

def get_pr_curve():
    """Generate a mock Precision-Recall curve representing CNN's performance."""
    # Create fake PR data
    recall = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 1.0])
    precision = np.array([1.0, 0.99, 0.98, 0.97, 0.96, 0.93, 0.85, 0.6, 0.2])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(recall, precision, color='#10b981', lw=2, label='CNN Model')
    ax.fill_between(recall, precision, alpha=0.1, color='#10b981')
    
    ax.set_title('Precision-Recall Curve', fontsize=12, pad=15)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim([0.0, 1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return _fig_to_base64(fig)

def get_confusion_matrix():
    """Generate a mock Confusion Matrix representing CNN's performance."""
    # Fake confusion matrix values based on ~94.5% accuracy
    # Total samples: 1000 validation images
    # Actual Normal: 450, Actual Stroke: 550
    cm = np.array([[422, 28],    # TN, FP
                   [ 21, 529]])  # FN, TP
                   
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Use seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=False, ax=ax,
                xticklabels=['Normal', 'Stroke'],
                yticklabels=['Normal', 'Stroke'],
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title('Model Confusion Matrix', fontsize=12, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    
    return _fig_to_base64(fig)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Computes standard classification evaluation metrics for fraud detection.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Calculate Precision-Recall AUC
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist()
    }

def plot_confusion_matrix_fig(cm, title="Confusion Matrix"):
    """
    Renders a clean Matplotlib Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "#0F172A"}
    )
    
    ax.set_xlabel('Predicted Label', color='#F8FAFC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Label', color='#F8FAFC', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Legitimate (0)', 'Fraud (1)'], color='#94A3B8')
    ax.set_yticklabels(['Legitimate (0)', 'Fraud (1)'], color='#94A3B8')
    ax.set_title(title, color='#60A5FA', fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison_fig(metrics_dict):
    """
    Renders a grouped bar chart comparing Precision, Recall, F1, and PR-AUC across models.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    models = list(metrics_dict.keys())
    metric_names = ['precision', 'recall', 'f1', 'pr_auc']
    display_names = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']
    
    x = np.arange(len(models))
    width = 0.18
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    
    for i, m_name in enumerate(metric_names):
        vals = [metrics_dict[m][m_name] for m in models]
        rects = ax.bar(x + i*width, vals, width, label=display_names[i], color=colors[i])
        
    ax.set_ylabel('Score', color='#F8FAFC', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance Comparison', color='#60A5FA', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, color='#94A3B8', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='#94A3B8')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')
    
    legend = ax.legend(facecolor='#1E293B', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#F8FAFC')
        
    plt.tight_layout()
    return fig

def plot_fraud_distribution_fig(legit_count, fraud_count):
    """
    Renders a clean donut chart showing Legitimate vs Fraud distribution.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    
    labels = ['Legitimate', 'Fraudulent']
    sizes = [legit_count, fraud_count]
    colors = ['#10B981', '#EF4444']
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors,
        textprops=dict(color="#F8FAFC", fontweight='bold'),
        wedgeprops=dict(width=0.4, edgecolor='#0F172A', linewidth=2)
    )
    
    for autotext in autotexts:
        autotext.set_color('#FFFFFF')
        autotext.set_fontsize(11)
        
    ax.set_title("Class Distribution", color='#60A5FA', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

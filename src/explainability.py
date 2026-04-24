import shap

def compute_shap(model, X_sample):
    explainer = shap.Explainer(model)
    return explainer(X_sample)

def save_summary_plot(shap_values, X, path="reports/shap_summary.png"):
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

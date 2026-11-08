# requirements
import shap
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s\n',
    datefmt='%H:%M:%S'
)

class QUEDExplainer():
    def __init__(self, model, feature_names):
        # Adapt model 
        self.model = model
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
        else:
            booster = self.model  
        self.explainer = shap.TreeExplainer(booster, feature_names=feature_names)
        self.feature_names = np.array(feature_names)

    def __call__(self, X):
        return self.explainer(X)

    def get_top_features(self, X, top_n=12, filename=None):
        # Get SHAP values
        shap_values = self.explainer(X)
        # Sort features by mean absolute value of SHAP values
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        ranking_idx = np.argsort(mean_abs)[::-1]

        # Select top_n features for the plot
        if filename is not None:
            top_idx = ranking_idx[:top_n]
            shap_values_top = shap_values[:,top_idx]
            try:
                fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top_idx))))
                shap.plots.beeswarm(shap_values_top, max_display=top_n, ax=ax, show=False)
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            finally:
                plt.close(fig)
        return self.feature_names[ranking_idx]
# src/shap_utils.py
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.pipeline.predict_pipeline import PredictPipeline, engineer_features, FEATURE_COLS

_pipeline = PredictPipeline()

class ShapExplainer:
    def __init__(self, model=None, preprocessor=None):
        self.model = model or _pipeline.model
        self.preprocessor = preprocessor or _pipeline.preprocessor
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, mix_dict: dict):
        """Return (shap_values array, scaled input array)."""
        df = pd.DataFrame([mix_dict])
        df_eng = engineer_features(df)
        scaled = self.preprocessor.transform(df_eng)
        shap_values = self.explainer.shap_values(scaled)
        return shap_values, scaled

    def waterfall_plot(self, mix_dict: dict):
        """Return (matplotlib figure, shap_values array)."""
        shap_values, scaled = self.explain(mix_dict)   # uses explain()
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0],
                             base_values=self.explainer.expected_value,
                             data=scaled[0],
                             feature_names=FEATURE_COLS),
            show=False
        )
        plt.tight_layout()
        return fig, shap_values

    def reasoning_text(self, shap_values, scaled):
        """Plain-English summary of top SHAP contributions."""
        df_contrib = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'SHAP': shap_values[0]
        }).sort_values('SHAP', ascending=False)

        top_positive = df_contrib.head(2)
        top_negative = df_contrib.tail(2)

        reasoning = "**🧠 What drives this prediction:**\n\n"
        for _, row in top_positive.iterrows():
            reasoning += f"- **{row['Feature']}** strongly increases strength (+{row['SHAP']:.2f} MPa)\n"
        for _, row in top_negative.iterrows():
            reasoning += f"- **{row['Feature']}** reduces strength ({row['SHAP']:.2f} MPa)\n"

        reasoning += "\n*Larger absolute SHAP value = bigger impact.*"
        return reasoning


# ------------------------------------------------------------
# Backward-compatible function (still available)
# ------------------------------------------------------------
def explain_with_shap(mix_dict: dict):
    """
    Returns (shap_values, matplotlib figure) – computes twice, kept for legacy.
    Use explain_with_shap_fast for better performance.
    """
    explainer_instance = ShapExplainer()
    shap_values, _ = explainer_instance.explain(mix_dict)
    fig, _ = explainer_instance.waterfall_plot(mix_dict)
    return shap_values, fig

# ------------------------------------------------------------
# Optimised single‑pass function for the API
# ------------------------------------------------------------
def explain_with_shap_fast(mix_dict: dict):
    """
    Returns (shap_values, matplotlib figure) – computes SHAP once only.
    """
    explainer_instance = ShapExplainer()
    shap_values, scaled = explainer_instance.explain(mix_dict)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0],
                         base_values=explainer_instance.explainer.expected_value,
                         data=scaled[0],
                         feature_names=FEATURE_COLS),
        show=False
    )
    plt.tight_layout()
    return shap_values, fig
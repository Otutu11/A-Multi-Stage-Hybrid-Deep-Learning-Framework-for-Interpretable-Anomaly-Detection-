import shap

def explain_model(model, background_data, test_instance):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_instance)
    shap.summary_plot(shap_values, test_instance)

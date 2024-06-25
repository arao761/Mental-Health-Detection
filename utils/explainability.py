import shap
import lime
import lime.lime_tabular

def explain_with_shap(model, data):
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data)

def explain_with_lime(model, data):
    explainer = lime.lime_tabular.LimeTabularExplainer(data)
    explanation = explainer.explain_instance(data[0], model.predict)
    explanation.show_in_notebook()

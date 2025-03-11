from functools import partial

import snip_h5py as snipH5
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
# reimport the data
folder_data = Path("../LBMC_marker_less_processing/Comparaison_2D")
dict_final = snipH5.load_dictionary_from_hdf(folder_data / "comparaison_final.h5")
dict_mean_std = snipH5.load_dictionary_from_hdf( folder_data/"mean_std.h5")

list_model = list(dict_final.keys())
camera_configurations = list(dict_final[list_model[0]].keys())
key_points = list(dict_final[list_model[0]][camera_configurations[0]].keys())

value_to_plot_list = ["norm"]#, "Y", "Z","norm"]

data = []
plot_outliers = False
list_config_to_remove = []
nb_keypoints = len(key_points)
key_points = ['SJC', 'EJC', 'WJC', 'HMJC']

for ind_point,points in enumerate(key_points):
    percentage_data = []
    for orientation_camera in camera_configurations:
        for model in list_model:
            for value in dict_final[model][orientation_camera][points]:
                data.append([model, orientation_camera, points, value])

    # Create DataFrames
    df = pd.DataFrame(data, columns=['Model', 'orientation_camera', 'Joint', 'Value'])


df_clean = df.dropna(subset=['Value'])
model_formula = "Value ~ C(Model) + C(orientation_camera) + C(Joint) + C(Model):C(orientation_camera)"
anova_model = ols(model_formula, df_clean).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# effect size
ss_model = anova_table['sum_sq']['C(Model)']
ss_error = anova_table['sum_sq']['Residual']
partial_eta_squared = ss_model / (ss_model + ss_error)
print(f"Partial eta squared for model: {partial_eta_squared}")

# Compute COhen s f from partial eta squared
cohens_f = np.sqrt(partial_eta_squared/(1-partial_eta_squared))
print(f"Cohen's f for model: {cohens_f}")



df_clean = df.dropna(subset=['Value'])
model_formula = "Value ~ C(Model) + C(orientation_camera) + C(Model):C(orientation_camera)"
anova_model = ols(model_formula, df_clean).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# effect size
ss_model = anova_table['sum_sq']['C(Model)']
ss_error = anova_table['sum_sq']['Residual']
partial_eta_squared = ss_model / (ss_model + ss_error)
print(f"Partial eta squared for model: {partial_eta_squared}")

ss_model = anova_table['sum_sq']['C(orientation_camera)']
ss_error = anova_table['sum_sq']['Residual']
partial_eta_squared = ss_model / (ss_model + ss_error)
print(f"Partial eta squared for name config: {partial_eta_squared}")

# Compute COhen s f from partial eta squared
cohens_f = np.sqrt(partial_eta_squared/(1-partial_eta_squared))
print(f"Cohen's f for model: {cohens_f}")


# def bootstrap_partial_eta_sq(data, formula, effect, n_boot=1000, random_state=42):
#     np.random.seed(random_state)
#     boot_eta_sq = []
#     for _ in range(n_boot):
#         # Sample with replacement (you might choose to sample a subset if data is huge)
#         sample = data.sample(frac=1, replace=True)
#         model = ols(formula, data=sample).fit()
#         anova_tab = sm.stats.anova_lm(model, typ=2)
#         # Compute partial eta squared for the specified effect
#         ss_effect = anova_tab.loc[effect, 'sum_sq']
#         ss_error = anova_tab.loc['Residual', 'sum_sq']
#         eta_sq = ss_effect / (ss_effect + ss_error)
#         boot_eta_sq.append(eta_sq)
#     # Get the 95% CI from the bootstrap distribution
#     lower = np.percentile(boot_eta_sq, 2.5)
#     upper = np.percentile(boot_eta_sq, 97.5)
#     return lower, upper
#
# formula = 'error ~ C(Model) * C(orientation_camera)'
# lower_ci, upper_ci = bootstrap_partial_eta_sq(df_clean, formula, 'C(Model)', n_boot=1000)
# print(f"95% CI for Partial Eta Squared (Model): [{lower_ci:.3f}, {upper_ci:.3f}]")
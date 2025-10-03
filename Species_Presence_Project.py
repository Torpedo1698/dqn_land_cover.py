# logistic_regression_species_display.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns
import os

# -------------------------------
# Ensure output folder exists
# -------------------------------
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 1. Data Generation (Dummy Dataset)
# -------------------------------
np.random.seed(42)
n_samples = 500

temperature = np.random.normal(25, 5, n_samples)
elevation = np.random.normal(300, 100, n_samples)
precipitation = np.random.normal(100, 20, n_samples)
habitat_type = np.random.choice([0,1], size=n_samples)  # 0=forest, 1=grassland

# Species presence (target)
presence_prob = 1 / (1 + np.exp(-(-5 + 0.2*temperature + 0.01*elevation + 0.03*precipitation + 1*habitat_type)))
presence_prob = np.clip(presence_prob, 0.05, 0.95)
presence = np.random.binomial(1, presence_prob)

df = pd.DataFrame({
    "temperature": temperature,
    "elevation": elevation,
    "precipitation": precipitation,
    "habitat_type": habitat_type,
    "presence": presence
})

# -------------------------------
# 2. Train-Test Split (Stratified)
# -------------------------------
X = df[["temperature", "elevation", "precipitation", "habitat_type"]]
y = df["presence"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3. Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# -------------------------------
# 4. Model Evaluation
# -------------------------------
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save metrics
metrics_file = os.path.join(output_folder, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")
    f.write(f"ROC-AUC: {roc_auc:.2f}\n")
    f.write(f"F1 Score: {f1:.2f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

print(f"Metrics saved to {metrics_file}")

# -------------------------------
# 5. ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
roc_path = os.path.join(output_folder, "roc_curve.png")
plt.savefig(roc_path)
plt.show()  # <-- Display inline
plt.close()

# -------------------------------
# 6. Feature Importance
# -------------------------------
importance = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(6,4))
sns.barplot(x=importance.index, y=importance.values)
plt.title("Feature Importance")
plt.ylabel("Coefficient Value")
feat_path = os.path.join(output_folder, "feature_importance.png")
plt.savefig(feat_path)
plt.show()  # <-- Display inline
plt.close()

print(f"ROC Curve saved to {roc_path}")
print(f"Feature Importance saved to {feat_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. Load your data
df = pd.read_csv("biometric_results.csv")

# Convert labels: genuine = 1, impostor = 0
y_true = (df['trial_type'] == 'genuine').astype(int)
y_scores = df['score']

# 2. Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr
roc_auc = auc(fpr, tpr)

# 3. Find Equal Error Rate (EER)
# EER is where FPR approx equals FNR
eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - fnr)))]
eer = fpr[np.nanargmin(np.absolute((fpr - fnr)))]

print(f"--- RESEARCH METRICS ---")
print(f"Area Under Curve (AUC): {roc_auc:.4f}")
print(f"Equal Error Rate (EER): {eer:.4f}")
print(f"Optimal Threshold: {eer_threshold:.4f}")

# 4. Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Accept Rate (FAR)')
plt.ylabel('True Accept Rate (TAR)')
plt.title('Receiver Operating Characteristic (ROC) - VGGFace2 Subset')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("biometric_roc_curve.png")
plt.show()

# 5. Plot Score Distribution
plt.figure(figsize=(8, 6))
df[df['trial_type'] == 'genuine']['score'].hist(alpha=0.5, label='Genuine', bins=20)
df[df['trial_type'] == 'impostor']['score'].hist(alpha=0.5, label='Impostor', bins=20)
plt.axvline(eer_threshold, color='red', linestyle='dashed', linewidth=1, label=f'EER Threshold ({eer_threshold:.2f})')
plt.title('Score Distribution: Genuine vs. Impostor')
plt.xlabel('Cosine Similarity Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("score_distribution.png")
plt.show()
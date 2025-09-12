import pandas as pd
import numpy as np
from evaluate_captions import evaluate_captions
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
gt_file = "data.csv"
models = {
    "BLIP-Fusion": "outputs/blip_fusion/generated_captions.csv",
    "ViT-GPT2": "outputs/vit_gpt2/generated_captions.csv",
    "BLIP-Large": "outputs/blip_large/generated_captions.csv"
}

# Store results
all_results = {}

print("\n================ MODEL COMPARISON ================\n")

for model_name, gen_file in models.items():
    print(f"Evaluating {model_name}...")
    results = evaluate_captions(gt_file, gen_file)
    all_results[model_name] = {
        "METEOR": results['meteor'],
        "CIDEr": results['cider'],
        "ROUGE-L": results['rouge_l'],
        "Precision": results['precision'],
        "Recall": results['recall'],
        "F1": results['f1_score'],
        "Accuracy": results['accuracy']
    }
    print(f"✅ {model_name} done.\n")

# Convert to DataFrame
df_results = pd.DataFrame(all_results).T

# Save results to CSV
df_results.to_csv("outputs/model_comparison_results.csv")
print("\nComparison results saved to outputs/model_comparison_results.csv")

# ---------------- BAR CHART ---------------- #
df_results[["METEOR", "CIDEr", "ROUGE-L"]].plot(kind="bar", figsize=(10,6))
plt.title("Model Comparison (Key Metrics)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.savefig("outputs/model_comparison_bar.png")
plt.close()

# ---------------- RADAR CHART ---------------- #
metrics = df_results.columns
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for model in df_results.index:
    values = df_results.loc[model].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model, linewidth=2)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_title("Radar Chart: Models vs Metrics", size=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
plt.savefig("outputs/model_comparison_radar.png")
plt.close()

# ---------------- HEATMAP ---------------- #
plt.figure(figsize=(8,6))
sns.heatmap(df_results, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Heatmap: Models vs Metrics")
plt.savefig("outputs/model_comparison_heatmap.png")
plt.close()

# ---------------- LINE PLOT ---------------- #
plt.figure(figsize=(10,6))
for model in df_results.index:
    plt.plot(df_results.columns, df_results.loc[model], marker='o', label=model)

plt.title("Line Plot: Performance Across Metrics")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("outputs/model_comparison_line.png")
plt.close()

# ---------------------- 3D Plot (METEOR, CIDEr, ROUGE-L) ---------------------- #
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

xs = df_results["METEOR"]
ys = df_results["CIDEr"]
zs = df_results["ROUGE-L"]

ax.scatter(xs, ys, zs, s=100, c=["red", "blue", "green"], depthshade=True)

# Annotate model names
for i, model_name in enumerate(df_results.index):
    ax.text(xs[i], ys[i], zs[i], model_name, fontsize=9, weight="bold")

ax.set_xlabel("METEOR")
ax.set_ylabel("CIDEr")
ax.set_zlabel("ROUGE-L")
ax.set_title("3D Comparison of Models (METEOR vs CIDEr vs ROUGE-L)")

plt.savefig("outputs/model_comparison_3d.png")
plt.show()
print("✅ Saved 3D plot → outputs/model_comparison_3d.png")

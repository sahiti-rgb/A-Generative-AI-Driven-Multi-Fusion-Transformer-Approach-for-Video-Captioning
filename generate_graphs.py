import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import textwrap
from mpl_toolkits.mplot3d import Axes3D

# Load Data
df = pd.read_csv('caption_evaluation_results.csv')

# Set Style
sns.set(style='whitegrid')
plt.rcParams.update({'figure.autolayout': True})

# ----------------------- 1. Histogram of METEOR Scores ----------------------- #
plt.figure(figsize=(8, 5))
sns.histplot(df['meteor_score'], bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of METEOR Scores')
plt.xlabel('METEOR Score')
plt.ylabel('Frequency')
plt.savefig('meteor_score_distribution.png')
plt.close()

# ----------------------- 2. Histogram of CIDEr Scores ----------------------- #
plt.figure(figsize=(8, 5))
sns.histplot(df['cider_score'], bins=10, kde=True, color='lightgreen', edgecolor='black')
plt.title('Distribution of CIDEr Scores')
plt.xlabel('CIDEr Score')
plt.ylabel('Frequency')
plt.savefig('cider_score_distribution.png')
plt.close()

# ----------------------- 3. Histogram of ROUGE-L Scores ----------------------- #
plt.figure(figsize=(8, 5))
sns.histplot(df['rouge_l'], bins=10, kde=True, color='plum', edgecolor='black')
plt.title('Distribution of ROUGE-L Scores')
plt.xlabel('ROUGE-L Score')
plt.ylabel('Frequency')
plt.savefig('rouge_l_score_distribution.png')
plt.close()

# -------------------- 4. Scatter Plot: METEOR vs CIDEr ---------------------- #
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='meteor_score', y='cider_score', color='coral', s=50)
plt.title('Scatter Plot of METEOR vs CIDEr Scores')
plt.xlabel('METEOR Score')
plt.ylabel('CIDEr Score')
plt.savefig('meteor_vs_cider_scatter.png')
plt.close()

# -------------------- 5. Scatter Plot: METEOR vs ROUGE ---------------------- #
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='meteor_score', y='rouge_l', color='blue', s=50)
plt.title('Scatter Plot of METEOR vs ROUGE-L Scores')
plt.xlabel('METEOR Score')
plt.ylabel('ROUGE-L Score')
plt.savefig('meteor_vs_rouge_scatter.png')
plt.close()

# -------------------- 6. Scatter Plot: ROUGE vs CIDEr ---------------------- #
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='rouge_l', y='cider_score', color='green', s=50)
plt.title('Scatter Plot of ROUGE-L vs CIDEr Scores')
plt.xlabel('ROUGE-L Score')
plt.ylabel('CIDEr Score')
plt.savefig('rouge_vs_cider_scatter.png')
plt.close()

# -------------------- 7. 3D Scatter: METEOR vs CIDEr vs ROUGE --------------- #
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['meteor_score'], df['cider_score'], df['rouge_l'], c='purple', s=50)
ax.set_xlabel('METEOR')
ax.set_ylabel('CIDEr')
ax.set_zlabel('ROUGE-L')
ax.set_title('3D Scatter: METEOR vs CIDEr vs ROUGE-L')
plt.savefig('meteor_cider_rouge_3d.png')
plt.close()

# --------------------- 8. Average Score Bar Chart -------------------------- #
average_scores = df[['meteor_score', 'cider_score', 'rouge_l', 'precision', 'recall', 'f1_score', 'accuracy']].mean()
metrics = average_scores.index

plt.figure(figsize=(10, 5))
sns.barplot(x=metrics, y=average_scores, palette="coolwarm")
plt.title('Average Scores Across All Metrics')
plt.ylabel('Average Score')
plt.xlabel('Metric')
plt.savefig('average_scores_bar.png')
plt.close()

# ---------------------- 9. Correlation Heatmap ------------------------------ #
plt.figure(figsize=(7, 6))
sns.heatmap(df[['meteor_score', 'cider_score', 'rouge_l', 'precision', 'recall', 'f1_score', 'accuracy']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Scores')
plt.savefig('correlation_heatmap.png')
plt.close()

# ------------------ 10. Terminal Table for Qualitative Examples ------------- #
def wrap_text(text, width=40):
    return '\n'.join(textwrap.wrap(str(text), width=width))

# ------------------ Top 3 High-Scoring Captions ------------------ #
top_captions = df.sort_values(by=['meteor_score', 'cider_score', 'rouge_l'], ascending=False).head(3)
print("\n================ TOP 3 HIGH-SCORING CAPTIONS ================")
print(tabulate(
    [
        [
            wrap_text(row['caption']),
            wrap_text(row['generated_caption']),
            f"{row['meteor_score']:.4f}",
            f"{row['cider_score']:.4f}",
            f"{row['rouge_l']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['accuracy']:.4f}"
        ]
        for _, row in top_captions.iterrows()
    ],
    headers=['Ground Truth Caption', 'Generated Caption', 'METEOR', 'CIDEr', 'ROUGE-L',
             'Precision', 'Recall', 'F1', 'Accuracy'],
    tablefmt='grid'
))

# ------------------ Bottom 3 Low-Scoring Captions ------------------ #
low_captions = df.sort_values(by=['meteor_score', 'cider_score', 'rouge_l'], ascending=True).head(3)
print("\n================ BOTTOM 3 LOW-SCORING CAPTIONS ================")
print(tabulate(
    [
        [
            wrap_text(row['caption']),
            wrap_text(row['generated_caption']),
            f"{row['meteor_score']:.4f}",
            f"{row['cider_score']:.4f}",
            f"{row['rouge_l']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['accuracy']:.4f}"
        ]
        for _, row in low_captions.iterrows()
    ],
    headers=['Ground Truth Caption', 'Generated Caption', 'METEOR', 'CIDEr', 'ROUGE-L',
             'Precision', 'Recall', 'F1', 'Accuracy'],
    tablefmt='grid'
))

# ------------------ 11. Print Average Scores in Terminal ------------------ #
print("\n ===== OVERALL AVERAGE SCORES =====")
avg_table = [[metric.upper(), f"{score:.4f}"] for metric, score in average_scores.items()]
print(tabulate(avg_table, headers=['Metric', 'Average Score'], tablefmt='grid'))

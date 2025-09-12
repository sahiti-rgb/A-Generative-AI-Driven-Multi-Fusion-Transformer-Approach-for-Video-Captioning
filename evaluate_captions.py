import pandas as pd
import numpy as np
import re
import math
import collections
from tqdm import tqdm
from rouge_score import rouge_scorer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

def load_captions(gt_file, gen_file):
    """
    Load ground truth and generated captions from CSV files
    """
    gt_df = pd.read_csv(gt_file)
    gen_df = pd.read_csv(gen_file)
    
    # Merge on video_path to ensure alignment
    merged_df = pd.merge(gt_df, gen_df, on='video_path', how='inner')
    
    if len(merged_df) < len(gt_df) or len(merged_df) < len(gen_df):
        print(f"Warning: Only {len(merged_df)} videos matched out of {len(gt_df)} ground truth and {len(gen_df)} generated captions")
    
    return merged_df

def tokenize(text):
    """
    Tokenize + Lemmatize text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [token for token in text.split() if token]
    return [lemmatizer.lemmatize(t) for t in tokens]

def compute_meteor_simple(reference, candidate):
    """
    Simplified METEOR score calculation without using NLTK
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    matches = 0
    for token in cand_tokens:
        if token in ref_tokens:
            matches += 1
            ref_tokens.remove(token)

    precision = matches / len(cand_tokens) if cand_tokens else 0
    recall = matches / len(tokenize(reference)) if tokenize(reference) else 0

    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0

    return f_score

def compute_cider_simple(references, candidates, n=4):
    """
    Compute CIDEr score using TF-IDF weighted cosine similarity
    """
    def compute_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    doc_frequency = {i: collections.Counter() for i in range(1, n+1)}
    
    for ref in references:
        ref_tokens = tokenize(ref)
        for i in range(1, n+1):
            if len(ref_tokens) >= i:
                ngrams = set(compute_ngrams(ref_tokens, i))
                for ngram in ngrams:
                    doc_frequency[i][ngram] += 1

    ref_len = len(references)
    scores = []

    for ref, cand in zip(references, candidates):
        ref_tokens = tokenize(ref)
        cand_tokens = tokenize(cand)
        score = 0.0
        
        for i in range(1, n+1):
            if len(ref_tokens) < i or len(cand_tokens) < i:
                continue

            ref_ngrams = collections.Counter(compute_ngrams(ref_tokens, i))
            cand_ngrams = collections.Counter(compute_ngrams(cand_tokens, i))
            
            ref_tfidf = {}
            cand_tfidf = {}

            for ngram, count in ref_ngrams.items():
                df = doc_frequency[i].get(ngram, 0)
                if df > 0:
                    ref_tfidf[ngram] = count * math.log(ref_len / df)
            
            for ngram, count in cand_ngrams.items():
                df = doc_frequency[i].get(ngram, 0)
                if df > 0:
                    cand_tfidf[ngram] = count * math.log(ref_len / df)
            
            ref_norm = math.sqrt(sum(w*w for w in ref_tfidf.values()))
            cand_norm = math.sqrt(sum(w*w for w in cand_tfidf.values()))

            if ref_norm == 0 or cand_norm == 0:
                continue

            cosine_sim = sum(cand_tfidf.get(ngram, 0) * ref_tfidf.get(ngram, 0) for ngram in cand_tfidf)
            cosine_sim /= (ref_norm * cand_norm)

            score += cosine_sim / n
        
        scores.append(score)
    
    return scores

def evaluate_captions(gt_file, gen_file):
    """
    Evaluate captions using METEOR, CIDEr, ROUGE-L, Precision, Recall, F1-score, and Accuracy
    """
    print(f"Loading captions from {gt_file} and {gen_file}...")
    data = load_captions(gt_file, gen_file)

    print(f"Evaluating {len(data)} caption pairs...")

    # Compute METEOR scores
    meteor_scores = []
    for ref, cand in tqdm(zip(data['caption'], data['generated_caption']), total=len(data), desc="Computing METEOR"):
        meteor_scores.append(compute_meteor_simple(ref, cand))

    # Compute CIDEr scores
    cider_scores = compute_cider_simple(data['caption'].tolist(), data['generated_caption'].tolist())

    # Initialize ROUGE
    rouge_l_scores = []
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Compute Precision, Recall, F1-score, Accuracy
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    for ref, cand in tqdm(zip(data['caption'], data['generated_caption']), total=len(data), desc="Computing Metrics"):
        ref_tokens = tokenize(ref)
        cand_tokens = tokenize(cand)

        # ROUGE-L
        rouge_l = rouge.score(ref, cand)['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l)

        # Precision, Recall, F1, Accuracy
        tp = sum(1 for token in cand_tokens if token in ref_tokens)
        fp = sum(1 for token in cand_tokens if token not in ref_tokens)
        fn = sum(1 for token in ref_tokens if token not in cand_tokens)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / len(ref_tokens) if ref_tokens else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    # Store results in DataFrame
    data['meteor_score'] = meteor_scores
    data['cider_score'] = cider_scores
    data['rouge_l'] = rouge_l_scores
    data['precision'] = precision_scores
    data['recall'] = recall_scores
    data['f1_score'] = f1_scores
    data['accuracy'] = accuracy_scores

    # Calculate overall scores
    avg_meteor = np.mean(meteor_scores)
    avg_cider = np.mean(cider_scores)
    avg_rouge = np.mean(rouge_l_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    # Print summary statistics
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"METEOR Score:   {avg_meteor:.4f}")
    print(f"CIDEr Score:    {avg_cider:.4f}")
    print(f"ROUGE-L Score:  {avg_rouge:.4f}")
    print(f"Precision:      {avg_precision:.4f}")
    print(f"Recall:         {avg_recall:.4f}")
    print(f"F1 Score:       {avg_f1:.4f}")
    print(f"Accuracy:       {avg_accuracy:.4f}")
    print("-" * 40)

    # Save detailed results
    data.to_csv('caption_evaluation_results.csv', index=False)
    print("\nDetailed results saved to 'caption_evaluation_results.csv'")

    return {
        'meteor': avg_meteor,
        'cider': avg_cider,
        'rouge_l': avg_rouge,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'accuracy': avg_accuracy,
        'detailed_results': data
    }

if __name__ == "__main__":
    gt_file = 'data.csv'
    gen_file = 'generated_captions.csv'
    results = evaluate_captions(gt_file, gen_file)


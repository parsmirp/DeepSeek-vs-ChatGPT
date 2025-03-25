from transformers import pipeline
import pandas as pd

# Load CSV and rename columns
ai_data = pd.read_csv("/Users/parsa/Desktop/BroCode/python/AIdata.csv")
ai_data.columns = ["Question", "Country", "DeepSeek", "ChatGPT"]

# Split data by country and drop last row from each group
america  = ai_data[ai_data["Country"] == "a"].iloc[:-1].copy()
china    = ai_data[ai_data["Country"] == "c"].iloc[:-1].copy()
both     = ai_data[ai_data["Country"] == "b"].iloc[:-1].copy()
neither  = ai_data[ai_data["Country"] == "n"].iloc[:-1].copy()

# Optional: Combine reduced dataset if needed later
ai_data_reduced = pd.concat([america, china, both, neither], ignore_index=True)

# Load sentiment analysis pipeline (with truncation)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    truncation=True  # prevents token length errors
)

# Function to get both label and score
def get_sentiment_labels(texts):
    results = sentiment_analyzer(texts.tolist())
    labels = [res["label"] for res in results]
    scores = [res["score"] for res in results]
    return labels, scores

# Analyze America
labels, scores = get_sentiment_labels(america["ChatGPT"])
america["ChatGPT_Sentiment"] = labels
america["ChatGPT_Score"] = scores

labels, scores = get_sentiment_labels(america["DeepSeek"])
america["DeepSeek_Sentiment"] = labels
america["DeepSeek_Score"] = scores

# Analyze China
labels, scores = get_sentiment_labels(china["ChatGPT"])
china["ChatGPT_Sentiment"] = labels
china["ChatGPT_Score"] = scores

labels, scores = get_sentiment_labels(china["DeepSeek"])
china["DeepSeek_Sentiment"] = labels
china["DeepSeek_Score"] = scores

# Analyze Both
labels, scores = get_sentiment_labels(both["ChatGPT"])
both["ChatGPT_Sentiment"] = labels
both["ChatGPT_Score"] = scores

labels, scores = get_sentiment_labels(both["DeepSeek"])
both["DeepSeek_Sentiment"] = labels
both["DeepSeek_Score"] = scores

# Analyze Neither
labels, scores = get_sentiment_labels(neither["ChatGPT"])
neither["ChatGPT_Sentiment"] = labels
neither["ChatGPT_Score"] = scores

labels, scores = get_sentiment_labels(neither["DeepSeek"])
neither["DeepSeek_Sentiment"] = labels
neither["DeepSeek_Score"] = scores

# AMERICA LABELS
print("America - ChatGPT Sentiment Counts:")
print(america["ChatGPT_Sentiment"].value_counts())
print("America - DeepSeek Sentiment Counts:")
print(america["DeepSeek_Sentiment"].value_counts())


# CHINA LABELS
print("China - ChatGPT Sentiment Counts:")
print(china["ChatGPT_Sentiment"].value_counts())
print("China - DeepSeek Sentiment Counts:")
print(china["DeepSeek_Sentiment"].value_counts())

# BOTH LABELS
print("Both - ChatGPT Sentiment Counts:")
print(both["ChatGPT_Sentiment"].value_counts())
print("Both - DeepSeek Sentiment Counts:")
print(both["DeepSeek_Sentiment"].value_counts())

# NEITHER LABELS
print("Neither - ChatGPT Sentiment Counts:")
print(neither["ChatGPT_Sentiment"].value_counts())
print("Neither - DeepSeek Sentiment Counts:")
print(neither["DeepSeek_Sentiment"].value_counts())

# Print average confidence scores
print("\nAverage Confidence Scores:")
print("America - ChatGPT:", round(america["ChatGPT_Score"].mean(), 4))
print("America - DeepSeek:", round(america["DeepSeek_Score"].mean(), 4))

print("China - ChatGPT:", round(china["ChatGPT_Score"].mean(), 4))
print("China - DeepSeek:", round(china["DeepSeek_Score"].mean(), 4))

print("Both - ChatGPT:", round(both["ChatGPT_Score"].mean(), 4))
print("Both - DeepSeek:", round(both["DeepSeek_Score"].mean(), 4))

print("Neither - ChatGPT:", round(neither["ChatGPT_Score"].mean(), 4))
print("Neither - DeepSeek:", round(neither["DeepSeek_Score"].mean(), 4))

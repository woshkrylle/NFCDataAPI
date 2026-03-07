import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
# Ensure your API key is set in the environment variables as OPENAI_API_KEY
client = None 

# Set to an integer (e.g., 20) to test on a small subset first, or None to run on the full test set.
# Running on the full test set (approx 1725 samples) may insure costs and take time.
LIMIT_SAMPLES = None 

def load_and_preprocess_data(filepath):
    """
    Loads the test set. 
    Prioritizes 'test_split_dataset.csv' (exact notebook split).
    Falls back to 'ths-currdata-10k.csv' and performs the split (same random_state).
    """
    test_csv_path = "test_split_dataset.csv"
    original_csv_path = r"E:\Downloads\NFCDataAPI\ths-currdata-10k.csv"
    
    if os.path.exists(test_csv_path):
        print(f"Loading pre-split test data from {test_csv_path}...")
        df_test = pd.read_csv(test_csv_path)
        X_test = df_test['Tag']
        y_test = df_test['Sensitivity']
        print(f"Test data loaded from export. Size: {len(X_test)}")
        return X_test, y_test
        
    elif os.path.exists(original_csv_path):
        print(f"'{test_csv_path}' not found. Falling back to original file: {original_csv_path}")
        print("Performing train_test_split (random_state=42)...")
        
        try:
            df = pd.read_csv(original_csv_path, on_bad_lines='skip')
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None

        # Rename columns
        df = df.rename(columns={
            df.columns[0]: 'Tag',
            df.columns[1]: 'Sensitivity'
        })

        # Preprocess 'Sensitivity' column
        df['Sensitivity'] = (
            df['Sensitivity']
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

        # Clean data (drop NaN and duplicates)
        df = df.dropna(subset=['Sensitivity'])
        df = df.drop_duplicates(subset='Tag', keep=False)

        # Split data (Same parameters as notebook: test_size=0.2, random_state=42, stratify=y)
        X = df['Tag']
        y = df['Sensitivity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Data loaded and split. Test set size: {len(X_test)}")
        return X_test, y_test
    
    else:
        print(f"Error: Neither '{test_csv_path}' nor '{original_csv_path}' found.")
        return None, None

def classify_batch_with_ai(texts, max_retries=5):
    """
    Sends a batch of texts to the AI model for classification.
    Returns a list of predicted labels and the time taken.
    Retries if the output length doesn't match the input or on API errors.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return ["Error"] * len(texts), [], 0

    try:
        local_client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return ["Error"] * len(texts), [], 0

    start_time = time.time()
    
    # Construct the batch prompt
    prompt_text = "Classify the following NFC Data Payloads as 'High' or 'Low' sensitivity.\n"
    prompt_text += "Return ONLY a list of labels separated by newlines. No numbering, no bullets, no extra text. Example:\nHigh\nLow\nHigh\n\nData to classify:\n"
    
    for i, text in enumerate(texts):
        prompt_text += f"{text}\n"

    messages = [
        {"role": "system", "content": "You are a data sensitivity classifier. You must return exactly one label ('High' or 'Low') for each item provided in the list, in the same order. Do not provide explanations. Do not use numbering."},
        {"role": "user", "content": prompt_text}
    ]

    predictions = []
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  Batch attempt {attempt+1}/{max_retries}...", end='\r')
                time.sleep(1 * attempt) # Exponential backoff

            print(f" DEBUG: Sending request to OpenAI...", end='\r')
            response = local_client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                temperature=1
            )

            if not response or not response.choices:
                raise ValueError(f"Empty or invalid response from API: {response}")

            content = response.choices[0].message.content
            if content is None:
                 raise ValueError("API returned None for message content")
            
            content = content.strip()
            # print(f" DEBUG: Raw content glimpse: {content[:50]}...") # Optional debug
            
            # Parse the response (split by newline)
            raw_lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            temp_predictions = []
            # Robust Parsing
            for line in raw_lines:
                # Remove common numbering/bullets (e.g., "1. High", "- High")
                clean_line = line.lstrip("0123456789.-_•* ").strip().lower()
                
                if "high" in clean_line:
                    temp_predictions.append("High")
                elif "low" in clean_line:
                    temp_predictions.append("Low")
                # If neither found, ignore it if it looks like conversational filler, or append Unknown
                # But here we ideally want 1:1 mapping
            
            # Validation: Check if the number of predictions matches input
            if len(temp_predictions) == len(texts):
                predictions = temp_predictions
                break # Success!
            else:
                print(f"  Attempt {attempt+1} failed: Got {len(temp_predictions)} results for {len(texts)} inputs. Retrying...")
                # Optional: Print glimpse of what went wrong for debugging
                # print(f"DEBUG: Content peek: {content[:100]}...")
        
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}. Retrying...")
            if attempt == max_retries - 1:
                # On final failure, fill with Error to keep alignment
                print("  Maximum retries reached. Marking batch as Error.")
                predictions = ["Error"] * len(texts)

    # Final safety check if loop finished without break and predictions not set
    if not predictions:
         predictions = ["Error"] * len(texts)

    end_time = time.time()
    duration = end_time - start_time
    
    return predictions, messages, duration

def run_classification_evaluation(X_test, y_test, limit_samples=None):
    """
    Runs the GPT-4 classification on the provided X_test and y_test sets.
    
    Args:
        X_test (pd.Series or list): The input text data to classify.
        y_test (pd.Series or list): The true labels ('High'/'Low').
        limit_samples (int, optional): Number of samples to process. Defaults to None (all).
    """
    global client
    # If client is None, try to re-init (e.g. env var set later)
    if client is None:
        try:
             client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except:
             print("Please ensure OPENAI_API_KEY is set.")
             return

    # Apply limit if set
    if limit_samples is not None:
        print(f"Limiting to first {limit_samples} samples for testing.")
        if hasattr(X_test, 'iloc'):
             X_test = X_test.iloc[:limit_samples]
             y_test = y_test.iloc[:limit_samples]
        else:
             X_test = X_test[:limit_samples]
             y_test = y_test[:limit_samples]

    predictions = []
    times = []
    
    print("Starting classification with GPT-4 (Batch Mode)...")
    print(f"Total samples to process: {len(X_test)}")
    
    # Batch processing configuration
    BATCH_SIZE = 50  # Process 50 items per API call
    
    # Iterate through the test set in batches
    X_iter = X_test if isinstance(X_test, list) else X_test.tolist()
    y_iter = y_test if isinstance(y_test, list) else y_test.tolist()

    report_data = []
    total = len(X_iter)

    for i in range(0, total, BATCH_SIZE):
        batch_texts = X_iter[i:i + BATCH_SIZE]
        batch_labels = y_iter[i:i + BATCH_SIZE]
        
        print(f"Processing batch {i // BATCH_SIZE + 1} ({min(i + BATCH_SIZE, total)}/{total})...", end='\r')
        
        batch_preds, messages, duration = classify_batch_with_ai(batch_texts)
        
        # Distribute time across batch items for stats
        avg_time_per_item = duration / len(batch_texts)

        for j, (text, true_label, pred) in enumerate(zip(batch_texts, batch_labels, batch_preds)):
            predictions.append(pred)
            times.append(avg_time_per_item)
            
            report_data.append({
                'Index': i + j,
                'Original Text': text,
                'True Label': true_label,
                'Predicted Label': pred,
                'Duration (s)': avg_time_per_item
            })

    print("\nClassification complete.")

    # Create DataFrame for reporting
    report_df = pd.DataFrame(report_data)
    
    # Save to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"gpt4_classification_report_{timestamp}.csv"
    
    report_df.to_csv(report_filename, index=False)
    print(f"\nFull report saved to '{report_filename}'")
    
    # Print tabulated report (first 10 rows)
    print("\n--- Classification Report (First 10 Rows) ---")
    print(report_df.head(10).to_string(index=False))
    print("-" * 50)

    # Calculate Metrics
    y_pred = np.array(predictions)
    y_true = np.array(y_iter) # Ensure we use the processed y

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    avg_time = np.mean(times)
    total_time = np.sum(times)

    print("\n" + "=" * 50)
    print("GPT-4 Classification Results")
    print("=" * 50)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"F1 Score:      {f1:.4f} (Macro)")
    print(f"Precision:     {precision:.4f} (Macro)")
    print(f"Recall:        {recall:.4f} (Macro)")
    print("-" * 50)
    print(f"Average Time per Call: {avg_time:.4f} seconds")
    print(f"Total Time Taken:      {total_time:.2f} seconds")
    print("=" * 50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'avg_time': avg_time,
        'predictions': predictions
    }

def main():
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    X_test, y_test = load_and_preprocess_data(None)
    
    if X_test is None:
        return

    run_classification_evaluation(X_test, y_test, limit_samples=LIMIT_SAMPLES)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
         print("Please set the OPENAI_API_KEY environment variable.")
    else: 
        main()
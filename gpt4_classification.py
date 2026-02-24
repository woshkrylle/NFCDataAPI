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

def classify_with_gpt4(text):
    """
    Sends a prompt to GPT-4 to classify the text.
    Returns the predicted label, the messages sent, and the time taken.
    """
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": "You are a data sensitivity classifier for NFC Data Payloads. Your task is to determine if the following data entry is 'High' or 'Low' sensitivity. Reply ONLY with 'High' or 'Low'."},
        {"role": "user", "content": f"Data: {text}\n\nClassification:"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0  # Low temperature for consistent output
        )
        prediction = response.choices[0].message.content.strip()
        
        # Normalize prediction to match dataset labels ('High' or 'Low')
        if "High" in prediction:
            prediction = "High"
        elif "Low" in prediction:
            prediction = "Low"
        else:
            prediction = "Unknown" 
            
    except Exception as e:
        print(f"Error classifying text: {e}")
        prediction = "Error"

    end_time = time.time()
    duration = end_time - start_time
    
    return prediction, messages, duration

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
    
    print("Starting classification with GPT-4...")
    print(f"Total samples to process: {len(X_test)}")
    print("This may take a while depending on the dataset size and API rate limits.")

    # Iterate through the test set
    # Handle both Series/DataFrame and raw lists
    total = len(X_test)
    
    # Convert to list if it's pandas series for zipping
    X_iter = X_test if isinstance(X_test, list) else X_test.tolist()
    y_iter = y_test if isinstance(y_test, list) else y_test.tolist()

    report_data = []

    for i, (text, true_label) in enumerate(zip(X_iter, y_iter)):
        print(f"Processing {i+1}/{total}...", end='\r')
        
        pred, messages, duration = classify_with_gpt4(text)
        predictions.append(pred)
        times.append(duration)

        # Store details for report
        # We capture the last message content which corresponds to the user prompt with the data
        user_query = messages[-1]['content'] 
        report_data.append({
            'Index': i,
            'Original Text': text,
            'True Label': true_label,
            'Query Sent': user_query,
            'Predicted Label': pred,
            'Duration (s)': duration
        })

    print("\nClassification complete.")

    # Create DataFrame for reporting
    report_df = pd.DataFrame(report_data)
    
    # Save to CSV
    report_filename = "gpt4_classification_report.csv"
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

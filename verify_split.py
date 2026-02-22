from gpt4_classification import load_and_preprocess_data
import pandas as pd

def verify_partition():
    print("Verifying data partition...")
    X_test, y_test = load_and_preprocess_data(None)
    
    if X_test is None:
        print("Failed to load data.")
        return

    print("\n" + "="*30)
    print("Partition Details")
    print("="*30)
    print(f"Test Set Size: {len(X_test)}")
    print(f"Test Set Distribution:\n{y_test.value_counts()}")
    
    print("\n" + "="*30)
    print("First 5 Samples (Test Set)")
    print("="*30)
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"Label: {y_test.iloc[i]}")
        print(f"Text: {X_test.iloc[i][:100]}...") # Truncate text for display
        print("-" * 20)

if __name__ == "__main__":
    verify_partition()
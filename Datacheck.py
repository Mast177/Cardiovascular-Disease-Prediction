def datacheck(data):
    # Display first 5 rows (testing)
    print("--- First 5 rows of data ---")
    print(data.head())

    # Check data structure
    print("\n--- Information & Missing Values ---")
    print(data.info())

    # Statistical Summary
    print("\n--- Statistical Summary ---")
    print(data.describe())
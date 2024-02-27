def parse_arff_file(file_path):
    """
    Parse the ARFF file to handle possible issues with attribute count and data rows.
    """
    with open(file_path, "r") as file:
        content = file.readlines()

    # Remove comments and empty lines
    content = [
        line.strip()
        for line in content
        if line.strip() and not line.strip().startswith("%")
    ]

    # Find @DATA index
    data_index = next(i for i, line in enumerate(content) if line.upper() == "@DATA")

    # Extract attribute names
    attributes = [
        line.split(" ")[1]
        for line in content[:data_index]
        if line.upper().startswith("@ATTRIBUTE")
    ]

    # Extract data rows, handling potential comma within quotes
    data_rows = [line for line in content[data_index + 1 :]]

    # Process data rows into a list of lists, handling quotes if necessary
    processed_data = []
    for row in data_rows:
        processed_row = []
        within_quote = False
        start_index = 0
        for i, char in enumerate(row):
            if char == '"' and not within_quote:
                within_quote = True
                start_index = i + 1
            elif char == '"' and within_quote:
                within_quote = False
                processed_row.append(row[start_index:i])
                start_index = i + 2  # Skip over "," to the start of the next value
            elif char == "," and not within_quote:
                if i > start_index:
                    processed_row.append(row[start_index:i])
                start_index = i + 1
        if start_index < len(row):
            processed_row.append(row[start_index:])

        # Append processed row to data
        processed_data.append(processed_row)

    # Create DataFrame
    df = pd.DataFrame(processed_data, columns=attributes)

    # Convert columns to appropriate dtypes, if possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue  # Keep as object type if conversion fails

    return df


# Parse the ARFF file
df_corrected = parse_arff_file("/mnt/data/cora.arff")

# Save the corrected DataFrame as a Parquet file
df_corrected.to_parquet("/mnt/data/cora_corrected.parquet", index=False)

# Display the first few rows to confirm correction
df_corrected.head()

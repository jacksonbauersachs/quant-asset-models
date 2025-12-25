import csv
import os


def reverse_bitcoin_data():
    # Define the paths
    input_path = os.path.join('..', '..', 'Bitcoin Data', 'Bitcoin_Historical_Data.csv')
    output_path = '../../Data Sets/Bitcoin Data/Bitcoin_Cleaned_Data.csv'

    # Get absolute paths for debugging
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path)

    print(f"Attempting to read from: {abs_input}")
    print(f"Will write output to: {abs_output}")

    # Verify input file exists
    if not os.path.exists(abs_input):
        print("\nError: Input file not found at the specified location.")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir(os.path.join('..', '..')))
        return

    # Read and reverse the data
    try:
        with open(abs_input, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        with open(abs_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(reversed(rows))

        print("\nSuccess! Data reversed and saved to:", abs_output)

    except Exception as e:
        print("\nError processing files:", e)


if __name__ == "__main__":
    # Change to the script's directory for reliable path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    reverse_bitcoin_data()
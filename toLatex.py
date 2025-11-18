import pandas as pd
import numpy as np
import os


def convert_to_latex_table(df, output_file):
    # Columns to check for maximum values
    max_cols = ['Best Accuracy', 'Precision', 'F1 Score', 'Sensitivity', 'Specificity', 'Final Score']

    # Create a copy of the dataframe for LaTeX conversion
    latex_df = df.copy()

    # Find the maximum value for each column and bold it
    for col in max_cols:
        if col in df.columns:
            max_val = df[col].max()
            if not np.isnan(max_val):  # Check if max_val is a valid number
                latex_df.loc[df[col].idxmax(), col] = f"\\textbf{{{max_val:.2f}}}"
            else:
                latex_df[col] = df[col].round(2).astype(str)

    # Convert to LaTeX table with custom formatting
    latex_table = latex_df.to_latex(index=False, float_format="%.2f",
                                    column_format='|l|' + 'c' * (len(df.columns) - 1) + '|', escape=False)

    # Customize the LaTeX table environment
    latex_code = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{siunitx}\n"
        "\\usepackage{graphicx}\n"  # For resizebox
        "\\begin{document}\n\n"
        "\\begin{table*}[ht]\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\setlength{\\tabcolsep}{9.5pt}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        f"{latex_table}\n"
        "}}\n"
        "\\caption{Chest X-ray Classification Performance Metric Results}\n"
        "\\label{tab:metric-results}\n"
        "\\end{table*}\n\n"
        "\\end{document}"
    )

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(latex_code)


def process_directory(directory_path):
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Process each CSV file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(directory_path, filename)
            output_file = os.path.join(directory_path, f"{os.path.splitext(filename)[0]}.tex")
            df = pd.read_csv(csv_path)
            convert_to_latex_table(df, output_file)
            print(f"Converted {filename} to {output_file}")


if __name__ == "__main__":
    directory_path = "/Users/ayano/Desktop/2class_mediastinal_results"
    process_directory(directory_path)
import sys
import pandas as pd
import numpy as np
import os

def error(msg):
    print("Error:", msg)
    sys.exit(1)

def main():
    # Expecting only weights, impacts, output file
    if len(sys.argv) != 4:
        error("Usage: python topsis.py <Weights> <Impacts> <OutputFile>")

    # Fixed input file name
    input_file = "data.csv"

    weights = sys.argv[1].split(",")
    impacts = sys.argv[2].split(",")
    output_file = sys.argv[3]

    # Check if file exists in same folder
    if not os.path.exists(input_file):
        error("data.csv not found in the current folder")

    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        error(f"Unable to read CSV file: {e}")

    if data.shape[1] < 3:
        error("Input file must contain at least 3 columns")

    matrix = data.iloc[:, 1:]

    # Check numeric values
    if not np.all(matrix.applymap(np.isreal)):
        error("From 2nd column onwards, values must be numeric")

    if len(weights) != matrix.shape[1]:
        error("Number of weights must match number of criteria")

    if len(impacts) != matrix.shape[1]:
        error("Number of impacts must match number of criteria")

    for i in impacts:
        if i not in ['+', '-']:
            error("Impacts must be either + or -")

    weights = np.array(weights, dtype=float)

    # Normalization
    norm = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norm

    # Weighted normalized matrix
    weighted = norm_matrix * weights

    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted.iloc[:, i].max())
            ideal_worst.append(weighted.iloc[:, i].min())
        else:
            ideal_best.append(weighted.iloc[:, i].min())
            ideal_worst.append(weighted.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Distance calculation
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    data["Topsis Score"] = score
    data["Rank"] = score.rank(ascending=False).astype(int)

    data.to_csv(output_file, index=False)
    print("Output saved to", output_file)

if __name__ == "__main__":
    main()

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def get_data():
    """
    Function to load the ATP tennis matches dataset from 2004 to 2024.
    The dataset is hosted on GitHub and contains match statistics.
    """
    # URL for the dataset
    url = "https://github.com/JeffSackmann/tennis_atp/blob/master"

    # Load the dataset, going from 2004 to 2024 (picking only atp_matches_years.csv)
    years = list(range(2004, 2025))
    matches = []
    for year in years:
        file_url = f"{url}/atp_matches_{year}.csv?raw=true"
        df = pd.read_csv(file_url)
        matches.append(df)
    # Concatenate all dataframes into one
    matches_df = pd.concat(matches, ignore_index=True)
    # Reset index
    matches_df.reset_index(drop=True, inplace=True)

    FEATURES = [
        "w_ace",
        "w_1stWon",
        "w_2ndWon",
        "w_svpt",
        "best_of",
        # "l_ace", # Comment out when using the winner model
        # "l_1stWon",
    ]

    TARGET = [
        "ace_diff",
        "1stWon_diff",
        "2ndWon_diff",
        "svpt_diff",
    ]

    matches_df["ace_diff"] = (matches_df["w_ace"] - matches_df["l_ace"]) / matches_df[
        "best_of"
    ]
    matches_df["1stWon_diff"] = (
        matches_df["w_1stWon"] - matches_df["l_1stWon"]
    ) / matches_df["best_of"]
    matches_df["2ndWon_diff"] = (
        matches_df["w_2ndWon"] - matches_df["l_2ndWon"]
    ) / matches_df["best_of"]
    matches_df["svpt_diff"] = (
        matches_df["w_svpt"] - matches_df["l_svpt"]
    ) / matches_df["best_of"]

    X = matches_df[FEATURES]
    y = matches_df[TARGET]

    return X, y


def plot_data(X, y):
    """
    Function to plot the data using seaborn and matplotlib.
    """
    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a pairplot for the features
    sns.pairplot(X)
    plt.show()

    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


X, y = get_data()
# Check the data
print(X.describe())
print(y.describe())

# Plot the data
plot_data(X, y)

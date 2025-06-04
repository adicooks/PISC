import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import chi2_contingency

df = pd.read_csv("philly_shooting_trends.csv")

# Age Analysis
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]

    # Basic stats
    print("\nAge Statistics:")
    print(df['age'].describe())

    print("\n")

    # Histogram of ages
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Age in Shootings")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("age_distribution.png")
    plt.show()

    print("\n")

    # Age grouped into bins
    age_bins = [0, 12, 18, 25, 35, 50, 65, 120]
    age_labels = ['Child (0–12)', 'Teen (13–18)', 'Young Adult (19–25)', 'Adult (26–35)', 'Mid Age (36–50)', 'Older (51–65)', 'Senior (66+)']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

    age_group_counts = df['age_group'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.bar(age_group_counts.index, age_group_counts.values, color='coral')
    plt.title("Shootings by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Shootings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("shootings_by_age_group.png")
    plt.show()

    print("\n")

    # Age trends over time
    if 'Year' in df.columns:
        avg_age_by_year = df.groupby('Year')['age'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(avg_age_by_year.index, avg_age_by_year.values, marker='o', color='darkblue')
        plt.title("Average Age of Shooting Victims by Year")
        plt.xlabel("Year")
        plt.ylabel("Average Age")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("avg_age_by_year.png")
        plt.show()
else:
    print("No 'age' column found in the dataset.")

print("\n")

if 'date_' in df.columns and 'time' in df.columns:
    df['date_'] = pd.to_datetime(df['date_'], errors='coerce')
    df['time'] = pd.to_timedelta(df['time'])
    df['datetime'] = df['date_'] + df['time']
    date_col = 'datetime'
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['YearMonth'] = df[date_col].dt.to_period('M')
else:
    print("No valid date/time columns found! Please check your CSV.")
    date_col = None

if 'Year' in df.columns:
    df = df[df['Year'] != 2025]

summary_stats = df.describe(include='all')
summary_stats.to_csv("summary_statistics.csv")

print("\n")


if 'Year' in df.columns:
    yearly_counts = df.groupby('Year').size()
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-')
    plt.title("Yearly Philadelphia Shootings")
    plt.xlabel("Year")
    plt.ylabel("Number of Shootings")
    plt.grid(True)
    plt.savefig("yearly_shootings.png")
    plt.show()

print("\n")

if 'YearMonth' in df.columns:
    monthly_counts = df.groupby('YearMonth').size()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts.index.astype(str), monthly_counts.values, marker='o', linestyle='-')
    plt.xticks(rotation=45)
    plt.title("Monthly Philadelphia Shootings")
    plt.xlabel("Year-Month")
    plt.ylabel("Number of Shootings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monthly_shootings.png")
    plt.show()

print("\n")

if 'race' in df.columns:
    race_counts = df['race'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(race_counts.index, race_counts.values)
    plt.title("Shootings by Race")
    plt.xlabel("Race")
    plt.ylabel("Number of Shootings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("shootings_by_race.png")
    plt.show()

print("\n")

if 'sex' in df.columns:
    sex_counts = df['sex'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(sex_counts.index, sex_counts.values, color='orange')
    plt.title("Shootings by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Number of Shootings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("shootings_by_sex.png")
    plt.show()

print("\n")

if ('race' in df.columns) and ('Year' in df.columns):
    race_yearly = df.groupby(['Year', 'race']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    for race in race_yearly.columns:
        plt.plot(race_yearly.index, race_yearly[race], marker='o', linestyle='-', label=str(race))
    plt.title("Yearly Shooting Trends by Race")
    plt.xlabel("Year")
    plt.ylabel("Number of Shootings")
    plt.legend(title="Race")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("yearly_shootings_by_race.png")
    plt.show()

print("\n")

if ('sex' in df.columns) and ('Year' in df.columns):
    sex_yearly = df.groupby(['Year', 'sex']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    for s in sex_yearly.columns:
        plt.plot(sex_yearly.index, sex_yearly[s], marker='o', linestyle='-', label=str(s))
    plt.title("Yearly Shooting Trends by Sex")
    plt.xlabel("Year")
    plt.ylabel("Number of Shootings")
    plt.legend(title="Sex")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("yearly_shootings_by_sex.png")
    plt.show()

print("\n")

if ('race' in df.columns) and ('sex' in df.columns):
    contingency_table = pd.crosstab(df['race'], df['sex'])
    print("Contingency Table for Race and Sex:")
    print(contingency_table)

    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-squared test results:")
    print(f"Chi-squared statistic: {chi2:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.15f}")
    if p < 0.05:
        print("There is a statistically significant association between race and sex (p < 0.05).")
    else:
        print("There is no statistically significant association between race and sex (p >= 0.05).")

print("\n")

if date_col:
    df['DayOfWeek'] = df[date_col].dt.day_name()
    day_counts = df['DayOfWeek'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    plt.bar(day_counts.index, day_counts.values, color='green')
    plt.title("Shootings by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Shootings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("shootings_by_dayofweek.png")
    plt.show()

print("\n")

if 'time' in df.columns:
    if date_col:
        df['Hour'] = df[date_col].dt.hour
    else:
        df['Hour'] = df['time'].dt.components['hours']
    hour_counts = df['Hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.bar(hour_counts.index, hour_counts.values, color='purple')
    plt.title("Shootings by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Shootings")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("shootings_by_hour.png")
    plt.show()

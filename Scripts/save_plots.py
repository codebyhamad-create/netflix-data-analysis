import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def save_visualizations():
    # Define directories
    # Assuming script is run from 'scripts' directory
    data_path = '../data/netflix_titles.csv'
    output_dir = '../plots'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {os.path.abspath(output_dir)}")

    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    # --- Data Cleaning & Feature Engineering ---
    # Handle dates
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    # Extract numeric duration for movies (minutes)
    # TV Shows have 'Season' in duration, so we'll set them to NaN for duration_min
    df['duration_min'] = df.loc[df['type'] == 'Movie', 'duration'].str.replace(' min', '', regex=False)
    df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce')

    # Primary Genre (first listed)
    df['primary_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')

    # Set style
    sns.set_theme(style="whitegrid")

    print("Generating plots...")

    # 1. Line Plot: Content Added per Year
    plt.figure(figsize=(12, 6))
    growth = df['year_added'].value_counts().sort_index()
    # Filter out future years or bad data if any
    growth = growth[growth.index < 2025] 
    sns.lineplot(x=growth.index, y=growth.values, marker='o', color='tab:blue', linewidth=2.5)
    plt.title('1. Line Plot: Content Added per Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/1_line_plot_growth.png')
    plt.close()

    # 2. Bar Plot: Top 10 Primary Genres
    plt.figure(figsize=(12, 6))
    top_genres = df['primary_genre'].value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
    plt.title('2. Bar Plot: Top 10 Primary Genres')
    plt.xlabel('Count')
    plt.savefig(f'{output_dir}/2_bar_plot_genres.png')
    plt.close()

    # 3. Histogram: Movie Duration Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['duration_min'].dropna(), bins=30, kde=True, color='tab:purple')
    plt.title('3. Histogram: Distribution of Movie Durations')
    plt.xlabel('Duration (minutes)')
    plt.savefig(f'{output_dir}/3_histogram_duration.png')
    plt.close()

    # 4. Scatter Plot: Release Year vs Year Added
    plt.figure(figsize=(12, 6))
    # Sample if too large for cleaner scatter, but 8k is fine
    sns.scatterplot(data=df, x='release_year', y='year_added', hue='type', alpha=0.6, palette='deep')
    plt.title('4. Scatter Plot: Release Year vs Year Added')
    plt.savefig(f'{output_dir}/4_scatter_plot_years.png')
    plt.close()

    # 5. Box Plot: Release Year by Rating
    plt.figure(figsize=(14, 8))
    top_ratings = df['rating'].value_counts().index[:10] # Top 10 ratings
    df_top_ratings = df[df['rating'].isin(top_ratings)]
    sns.boxplot(data=df_top_ratings, x='rating', y='release_year', palette='Set3', order=top_ratings)
    plt.title('5. Box Plot: Release Year Distribution by Rating')
    plt.savefig(f'{output_dir}/5_box_plot_rating_year.png')
    plt.close()

    # 6. Area Plot: Cumulative Content Added by Type
    plt.figure(figsize=(12, 6))
    ct_year = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
    ct_year = ct_year[ct_year.index >= 2008] # Focus on streaming era
    ct_year.plot(kind='area', stacked=True, alpha=0.5, figsize=(12, 6))
    plt.title('6. Area Plot: Content Added Over Time by Type')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/6_area_plot_type_growth.png')
    plt.close()

    # 7. Pie Chart: Content Type Distribution
    plt.figure(figsize=(8, 8))
    type_counts = df["type"].value_counts()
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title('7. Pie Chart: Content Type Distribution')
    plt.savefig(f'{output_dir}/7_pie_chart_type.png')
    plt.close()

    # 8. Pair Plot: Numerical Variables
    # Select numerical columns
    pair_df = df[['release_year', 'year_added', 'duration_min', 'type']].dropna()
    # Filter outliers for better visualization
    pair_df = pair_df[pair_df['release_year'] > 1990] 
    pp = sns.pairplot(pair_df, hue='type', diag_kind='kde', palette='husl')
    pp.fig.suptitle('8. Pair Plot: Relationships between Numerical Features', y=1.02)
    plt.savefig(f'{output_dir}/8_pair_plot.png')
    plt.close()

    # 9. Heatmap: Content Added by Month and Year
    plt.figure(figsize=(12, 8))
    # Filter for recent years
    df_recent = df[df['year_added'] >= 2010]
    heatmap_data = df_recent.groupby(['year_added', 'month_added']).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=.5, annot=True, fmt='d')
    plt.title('9. Heatmap: Content Added by Month and Year')
    plt.ylabel('Year Added')
    plt.xlabel('Month Added')
    plt.savefig(f'{output_dir}/9_heatmap_activity.png')
    plt.close()
    
    # 10. Stack Plot: Ratings Distribution Over Years
    plt.figure(figsize=(12, 6))
    top_5_ratings = df['rating'].value_counts().head(5).index
    df['rating_grouped'] = df['rating'].apply(lambda x: x if x in top_5_ratings else 'Other')
    
    pivot_table = df.groupby(['year_added', 'rating_grouped']).size().unstack(fill_value=0)
    pivot_table = pivot_table[pivot_table.index >= 2010] # Focus on recent years
    
    plt.stackplot(pivot_table.index, pivot_table.T, labels=pivot_table.columns, alpha=0.8)
    plt.legend(loc='upper left')
    plt.title('10. Stack Plot: Rating Distribution Over Years')
    plt.xlabel('Year Added')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/10_stack_plot_ratings.png')
    plt.close()

    print(f"Success! All 10 plots have been saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    save_visualizations()

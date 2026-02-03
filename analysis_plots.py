import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_style():
    """Sets the visual style for plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

def plot_content_distribution(df):
    """Plots the distribution of Movies vs TV Shows."""
    plt.figure(figsize=(6, 6))
    type_counts = df['type'].value_counts()
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=['#e50914', '#221f1f'])
    plt.title('Distribution of Content Types (Movies vs TV Shows)')
    plt.show()

def plot_content_growth(df):
    """Plots the growth of content added over the years."""
    plt.figure(figsize=(12, 6))
    df_growth = df.groupby('year_added').size()
    sns.lineplot(x=df_growth.index, y=df_growth.values, color='#e50914', linewidth=2.5)
    plt.title('Netflix Content Growth Over Years')
    plt.xlabel('Year Added')
    plt.ylabel('Number of Titles')
    plt.xlim(2008, df['year_added'].max()) # Focus on recent years where growth is visible
    plt.show()

def plot_top_countries(df):
    """Plots the top 10 countries producing content."""
    plt.figure(figsize=(12, 6))
    top_countries = df['primary_country'].value_counts().head(10)
    
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r')
    plt.title('Top 10 Countries Contributing Content')
    plt.xlabel('Number of Titles')
    plt.ylabel('Country')
    plt.show()

def plot_rating_distribution(df):
    """Plots the distribution of ratings."""
    plt.figure(figsize=(14, 7))
    order = df['rating'].value_counts().index
    sns.countplot(data=df, x='rating', order=order, hue='type', palette='Reds')
    plt.title('Rating Distribution by Content Type')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend(title='Type')
    plt.xticks(rotation=45)
    plt.show()
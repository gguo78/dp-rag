import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('ggplot')
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 12})

def extract_disease_data(evaluation_data, epsilon):

    diseases = []

    disease_names = set()
    for key in evaluation_data.keys():
        if isinstance(key, str) and key.startswith('["symptoms", "') and '*' not in key:
            parts = key.split('", "')
            if len(parts) >= 2:
                disease_name = parts[1].split('"]')[0].split('", ')[0]
                disease_names.add(disease_name)
    
    for disease_name in disease_names:
        count_key = f'["symptoms", "{disease_name}", {epsilon}]'
        success_key = f'["symptoms", "{disease_name}", {epsilon}, "success"]'
        
        # total & success count 
        total_count = evaluation_data.get(count_key, 0)
        success_count = evaluation_data.get(success_key, 0)
        
        success_rate = success_count / total_count if total_count > 0 else 0
        
        diseases.append({
            "disease": disease_name,
            "total_count": total_count,
            "success_count": success_count,
            "success_rate": success_rate,
            "epsilon": epsilon
        })
    
    return pd.DataFrame(diseases)

def load_all_data():
    epsilon_values = [0.001, 0.008, 0.028, 0.098, 0.248]
    results_data = {}
    all_disease_data = []

    for eps in epsilon_values:
        json_path = f'results/evaluation_eps_{eps}.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results_data[eps] = json.load(f)
            
            df_eps = extract_disease_data(results_data[eps], eps)
            all_disease_data.append(df_eps)
        else:
            print(f"Warning: File {json_path} not found")

    # combine
    if all_disease_data:
        return pd.concat(all_disease_data)
    else:
        return pd.DataFrame()

def plot_overall_success_rate(df):
    # annotation shows real epsilon values
    # x-axis of the plot for epsilon using a log scale for a clear plot
    plt.figure(figsize=(10, 6))
    
    agg_data = df.groupby('epsilon').agg({
        'success_count': 'sum',
        'total_count': 'sum'
    }).reset_index()
    
    agg_data['success_rate'] = agg_data['success_count'] / agg_data['total_count']
    
    # plot
    plt.plot(agg_data['epsilon'], agg_data['success_rate'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Success Rate')
    plt.title('Overall Success Rate vs. Privacy Budget (ε)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1)
    
    # add anotation
    for i, row in agg_data.iterrows():
        plt.annotate(f'ε={row["epsilon"]}\n{row["success_rate"]:.3f}', 
                     (row['epsilon'], row['success_rate']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualization/overall_success_rate.png', dpi=300)
    plt.close()

def plot_disease_success_rates(df):
    # this plot is plot for success rate for top diseases (diseases with highest counts)
    # ordered by decreasing values (The top disease is at the top bar.)

    disease_counts = df.groupby('disease')['total_count'].sum().sort_values(ascending=False)
    top_diseases = disease_counts.head(15).index.tolist()
    
    top_df = df[df['disease'].isin(top_diseases)]
    
    fig, axes = plt.subplots(1, 5, figsize=(24, 10), sharey=True)
    epsilon_values = sorted(df['epsilon'].unique())
    
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0.0, 0.8, 256))  
    cmap = LinearSegmentedColormap.from_list('dark_viridis', colors)
    
    plot_diseases = top_diseases.copy()
    plot_diseases.reverse()
    
    y_positions = range(len(plot_diseases))
    
    for i, eps in enumerate(epsilon_values):
        eps_df = top_df[top_df['epsilon'] == eps]
        
        plot_data = []
        for disease in plot_diseases:
            disease_row = eps_df[eps_df['disease'] == disease]
            if not disease_row.empty:
                success_rate = disease_row['success_rate'].values[0]
                count = disease_row['total_count'].values[0]
            else:
                success_rate = 0
                count = 0
            plot_data.append({'disease': disease, 'success_rate': success_rate, 'total_count': count})
        
        plot_df = pd.DataFrame(plot_data)
        
        bars = axes[i].barh(y_positions, plot_df['success_rate'], height=0.7, 
                    color=[cmap(rate) for rate in plot_df['success_rate']])
        
        axes[i].set_title(f'ε = {eps}', fontsize=14)
        axes[i].set_xlabel('Success Rate')
        axes[i].grid(True, alpha=0.3, axis='x')
    
    y_labels = [f"{disease} ({int(disease_counts[disease])})" for disease in plot_diseases]
    
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel('Disease (Total Count)')
    
    plt.suptitle('Success Rate by Disease for Different Privacy Budgets (ε)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig('visualization/disease_success_rates_by_count.png', dpi=300)
    plt.close()

def plot_count_distribution(df):
    # shows the distribution of all diseases count
    # unable to show all diseases name - just for a general understanding of the distribution
    disease_counts = df.groupby('disease')['total_count'].sum().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(14, 8))
    
    plt.bar(range(len(disease_counts)), disease_counts['total_count'])
    plt.xticks([]) 
    plt.xlabel('Diseases (sorted by count)')
    plt.ylabel('Total Count')
    plt.title('Distribution of Disease Counts')
    
    plt.tight_layout()
    plt.savefig('visualization/disease_count_distribution.png', dpi=300)
    plt.close()

def plot_heatmap(df):
    # heatmap for success rate for top diseases across epsilon values 
    # get only the top 15 diseases by count
    top_diseases = df.groupby('disease')['total_count'].sum().sort_values(ascending=False).head(15).index.tolist()
    
    top_df = df[df['disease'].isin(top_diseases)]
    pivot_df = top_df.pivot(index='disease', columns='epsilon', values='success_rate')
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Success Rate'})
    
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Disease')
    plt.title('Success Rate Heatmap by Disease and Privacy Budget')
    
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('visualization/success_rate_heatmap.png', dpi=300)
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("Loading data...")
    df = load_all_data()
    
    print(f"Data loaded successfully. Found {len(df)} entries across {df['disease'].nunique()} diseases.")
    
    print("Generating visualizations...")
    plot_overall_success_rate(df)
    plot_disease_success_rates(df)
    plot_count_distribution(df)
    plot_heatmap(df)
    
    print("Visualizations complete. Output saved to PNG files in /visualization.")

if __name__ == "__main__":
    main()

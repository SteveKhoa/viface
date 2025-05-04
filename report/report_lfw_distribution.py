"""
This code was generated using Claude 3.7 Sonnet with this prompt:

```
# Context
* I have a folder named lfw, which has the contents as images of different people.
# Instruction
* Write a minimal python script to create the figure of the distribution of the dataset. 
* The distribution is the histogram of number of images per person. 
# Constraints
* Use bins of width 5 in the histogram
* Try to use seaborn and matplotlib.pyplot in the Python code as the plotting library.
```

```attachment
George_W_Bush_0407.jpg.npy                       Maura_Tierney_0001.jpg.npy                       Zafarullah_Khan_Jamali_0001.jpg.npy
George_W_Bush_0408.jpg.npy                       Maureen_Fanning_0001.jpg.npy                     Zafarullah_Khan_Jamali_0002.jpg.npy
George_W_Bush_0409.jpg.npy                       Maureen_Fanning_0002.jpg.npy                     Zahir_Shah_0001.jpg.npy
George_W_Bush_0410.jpg.npy                       Maureen_Kanka_0001.jpg.npy                       Zaini_Abdullah_0001.jpg.npy
George_W_Bush_0411.jpg.npy                       Maurice_Cheeks_0001.jpg.npy                      Zakia_Hakki_0001.jpg.npy
George_W_Bush_0412.jpg.npy                       Maurice_Papon_0001.jpg.npy                       Zalmay_Khalilzad_0001.jpg.npy
George_W_Bush_0413.jpg.npy                       Maurice_Strong_0001.jpg.npy                      ...
```
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


FOLDER_PATH = os.getenv("REPORT_DATASET_LFW_FOLDER_PATH")
FIGURE_PATH = os.getenv("REPORT_FIGURE_PATH")


# Set the style for seaborn
sns.set_style("whitegrid")

def analyze_lfw_dataset(folder_path):
    """
    Analyze the LFW dataset and create a histogram of images per person.
    
    Args:
        folder_path (str): Path to the LFW dataset folder
    """
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Extract person names from filenames
    people = []
    for file in files:
        parts = file.split('_')
        person_name = '_'.join(parts[:-1])
        people.append(person_name)
    
    # Count images per person
    person_counts = Counter(people)
    
    # Get the counts as a list
    counts = list(person_counts.values())
    
    # Create a figure with a specific size
    plt.figure(figsize=(5, 4))
    
    # Create histogram with bins of width 5
    # Calculate the bin edges with width 5
    max_count = max(counts)
    bin_edges = np.arange(0, 30 + 6, 5)  # Add 6 to include the maximum value
    
    # Create histogram with seaborn
    sns.histplot(counts, bins=bin_edges, kde=False)

    # Get the histogram data
    hist_data = np.histogram(counts, bins=bin_edges)
    hist_heights = hist_data[0]
    hist_bins = hist_data[1]

    # Add frequency numbers on top of each bar
    for i in range(len(hist_heights)-1):
        if hist_heights[i] > 0:  # Only add text for bars with data
            # Calculate the center of each bar
            center = (hist_bins[i] + hist_bins[i+1]) / 2
            plt.text(center, hist_heights[i] + 1, str(hist_heights[i]), 
                    ha='center', va='bottom', fontweight='bold')  # Increased font size to 14

    # Cutoff outliners
    plt.xlim(0, 30)
    
    # Set plot labels and title
    plt.xlabel('Number of Images per Person')
    plt.ylabel('Frequency (Number of People)')
    plt.title('Distribution of Images per Person in LFW Dataset')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(FIGURE_PATH, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print some statistics
    total_people = len(person_counts)
    total_images = len(files)
    avg_images = total_images / total_people if total_people > 0 else 0
    
    print(f"Total number of people: {total_people}")
    print(f"Total number of images: {total_images}")
    print(f"Average images per person: {avg_images:.2f}")
    print(f"Max images for a single person: {max_count}")

# Call the function with the path to your LFW folder
if __name__ == "__main__":
    analyze_lfw_dataset(FOLDER_PATH)
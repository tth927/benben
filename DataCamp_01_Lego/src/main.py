#%%
# Import modules
import pandas as pd

# Read colors data
colors = pd.read_csv('datasets/colors.csv')

# Print the first few rows
colors.head()

# How many distinct colors are available?
num_colors = colors['name'].nunique()

# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby(by=colors.is_trans).count()


import matplotlib.pyplot as plt
# Read sets data as `sets`
sets = pd.read_csv('datasets/sets.csv')

# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets.groupby(sets.year)['num_parts'].mean()
print(parts_by_year)

# Plot trends in average number of parts by year
plt.figure()
plt.plot(parts_by_year)
plt.show()

# themes_by_year: Number of themes shipped by year
themes_by_year = sets.groupby(sets.year)['theme_id'].nunique().to_frame().reset_index()
print(themes_by_year[:5])

themes_by_year.info()
#%%
# Importing pandas
import pandas as pd
import numpy as np

# Reading in the data
df = pd.read_csv('C:\MyWorkspace\python_pj\DataCamp_02_CookieCat\datasets\cookie_cats.csv')

# Showing the first few rows
df.info()

# Counting the number of players in each AB group.
df.version.value_counts()
# or
df.groupby('version').count()

# This command makes plots appear in the notebook
%matplotlib inline

# Counting the number of players for each number of gamerounds 
plot_df = df.groupby('sum_gamerounds')['userid'].count().to_frame()
plot_df.info()


# Plotting the distribution of players that played 0 to 100 game rounds
ax = plot_df[:100].plot()
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
ax.set_xlabel("number of game rounds played")
ax.set_ylabel("Number of User")

# The % of users that came back the day after they installed
# This works because when using .sum() the True/False values will first be converted to 1/0
ret1 = df.retention_1.sum() / df.retention_1.count()
print(ret1)

# or
# This works because the mean is calculated by summing the values and dividing by the total number of values.
print(df.retention_1.mean())

# Calculating 1-day retention for each AB-group
df.groupby('version')['retention_1'].mean()

# df.head()
# # Creating an list with bootstrapped means for each AB-group
# bs_sample = df.sample(frac=1, replace=True)
# bs_sample.head()
# print(boot_mean)
#sub_df = df[['version','retention_1']]

iterations = 500
boot_1d = []
for i in range(iterations):
    bs_sample = df.sample(frac=1,  replace=True)
    boot_mean = bs_sample.groupby('version')['retention_1'].mean()
    boot_1d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_1d = pd.DataFrame(boot_1d)
    
# A Kernel Density Estimate plot of the bootstrap distributions
boot_1d.plot.kde()

# Adding a column with the % difference between the two AB-groups
boot_1d['diff'] = (boot_1d['gate_30'] - boot_1d['gate_40']) / boot_1d['gate_40'] * 100
boot_1d.head(10)

# Ploting the bootstrap % difference
ax = boot_1d['diff'].plot()
ax.set_xlabel("Iterations")


# Calculating the probability that 1-day retention is greater when the gate is at level 30
prob = (boot_1d['diff'] > 0).sum() / len(boot_1d)
# or 
prob = (boot_1d['diff'] > 0).mean()

# Pretty printing the probability
print(prob)


# Calculating 7-day retention for both AB-groups
df.groupby('version')['retention_7'].sum() / df.groupby('version')['retention_7'].count()

# Creating a list with bootstrapped means for each AB-group
boot_7d = []
for i in range(iterations):
    bs_sample = df.sample(frac=1, replace=True)
    boot_mean = bs_sample.groupby('version')['retention_7'].mean()
    boot_7d.append(boot_mean)
   
   
# Transforming the list to a DataFrame
boot_7d = pd.DataFrame(boot_7d)

# Adding a column with the % difference between the two AB-groups
boot_7d['diff'] = (boot_7d['gate_30'] - boot_7d['gate_40']) / boot_7d['gate_40'] * 100
boot_7d.head()

# Ploting the bootstrap % difference
ax = boot_7d['diff'].plot.kde()
ax.set_xlabel("% difference in means")

# Calculating the probability that 7-day retention is greater when the gate is at level 30
prob = (boot_7d['diff'] > 0).sum() / len(boot_7d)

# Pretty printing the probability
print(prob)

# So, given the data and the bootstrap analysis
# Should we move the gate from level 30 to level 40 ?
move_to_level_40 = False

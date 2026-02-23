# Alternative printing of tabulate topics to emphasize the distribution of
# human and AI generated content 

summary = df.groupby(['topic']).agg({'generated':['count','mean']}).reset_index()
summary.columns = ["Topic", "Records", "Percent AI"]
summary['Percent AI'] = np.round(summary['Percent AI']*100,2)
print(summary.to_markdown())

#### OUTPUT
# |    | Topic              |   Records |   Percent AI |
# |---:|:-------------------|----------:|-------------:|
# |  0 | Car Usage          |      7925 |         1.24 |
# |  1 | Education          |     14107 |        96.37 |
# |  2 | Electoral College  |      6429 |        30.98 |
# |  3 | Exploring Venus    |      5684 |         7    |
# |  4 | Limiting Car Usage |      7747 |        95.39 |
# |  5 | Online Education   |      9831 |         0.63 |
# |  6 | student experience |     13866 |        25.62 |

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-dark')
phq_palette = sns.color_palette(['#df477e', '#67bea3', '#5d8bc6', '#f4b543', '#e87d52', '#757570'])

def plot_important_entities_over_time(df, col_oi, filter_list, max_rk, y_title, ax=None):
    """Build a plot showing the important entities (in terms of their of the number
    of times they have been mentioned in the news) over a period of time.
    
    Parameters
    ----------
    df : dataframe containing the columns ['date', col_oi] (pandas Dataframe)
    col_oi : column of interest containing list of entities (str)
    filter_list : entites containing one of these strings will be ignored (list)
    max_rk : maximum rank to filter the most important entities per day (int)
    y_title : title for the y axis in the resulting plot (str)
    ax : subplot where to draw (matplotlib.axes._subplots.AxesSubplot)
    
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    
    date_range = df.date.sort_values().unique()
    x_date_range = np.arange(len(date_range))
    lkup_date = pd.DataFrame({'date': date_range, 'x': x_date_range})

    to_plot = (df[col_oi].apply(pd.Series) 
                          .merge(df[['date']], right_index=True, left_index=True)
                          .melt(id_vars=['date'], value_name='entity')
                          .drop('variable', axis=1)
                          .dropna()
                          .assign(importance=lambda x: 1)
                          .groupby(['date', 'entity'], as_index=False)
                          .sum()
                          .merge(lkup_date))
    
    to_plot = to_plot[~to_plot.entity.str.contains('|'.join(filter_list), case=False)]
    to_plot.loc[:, 'rk'] = to_plot.groupby('date')['importance'].rank('dense', ascending=False)
    to_plot = to_plot[to_plot.rk <= max_rk]
    to_plot.loc[:, 'y'] = np.random.uniform(0, 5, size=to_plot.shape[0])
    to_plot.loc[:, 'importance_bin'] = pd.cut(to_plot.importance, bins=len(phq_palette), labels=False)
    
    g = sns.scatterplot(data=to_plot, x='x', y='y', size='importance', 
                        hue='importance_bin', sizes=(20, 500), alpha=0.7, 
                        palette=phq_palette[:to_plot.importance_bin.nunique()], legend=False, ax=ax)
    g.set_xticks(x_date_range)
    g.set_xticklabels(date_range, rotation=75, color='black')
    g.set_yticks([])
    g.set_ylabel(y_title, fontsize=14, color='black')    
    g.set_xlabel('')
    g.tick_params(color='black', labelsize=14, width=0)
    g.grid(b=True, which='major')
    for index, row in to_plot.iterrows():
        ha = ['right', 'left'][np.random.randint(0, 2)]
        g.text(row['x'], row['y'], row['entity'], size=14, color='dimgray', horizontalalignment=ha, verticalalignment='center')
    
    g.figure.set_size_inches(24, 8)
    
    return g


def plot_important_topics_over_time(df, max_rk, y_title, ax=None):
    """Build a plot showing the important topics (according to a previous
    Latent Dirichlet Allocation modeling step) over a period of time.
    
    Parameters
    ----------
    df : dataframe containing a columns for the date and one column per topic (pandas Dataframe)
    max_rk : maximum rank to filter the most important topics per day (int)
    y_title : title for the y axis in the resulting plot (str)
    ax : subplot where to draw (matplotlib.axes._subplots.AxesSubplot)
    
    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    
    date_range = df.date.sort_values().unique()
    x_date_range = np.arange(len(date_range))
    lkup_date = pd.DataFrame({'date': date_range, 'x': x_date_range})

    to_plot = df.groupby('date', as_index=False).mean().melt(id_vars=['date'], value_name='importance').merge(lkup_date)
    to_plot.loc[:, 'rk'] = to_plot.groupby('date')['importance'].rank('dense', ascending=False)
    to_plot = to_plot[to_plot.rk <= max_rk]
    to_plot.loc[:, 'y'] = np.random.uniform(0, 5, size=to_plot.shape[0])
    to_plot.loc[:, 'importance_bin'] = pd.cut(to_plot.importance, bins=6, labels=False)
    
    g = sns.scatterplot(data=to_plot, x='x', y='y', size='importance', 
                        hue='importance_bin', sizes=(20, 500), alpha=0.7, 
                        palette=phq_palette[:to_plot.importance_bin.nunique()], legend=False, ax=ax)
    g.set_xticks(x_date_range)
    g.set_xticklabels(date_range, rotation=75, color='black')
    g.set_yticks([])
    g.set_ylabel(y_title, fontsize=14, color='black')    
    g.set_xlabel('')
    g.tick_params(color='black', labelsize=14, width=0)
    g.grid(b=True, which='major')
    for index, row in to_plot.iterrows():
        ha = ['right', 'left'][np.random.randint(0, 2)]
        g.text(row['x'], row['y'], row['variable'], size=14, color='dimgray', horizontalalignment=ha, verticalalignment='center')
    
    g.figure.set_size_inches(24, 8)
    
    return g
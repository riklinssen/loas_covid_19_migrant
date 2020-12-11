##############IMPORTS########################
import numpy as np
import pandas as pd
import pathlib
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from matplotlib.ticker import PercentFormatter
import statsmodels.stats.api as sms
from statsmodels.stats.weightstats import DescrStatsW

from pathlib import Path

##########################BOILERPLATE##########
# Oxfam colors
hex_values = ['#E70052',  # rood
              '#F16E22',  # oranje
              '#E43989',  # roze
              '#630235',  # Bordeax
              '#53297D',  # paars
              '#0B9CDA',  # blauw
              '#61A534',  # oxgroen
              '#0C884A'  # donkergroen
              ]


# -	Female workers: blue bars
# -	Male workers: purple bars
# - Cross border migrants-->brown bars
# - Internal migrants --> darkgreen ox
# -	Non-migrant workers: pink bars
# -	Formal workers: green bars
# -	Informal workers: orange bars

# groupcolordict
gr_coldict = {
    'Female': '#0B9CDA',
    'Male': '#53297D',
    'Cross-border migrant': '#630235',
    'Internal migrant': '#0C884A',
    'Non-migrant': '#E70052',
    'Formal employment': '#61A534', 
    'Informal employment':  '#F16E22' }

##########################FIlEPATHS##########
currentwd_path = Path.cwd()
data_path = currentwd_path / "data"
cleandata_path = Path(r"C:\Users\RikL\Box\ONL-IMK\2.0 Projects\Current\2020-07 Covid-19 Informal workers Laos\07. Data\Clean")
labels_path = currentwd_path.parent/"docs"
graphs_path = Path(r"C:\Users\RikL\Box\ONL-IMK\2.0 Projects\Current\2020-07 Covid-19 Informal workers Laos\07. Data\Graphs")


clean = pd.read_stata(
    cleandata_path/"Covid-19 Assessment Informal workers Laos.dta", convert_categoricals=True)

varlabel_df = pd.read_excel(labels_path/"Variable_Labels_clean_IMWLaos.xlsx",
                            usecols=['name', 'varlab'], index_col='name')
title_anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center', size='small')

#make everything unweighted by taking weight=1. 
#these are CI's for averages, not proportions, but that'll be fine for now. 
#############################################################

##groupings
for c in [c for c in clean.columns if 'group' in c ]: 
    print(clean[c].value_counts(dropna=False))

def outcome_bygroup_df(df, outcomes, groupbyvars):
    """returns dataframe with proportions of outcomes in rows, groupbyvars (gender, migrant, informal, total) in columns

    Parameters
    ----------
    df: DataFrame
        Original clean dataframe to groupby
    outcomes : list
        list of outcomes
    groupbyvars : list
         list of groupbyvars
    returns: 
        df with outcomes in rows, groupbyvars in cols, proportions in cells
    """
    colselect = groupbyvars + outcomes
    colnames=[]
    bygender = df.loc[:, colselect].groupby('group_gender')[outcomes].mean().T
    colnames.extend(list(bygender.columns))
    bymigrant = df.loc[:, colselect].groupby('group_migrant')[
        outcomes].mean().T
    colnames.extend(list(bymigrant.columns))
    byinformal = df.loc[:, colselect].groupby('group_informal')[
        outcomes].mean().T
    colnames.extend(list(byinformal.columns))
    bytotal = df.loc[:, colselect].groupby('Total')[outcomes].mean().T
    colnames.extend(list(bytotal.columns))
    data = pd.concat([bygender, bymigrant, byinformal,
                      bytotal], axis=1, ignore_index=True)
    
    data.columns=colnames
    data['label'] = varlabel_df.loc[outcomes]
    data = data.set_index('label')
    return data

clean['Total'] = 'Total'
groupbyvars = [c for c in clean.columns if c.startswith('group')]+['Total']



outcomecols=['out_employment_pre_cov_1',
'out_employment_pre_cov_2',
'out_employment_pre_cov_3', 
'out_employment_pre_cov_4',
'out_employment_pre_cov_5',
'out_employment_pre_cov_6',
'out_employment_pre_cov_7',
'out_employment_pre_cov_8',
'out_employment_pre_cov_9',
'out_employment_pre_cov_10']
for c in outcomecols: 
    print(clean[c].value_counts(dropna=False))

precovidcols=['out_employment_pre_cov_1',
'out_employment_pre_cov_2',
'out_employment_pre_cov_3', 
'out_employment_pre_cov_4',
'out_employment_pre_cov_5',
'out_employment_pre_cov_6',
'out_employment_pre_cov_7',
'out_employment_pre_cov_8',
'out_employment_pre_cov_9',
'out_employment_pre_cov_10']
postcovidcols=['out_employment_current_1',
'out_employment_current_2', 
'out_employment_current_3', 
'out_employment_current_4', 
'out_employment_current_5', 
'out_employment_current_6', 
'out_employment_current_7', 
'out_employment_current_8', 
'out_employment_current_9', 
'out_employment_current_10', 
'out_employment_current_11']


##bygender
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower right',
           ncol=1, frameon=False, fontsize='small')    	
    sns.despine(ax=ax)

fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_gender.svg', bbox_inches='tight')




##by migrant
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_group_migrant.svg', bbox_inches='tight')
fig.show()






##by informal
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_group_informal.svg', bbox_inches='tight')


#####sector`
#	out_employment_sector_pre_cov 
#	out_employment_sector_current


precovidcols=['out_employment_sector_pre_cov_1',
'out_employment_sector_pre_cov_2',
'out_employment_sector_pre_cov_3',
'out_employment_sector_pre_cov_4']
postcovidcols=['out_employment_sector_current_1',
'out_employment_sector_current_2',
'out_employment_sector_current_3',
'out_employment_sector_current_4']



##bygender
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower right',
           ncol=1, frameon=False, fontsize='small')    	
    sns.despine(ax=ax)

fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_sector_bygender.svg', bbox_inches='tight')




##by migrant
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_sector_migrant.svg', bbox_inches='tight')
fig.show()






##by informal
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_employment_current_11']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'employment_by_sector_informal.svg', bbox_inches='tight')







##hours worked pre and post covid. 

precovidcols=['out_hours_worked_pre_cov_1',
'out_hours_worked_pre_cov_2',
'out_hours_worked_pre_cov_3',
'out_hours_worked_pre_cov_4']

postcovidcols=['out_hours_worked_covid_1',
'out_hours_worked_covid_2',
'out_hours_worked_covid_3',
'out_hours_worked_covid_4']

##bygender
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_gender', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower right',
           ncol=1, frameon=False, fontsize='small')    	
    sns.despine(ax=ax)

fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_hours_worked_covid_4']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'hoursworked_by_gender.svg', bbox_inches='tight')




##by migrant
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_migrant', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_hours_worked_covid_4']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'housworked_by_group_migrant.svg', bbox_inches='tight')







##by informal
pre_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=precovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
post_cov_mean=pd.DataFrame(clean.pivot_table(index='group_informal', values=postcovidcols,aggfunc='mean').unstack(1), columns=['mean']).reset_index()
for df in [pre_cov_mean, post_cov_mean]: 
    df.columns=['outcome', 'group', 'mean']

pre_cov_mean['label']=pre_cov_mean['outcome'].map(varlabel_df.loc[precovidcols].to_dict()['varlab'])
post_cov_mean['label']=post_cov_mean['outcome'].map(varlabel_df.loc[postcovidcols].to_dict()['varlab'])
pre_cov_mean['time']='pre-covid'
post_cov_mean['time']='current'

allmeans=pd.concat([pre_cov_mean, post_cov_mean], ignore_index=True)
pivot=allmeans.pivot_table(index=['group','label'], columns='time', values='mean')

idx=pd.IndexSlice
sns.set_style('ticks')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharey='row')
for i,gr in enumerate(pivot.index.get_level_values('group').unique()):
    ax=fig.axes[i]
    ax.set_title(gr, color=gr_coldict[gr])
    sel=pivot.loc[idx[gr,:,:]].droplevel('group')
    ypos=np.arange(len(sel.index))
    bar_width = 0.4
    ax.barh(y=ypos, height=bar_width, width=sel['pre-covid'], color=gr_coldict[gr], alpha=0.5, label=gr+"\n-pre-covid")
    ax.barh(y=ypos+bar_width, height=bar_width, width=sel['current'],color=gr_coldict[gr], label=gr+"\n-current")
    #add text
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.03, p.get_y()+(bar_width/2), "{:.0%}".format(p.get_width()), color=p.get_facecolor(), verticalalignment='center', size='xx-small')

# Fix the x-axes.
    ax.set_yticks(ypos + bar_width / 2)
    ax.set_yticklabels(sel.index)
    for label in ax.get_yticklabels():
        label.set_va("center")
        label.set_rotation(0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
# legend
    ax.legend(loc='lower left', bbox_to_anchor= (0, -0.5), ncol=1,
            borderaxespad=0, frameon=False)
    sns.despine(ax=ax)

fig.text(0, -0.3, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_hours_worked_covid_4']))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'hoursworked_by_group_informal.svg', bbox_inches='tight')


#not properly sorted but that should be ok. I guess. Also to check up to 20 hours. 


def outcome_bygroup_df(df, outcomes, groupbyvars):
    """returns dataframe with proportions of outcomes in rows, groupbyvars (gender, migrant, informal, total) in columns

    Parameters
    ----------
    df: DataFrame
        Original clean dataframe to groupby
    outcomes : list
        list of outcomes
    groupbyvars : list
         list of groupbyvars
    returns: 
        df with outcomes in rows, groupbyvars in cols, proportions in cells
    """
    colselect = groupbyvars + outcomes
    colnames=[]
    bygender = df.loc[:, colselect].groupby('group_gender')[outcomes].mean().T
    colnames.extend(list(bygender.columns))
    bymigrant = df.loc[:, colselect].groupby('group_migrant')[
        outcomes].mean().T
    colnames.extend(list(bymigrant.columns))
    byinformal = df.loc[:, colselect].groupby('group_informal')[
        outcomes].mean().T
    colnames.extend(list(byinformal.columns))
    bytotal = df.loc[:, colselect].groupby('Total')[outcomes].mean().T
    colnames.extend(list(bytotal.columns))
    data = pd.concat([bygender, bymigrant, byinformal,
                      bytotal], axis=1, ignore_index=True)
    
    data.columns=colnames
    data['label'] = varlabel_df.loc[outcomes]
    data = data.set_index('label')
    return data

clean['Total'] = 'Total'
groupbyvars = [c for c in clean.columns if c.startswith('group')]+['Total']





data=outcome_bygroup_df(clean, ['out_lose_job_covid_1', 'out_lose_job_covid_2'], groupbyvars)


sns.set_style('white')
fig, axes = plt.subplots(nrows=8, ncols=1, sharex='col', figsize=(
    3, 6.25))
# title row.
titleaxes = fig.axes[:8]
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.set_title(groupnames[group], color=gr_title_coldict[group], size='small', loc='left')
    lostjob=ax.barh(y =1, height= 0.2, width=data.loc['Lost job during covid-pandemic', group], color=gr_title_coldict[group])
    didnot=ax.barh(y =1, height= 0.2, width=data.loc['Did not lose job', group], left=data.loc['Lost job during covid-pandemic', group], color='gray', alpha=0.6)
    ax.axis('off')
        # labels
    for p in lostjob.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()/2, p.get_y()+0.1, "{:.0%}".format(
            round(p.get_width(),2))+"\nlost job", color='white', verticalalignment='center', size='xx-small')
    for p in didnot.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(data.loc['Did not lose job', group]+(p.get_width()/2), p.get_y()+0.1, "{:.0%}".format(
            round(p.get_width(),2))+"\ndid not", color='white', verticalalignment='center', size='xx-small')

fig.subplots_adjust(hspace=0.6, bottom = 0.1)
fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos,\ntotal n=" +
            str(len(clean['out_lose_job_covid_1'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'out_losejobs.svg', bbox_inches='tight')



def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('plasma')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    axlim=np.sum(data, axis=1).max()
    if axlim==np.nan:
        ax.set_xlim(0,1 )
    else: 
        ax.set_xlim(0)
    #spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

   
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y,  "{:.0%}".format(
            round(c,2)), ha='center', va='center',
                    color=text_color)
       
    if len(category_names)<6: 
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='small', frameon=False)
    if len(category_names)>5:
        ax.legend(ncol=3, bbox_to_anchor=(0, 1),
        loc='lower left', fontsize='small', frameon=False)


    fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos,\ntotal n="+str(len(clean.loc[:,outcomes[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
    return fig, ax


#now do for each var
outcomes=['out_samefood_covid_1', 'out_samefood_covid_2']
filename=graphs_path/'out_samefood_covid.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.to_dict(orient='list')
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



#now do for each var
outcomes=['out_unpaid_carework_who_1',
'out_unpaid_carework_who_2',
'out_unpaid_carework_who_3']

filename=graphs_path/'out_unpaid_carework_who.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.to_dict(orient='list')
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



#nut_unpaid_carework_chang
outcomes=['out_unpaid_carework_change_1',
'out_unpaid_carework_change_2',
'out_unpaid_carework_change_3',
'out_unpaid_carework_change_4']


filename=graphs_path/'out_unpaid_carework_change.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.to_dict(orient='list')
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

vartoviz=['out_gbv_violence_covid_', 'out_help_violence_covid_', 'out_discrimination_covid_', 'out_hh_cope_covid_',]
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    filname1=str(outcomes[0])[:-2]+".svg"
    filename=graphs_path/filname1
    data=outcome_bygroup_df(clean, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    try:
        fig=survey(results, category_names)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
    except ValueError as error:
        print(error)
        print(var)
#•	In the months after the COVID lockdown (since June), were there times when you run out of food for you and your family and there was no money to buy more?
#not available
#•	If your household cannot access the same type of foods as before, what are the reasons?

outcomes1=['out_reasons_nosamefood_1',
'out_reasons_nosamefood_2',
'out_reasons_nosamefood_3',
'out_reasons_nosamefood_4',
'out_reasons_nosamefood_5',
'out_reasons_nosamefood_6',
'out_reasons_nosamefood_7',
'out_reasons_nosamefood_8',
'out_reasons_nosamefood_9']

for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_reasons_nosamefood.svg', bbox_inches='tight')






outcomes=['out_migration_unsafe_1',
'out_migration_unsafe_2']
filename=graphs_path/'out_migration_unsafe.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



outcomes=['out_discrimination_covid_1',
'out_discrimination_covid_2']
filename=graphs_path/'out_discrimination_covid_.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



outcomes=['out_remittances_1','out_remittances_2']
filename=graphs_path/'out_remittances.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


outcomes=['out_remittances_cease_1','out_remittances_cease_2']

filename=graphs_path/'out_remittances_cease.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_hh_cope_covid

outcomes=['out_hh_cope_covid_1',
'out_hh_cope_covid_2',
'out_hh_cope_covid_3',
'out_hh_cope_covid_4',
'out_hh_cope_covid_5']

filename=graphs_path/'out_hh_cope_covid.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_challenges
outcomes=['out_challenges_1',
'out_challenges_2',
'out_challenges_3',
'out_challenges_4',
'out_challenges_5',
'out_challenges_6',
'out_challenges_7',
'out_challenges_8',
'out_challenges_9',
'out_challenges_10',
'out_challenges_11']
for c in outcomes:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


filename=graphs_path/'out_challenges.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

##better try different graph with many cats
# groupnames={'Male':'Men', 'Female': 'Women', 'Non-migrant':'Non-\nmigrants' , 'Internal migrant': 'Internal\nmigrants' ,
#        'Cross-border migrant':'Cross-border\nmigrants', 'Formal workers':'Formal\nworkers', 'Informal workers':'Informal\nworkers', 'Total':'Total'}

###out_migrant_where
outcomes1 = ['out_challenges_1',
'out_challenges_2',
'out_challenges_3',
'out_challenges_4',
'out_challenges_5',
'out_challenges_6',
'out_challenges_7',
'out_challenges_8',
'out_challenges_9',
'out_challenges_10',
'out_challenges_11']

fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('challenges:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_challenges_11']))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_challenges.svg', bbox_inches='tight')

#out_cope_covid_
outcomes1=['out_cope_covid_1',
'out_cope_covid_2',
'out_cope_covid_3',
'out_cope_covid_4',
'out_cope_covid_5',
'out_cope_covid_6',
'out_cope_covid_7',
'out_cope_covid_8',
'out_cope_covid_9',
'out_cope_covid_10',
'out_cope_covid_11',
'out_cope_covid_12']

for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('coping mechanisms:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_cope_covid_12']))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_cope_covid.svg', bbox_inches='tight')





#out_coverage_unemp

outcomes=['out_coverage_unemp_1',
'out_coverage_unemp_2',
'out_coverage_unemp_3',
'out_coverage_unemp_4']

filename=graphs_path/'out_coverage_unemp.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()









#out_coverage_unemp_where_1
outcomes=['out_coverage_unemp_where_1',
'out_coverage_unemp_where_2',
'out_coverage_unemp_where_3']

filename=graphs_path/'out_coverage_unemp_where_.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



#out_coverage_unemp_comp
outcomes=['out_coverage_unemp_comp_1',
'out_coverage_unemp_comp_2']


#nans in columns
filename=graphs_path/'out_coverage_unemp_comp.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_coverage_unemp_enough

outcomes=['out_coverage_unemp_enough_1',
'out_coverage_unemp_enough_2']


#nans in columns
filename=graphs_path/'out_coverage_unemp_enough.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_coverage_health

outcomes=['out_coverage_health_1',
'out_coverage_health_2', 'out_coverage_health_3', 'out_coverage_health_4']


#nans in columns
filename=graphs_path/'out_coverage_health.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_coverage_health_where


outcomes=['out_coverage_health_where_1','out_coverage_health_where_2']


#nans in columns
filename=graphs_path/'out_coverage_health_where.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_coverage_health_use

outcomes=['out_coverage_health_use_1','out_coverage_health_use_2']

filename=graphs_path/'out_coverage_health_use.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



#out_coverage_health_quality


outcomes=['out_coverage_health_quality_1',
'out_coverage_health_quality_2',
'out_coverage_health_quality_3',
'out_coverage_health_quality_4']


filename=graphs_path/'out_coverage_health_quality.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_coverage_sickleave


outcomes=['out_coverage_sickleave_1', 
'out_coverage_sickleave_2',
'out_coverage_sickleave_3',
'out_coverage_sickleave_4']


filename=graphs_path/'out_coverage_sickleave.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()


#out_diff_foods_cope
outcomes=['out_diff_foods_cope_1', 'out_diff_foods_cope_2']

filename=graphs_path/'out_diff_foods_cope.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_diff_foods



outcomes1=['out_diff_foods_1',
'out_diff_foods_2',
'out_diff_foods_3',
'out_diff_foods_4',
'out_diff_foods_5',
'out_diff_foods_6',
'out_diff_foods_7']


for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('different foods:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean['out_diff_foods_7']))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_diff_foods.svg', bbox_inches='tight')




#out_cope_nonfinance_who
outcomes=['out_cope_nonfinance_who_1', 'out_cope_nonfinance_who_2', 'out_cope_nonfinance_who_3', 'out_cope_nonfinance_who_4']

for c in outcomes:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


filename=graphs_path/'out_cope_nonfinance_who.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()



#out_cope_borrow_who

outcomes=['out_cope_borrow_who_1',
'out_cope_borrow_who_2',
'out_cope_borrow_who_3',
'out_cope_borrow_who_4',
'out_cope_borrow_who_5']


for c in outcomes:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


filename=graphs_path/'out_cope_borrow_who.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_outlook_opp

outcomes=['out_outlook_opp_1',
'out_outlook_opp_2',
'out_outlook_opp_3']



filename=graphs_path/'out_outlook_opp.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_outlook_inc

outcomes=['out_outlook_inc_1',
'out_outlook_inc_2',
'out_outlook_inc_3']



filename=graphs_path/'out_outlook_inc.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()

#out_outlook_migration
outcomes=['out_outlook_migration_1',
'out_outlook_migration_2',
'out_outlook_migration_3',
'out_outlook_migration_4',
'out_outlook_migration_5']




filename=graphs_path/'out_outlook_migration.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()













#5.4.3 NEEDS OF SOCIAL SERVICES
#vartoviz loop

vartoviz=['out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_']
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+".svg"
    filename=graphs_path/filname1
    data=outcome_bygroup_df(clean, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    try:
        fig=survey(results, category_names)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
    except ValueError as error:
        print(error)
        print(var)

#•	How would you rate the importance of these providers of social services when preparing better for a situation such as COVID?
vartoviz=['out_prov_socialserv_gov_', 'out_prov_socialserv_employer_', 'out_prov_socialserv_cso_', 'out_prov_socialserv_union_', 'out_prov_socialserv_other_']

for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+".svg"
    filename=graphs_path/filname1
    data=outcome_bygroup_df(clean, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    try:
        fig=survey(results, category_names)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
    except ValueError as error:
        print(error)
        print(var)

#####
#out_precond_remigrate_1 out_precond_remigrate_2 out_precond_remigrate_3 out_precond_remigrate_4 out_precond_remigrate_5 out_precond_remigrate_6 out_precond_remigrate_7




#• Which preconditions will be important for you in order to re-migrate to make a living away?
#out_precond_remigrate_1 out_precond_remigrate_2 out_precond_remigrate_3 out_precond_remigrate_4 out_precond_remigrate_5 out_precond_remigrate_6 out_precond_remigrate_7




outcomes1=[c for c in clean if 'out_precond_remigrate_' in c]

for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_precond_remigrate.svg', bbox_inches='tight')


# • Which preconditions will be important for you in order to ensure you can make a living here?
# out_precod_stay_1 out_precod_stay_2 out_precod_stay_3 out_precod_stay_4


outcomes1=[c for c in clean if 'out_precod_stay_' in c]
print(outcomes1)
for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_precond_stay.svg', bbox_inches='tight')



# In your opinion, what would be the key priorities to support for informal workers?
# out_key_support_rights out_key_support_health out_key_support_training out_key_support_migration out_key_support_finance out_key_support_other




outcomes1=[c for c in clean if 'out_key_' in c]
print(outcomes1)
for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_key_support.svg', bbox_inches='tight')

#remittances cease howmuch

outcomes=['out_remittances_cease_howmuch_1',
'out_remittances_cease_howmuch_2',
'out_remittances_cease_howmuch_3']
filename=graphs_path/'out_remittances_cease_howmuch.svg'
data=outcome_bygroup_df(clean, outcomes, groupbyvars)
category_names=list(data.index)
results=data.dropna(axis=1, how='all').to_dict(orient='list')
print(results)
fig=survey(results, category_names)
plt.savefig(filename, bbox_inches='tight')
plt.show()





##out_cope_remigration_1


outcomes1=[c for c in clean if 'out_cope_remigration_' in c]
print(outcomes1)
# for c in outcomes1:
#     print(clean[c].cat.codes.value_counts(dropna=False))
#     clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_cope_remigration.svg', bbox_inches='tight')



###income declines pre-post covid for quintiles. 
clean = pd.read_stata(
    cleandata_path/"Covid-19 Assessment Informal workers Laos.dta", convert_categoricals=True)

clean['Total']='Total'
by_quint_total=clean.groupby(['quintile_income_pre', 'Total'])['income_change'].mean().to_frame().dropna().reset_index()

by_quint_gndr=clean.groupby(['quintile_income_pre', 'group_gender'])['income_change'].mean().to_frame().dropna().reset_index()
by_quint_informal=clean.groupby(['quintile_income_pre', 'group_informal'])['income_change'].mean().to_frame().dropna().reset_index()
by_quint_migrant=clean.groupby(['quintile_income_pre', 'group_migrant'])['income_change'].mean().to_frame().dropna().reset_index()


dfs=[by_quint_gndr, by_quint_informal,by_quint_migrant, by_quint_total]
for df in dfs: 
    df.columns=['quintile_income_pre', 'group', 'income_change']

by_quint=pd.concat(dfs).pivot_table(index=['quintile_income_pre','group'])

groupnames={'Male':'Men', 'Female': 'Women', 'Non-migrant':'Non-\nmigrants' , 'Internal migrant': 'Internal\nmigrants' ,
       'Cross-border migrant':'Cross-border\nmigrants', 'Total':'Total', 'Formal employment': 'Formal\nemployment' , 'Informal employment': 'Informal\nemployment' }
gr_title_coldict = {
    'Male': '#0B9CDA',
    'Female': '#53297D',
    'Cross-border migrant': '#630235',
    'Internal migrant': '#0C884A',
    'Non-migrant': '#E70052',
    'Formal workers': '#61A534',
    'Informal workers': '#F16E22',
    'Total': '#000000',
    'Formal employment': '#61A534', 
    'Informal employment':  '#F16E22'}  
    
fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, ['Male', 'Female', 'Cross-border migrant', 'Internal migrant', 'Non-migrant', 'Formal employment', 'Informal employment', 'Total'])):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

for i, group in enumerate(['Male', 'Female', 'Cross-border migrant', 'Internal migrant', 'Non-migrant', 'Formal employment', 'Informal employment', 'Total']): 
    sel=by_quint.loc[idx[:,group,:]]
    sel['color']=np.where(sel['income_change']>0, 'green', 'red')
    axes[1,i].bar(x=sel.index, height=sel['income_change'],color=sel['color'])
    axes[1,i].set_xticks([1,2,3,4,5])
    axes[1,i].set_xticklabels(['1-lowest income group','2','3','4','5-highest income group'], rotation=90, size='small')
    axes[1,i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    sns.despine(ax=axes[1,i])
fig.savefig(graphs_path/'income_change_quintiles_by_group.svg', bbox_inches='tight')




###Only graph for totals for income change by quintile. 
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', figsize=(
    3, 3))
ax.barh(y=by_quint_total['quintile_income_pre'], width=by_quint_total['income_change'], color='red')
ax.set_xlim(-1,1)
ax.axvline(x=0, ymin=0, ymax=6, color='black', linestyle=':')
ax.set_yticks([1,2,3,4,5])
ax.set_yticklabels(['1-lowest income group','2','3','4','5-highest income group'], size='small')
ax.set_xlabel('change in income\npre-post covid-19', fontstyle='oblique', color='gray')
ax.set_ylabel('income group', fontstyle='oblique', color='gray')
for p in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
        p.get_width()), color='white', verticalalignment='top', size='small')

ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
sns.despine(ax=ax)
fig.text(-0.2, -0.2, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos,\ntotal n=" +
            str(len(clean.loc[:,'income_change'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.savefig(graphs_path/'income_change_quintiles_by_total.svg', bbox_inches='tight')





###Average income by group. 
#incomegroups
fig, ax = plt.subplots(nrows=4, ncols=1, sharex='col', figsize=(3, 6.25), gridspec_kw={'height_ratios': [1, 2, 3,2]})
grtit=['total','by gender','by migrant status', 'by worker status']
for i, (tit, grouping) in enumerate(zip(grtit, ['Total', 'group_gender', 'group_migrant', 'group_informal'])):
    ax[i].set_title(tit, loc='left')
    sel=clean.groupby(grouping)['income_pre_covid'].mean().to_frame()
    sel['color']=sel.index.map(gr_title_coldict)
    sel['label']=round(sel['income_pre_covid'],2).astype(str)
    sel=sel.reset_index()
    sel.columns=['group', 'income_pre_covid', 'color', 'label']
    ax[i].scatter(y=sel['group'], x=sel['income_pre_covid'], color=sel['color'])
    ax[i].hlines(y=sel['group'], xmin=0, xmax=sel['income_pre_covid'], color=sel['color'])
    for key, row in sel.iterrows():
        ax[i].text(x=row['income_pre_covid']+1, y=row['group'], s=row['label'],  ha='left', va='bottom', size='x-small', color=row['color'])
    ax[-1].set_xlabel('income in LAK\n(x100k)', fontstyle='oblique')
    sns.despine(ax=ax[i])
fig.text(0., 0., "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos,\ntotal n=" + str(len(clean.loc[:,'income_pre_covid'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(hspace=0.3)
fig.savefig(graphs_path/'average_pre_covid_income_by_grouping.svg', bbox_inches='tight')

###average % loss in each group

fig, ax = plt.subplots(nrows=4, ncols=1, sharex='col', figsize=(3, 6.25), gridspec_kw={'height_ratios': [1, 2, 3,2]})
grtit=['total','by gender','by migrant status', 'by worker status']
for i, (tit, grouping) in enumerate(zip(grtit, ['Total', 'group_gender', 'group_migrant', 'group_informal'])):
    ax[i].set_title(tit, loc='center', size='small')
    sel=clean.groupby(grouping)['income_change'].mean().to_frame()
    sel['color']=sel.index.map(gr_title_coldict)
    sel['label']=round(sel['income_change'],2)
    sel=sel.reset_index()
    sel.columns=['group', 'income_change', 'color', 'label']
    ax[i].scatter(y=sel['group'], x=sel['income_change'], color=sel['color'])
    ax[i].hlines(y=sel['group'], xmin=0, xmax=sel['income_change'], color=sel['color'])
    ax[i].axvline(x=0, linestyle='-', color='gray')
    for key, row in sel.iterrows():
        ax[i].text(x=row['income_change']-0.3, y=row['group'], s='{:,.0%}'.format(row['label']),  ha='left', va='center', size='small', color=row['color'])
    
    sns.despine(ax=ax[i])
    ax[i].set_xlim(-1,1)
    ax[i].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax[-1].set_xlabel('income change\n(in % of pre-covid income)', fontstyle='oblique')
fig.text(0., 0., "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos,\ntotal n=" + str(len(clean.loc[:,'income_pre_covid'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(hspace=0.3)
fig.savefig(graphs_path/'average__income_change_by_grouping.svg', bbox_inches='tight')






###cross by out_outlook_migration
#outvars: 
#out_key_support_rights out_key_support_health out_key_support_training out_key_support_migration out_key_support_finance out_key_support_other
#'out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_'
#Groupby's: 

#merge together
#Re-migrate 
#vs. 
#Stay and get a job 
#vs
#others


clean['outlookgrouper']=clean['outlook_migration'].cat.codes.map({0: 'Re-migrate', 1:'Stay and get a job', 2: 'Other', 3: 'Other', 4: 'Other', 5: 'Other', -1:np.nan})


##seperate graphs for values of outlookgrouper

#total
outcomes1=[c for c in clean if 'out_key_support_' in c]
print(outcomes1)
for c in outcomes1:
    print(clean[c].cat.codes.value_counts(dropna=False))
    clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})

#total
fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.suptitle('Total')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_key_support_total.svg', bbox_inches='tight')


###remigrate

outcomes1=[c for c in clean if 'out_key_support_' in c]

#total
fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean.loc[clean['outlookgrouper']=='Re-migrate'], outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +str(
            clean.loc[clean['outlookgrouper']=="Re-migrate"]['out_key_support_rights'].value_counts().sum())+".", size='x-small',  ha="left", color='gray')
fig.suptitle('Re-migrate')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_key_support_remigrate.svg', bbox_inches='tight')


###'Stay and get a job'

outcomes1=[c for c in clean if 'out_key_support_' in c]

#total
fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean.loc[clean['outlookgrouper']=='Stay and get a job'], outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(clean.loc[clean['outlookgrouper']=="Stay and get a job"]['out_key_support_rights'].value_counts().sum())+".", size='x-small',  ha="left", color='gray')
fig.suptitle('Stay and get a job')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_key_support_staygetjob.svg', bbox_inches='tight')




###'other'

outcomes1=[c for c in clean if 'out_key_support_' in c]

#total
fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean.loc[clean['outlookgrouper']=='Other'], outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(clean.loc[clean['outlookgrouper']=="Other"]['out_key_support_rights'].value_counts().sum())+".", size='x-small',  ha="left", color='gray')
fig.suptitle('Other')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_key_support_other.svg', bbox_inches='tight')



####outcomes='out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_'
##seperate graphs for values of outlookgrouper

#for total: 


vartoviz=['out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_']
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+"_total.svg"
    filename=graphs_path/filname1
    data=outcome_bygroup_df(clean, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    try:
        fig=survey(results, category_names)
        plt.suptitle('Total')
        plt.savefig(filename, bbox_inches='tight')
        
        plt.show()
    except ValueError as error:
        print(error)
        print(var)


##remigrate
vartoviz=['out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_']
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+"_re_migrate.svg"
    filename=graphs_path/filname1
    data_sel=clean.loc[clean['outlookgrouper']=='Re-migrate',:]     
    data=outcome_bygroup_df(data_sel, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    fig=survey(results, category_names)
    plt.suptitle('Re-migrate')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
   



##stay and get job
vartoviz=['out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_']
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+"_staygetjob.svg"
    filename=graphs_path/filname1
    data_sel=clean.loc[clean['outlookgrouper']=='Stay and get a job',:]     
    data=outcome_bygroup_df(data_sel, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    fig=survey(results, category_names)
    plt.suptitle('Re-migrate')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


##say and get a job
vartoviz=['out_needs_socialserv_money_', 'out_needs_socialserv_food_', 'out_needs_socialserv_health_', 'out_needs_socialserv_credit_', 'out_needs_socialserv_serv_', 'out_needs_socialserv_migration_', 'out_needs_socialserv_women_', 'out_needs_socialserv_other_']
for var in vartoviz: 
    outcomes=[o for o in varlabel_df.index if o.startswith(var)]
    print(outcomes)
    filname1=str(outcomes[0])[:-2]+"_other.svg"
    filename=graphs_path/filname1
    data_sel=clean.loc[clean['outlookgrouper']=='Other',:]     
    data=outcome_bygroup_df(data_sel, outcomes, groupbyvars)
    category_names=list(data.index)
    results=data.to_dict(orient='list')
    fig=survey(results, category_names)
    plt.suptitle('Other')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()



#out discrimination who






outcomes1=['discrimination_who_1', 'discrimination_who_2', 'discrimination_who_3', 'discrimination_who_4', 'discrimination_who_5', 'discrimination_who_6', 'discrimination_who_7', 'discrimination_who_8', 'discrimination_who_9']
print(outcomes1)
#not properly recoded, assuming 0 actually means 1. 
for c in outcomes1:
     print(clean[c].value_counts(dropna=False))
     clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'discrimination_who.svg', bbox_inches='tight')

###	out_cope_nonfinance_who_1 out_cope_nonfinance_who_2 out_cope_nonfinance_who_3 out_cope_nonfinance_who_4 with answers on y axis



outcomes1=[c for c in clean.columns if 'out_cope_nonfinance_who' in c]
print(outcomes1)

for c in outcomes1:
     print(clean[c].value_counts(dropna=False))
     clean[c]=clean[c].cat.codes.map({-1:np.nan,0:0, 1:1})


fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 7]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]

data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    #if i == 0:
        #ax.set_ylabel('precondition:', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)

fig.align_ylabels(fig.axes)
#footnote
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among migrant workers in Laos, total n=" +
            str(len(clean.loc[:,outcomes1[-1]].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_cope_nonfinance_who.svg', bbox_inches='tight')
##############IMPORTS########################
import numpy as np
import pandas as pd
import pathlib
import datetime
import seaborn as sns
import matplotlib.cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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
cleandata_path = data_path/"clean"
labels_path = currentwd_path.parent/"docs"
graphs_path = currentwd_path/"graphs"


clean = pd.read_stata(
    cleandata_path/"Covid-19 Assessment Informal workers Laos.dta")


# descriptive stats.

def value_counts_df(df, col):
    """
    Returns pd.value_counts() as a DataFrame

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe on which to run value_counts(), must have column `col`.
    col : str
        Name of column in `df` for which to generate counts
    normalize: BOOL
        if true returns proportions
    Returns
    -------
    Pandas Dataframe
        Returned dataframe will have a single column named "count" which contains the count_values()
        for each unique value of df[col]. The index name of this dataframe is `col`.

    Example
    -------
    >>> value_counts_df(pd.DataFrame({'a':[1, 1, 2, 2, 2]}), 'a')
       count
    a
    2      3
    1      2
    """
    df = pd.DataFrame(df[col].value_counts(normalize=True))
    df.index.name = col
    df.columns = ['proportion']
    return df


groupvars = [c for c in clean if c.startswith('group')]

# groupvars as descriptives.

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))
for i, var in enumerate(groupvars):
    ax = fig.axes[i]
    data = value_counts_df(clean, var)
    data['clr'] = data.index.map(gr_coldict)
    ax.bar(data.index, data.proportion, color=data['clr'])
    # labels
    for p in ax.patches:
        ax.text(x=(p.get_x() + (p.get_width()) / 2), y=(p.get_height()+0.02), s="{:.0%}".format(round(
            p.get_height(), 3)), color=p.get_facecolor(), horizontalalignment='center', size='smaller')
        # long labels on ax[1]
    # x-axis
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for p in ax.patches:
        ax.text(x=(p.get_x() + (p.get_width()) / 2), y=(p.get_height()/2), s="{:.0%}".format(round(
            p.get_height(), 3)), color=p.get_facecolor(), horizontalalignment='center', size='smaller')

    # rotate some labels for long labels ax[1:]
    if i > 0:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(30)
# add number of respondents in each bar.

    sns.despine(ax=ax)

axes[0].set_title('by gender',  loc='left', size='small')
axes[1].set_title('by migrant status', loc='left', size='small')
axes[2].set_title('by worker status', loc='left', size='small')
# footnotes
plt.figtext(0, -0.05, "Source: Socio-economic impacts of COVID-19\namong (in)formal migrant workers in Laos,\nTotal n=" +
            str(len(clean['group_gender'])), size='x-small',  ha="left", color='gray')
fig.tight_layout()
fig.savefig(graphs_path/'group_descr.svg', bbox_inches='tight')
#


# socio demographic profile.

groupnames={'Male':'Men', 'Female': 'Women', 'Non-migrant':'Non-\nmigrants' , 'Internal migrant': 'Internal\nmigrants' ,
       'Cross-border migrant':'Cross-border\nmigrants', 'Formal employment':'Formal\nemployment', 'Informal employment':'Informal\nemployment', 'Total':'Total'}

gr_title_coldict = {
    'Male': '#0B9CDA',
    'Female': '#53297D',
    'Cross-border migrant': '#630235',
    'Internal migrant': '#0C884A',
    'Non-migrant': '#E70052',
    'Formal employment': '#61A534',
    'Informal employment': '#F16E22',
    'Total': '#000000'
}

title_anno_opts = dict(xy=(0.5, 0.5), size='small', xycoords='axes fraction',
                       va='center', ha='center')

# add a total columns to clean
clean['Total'] = 'Total'


varlabel_df = pd.read_excel(labels_path/"Variable_Labels_clean_IMWLaos.xlsx",
                            usecols=['name', 'varlab'], index_col='name')


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




groupbyvars = [c for c in clean.columns if c.startswith('group')]+['Total']

sns.set_style('white')
fig, axes = plt.subplots(nrows=4, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 6), gridspec_kw={'height_ratios': [1, 7, 5,6]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

# educ
educaxes = fig.axes[8:16]
educ = [c for c in clean.columns if c.startswith('out_educ_')]
data = outcome_bygroup_df(clean, educ, groupbyvars)
for i, (ax, group) in enumerate(zip(educaxes, data.columns)):
    ax = educaxes[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # remove spines)
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('Level of\neducation', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)
# marital status
maritalaxes = fig.axes[16:24]
marital = ['out_marital_1', 'out_marital_2',
           'out_marital_3', 'out_marital_4', 'out_marital_5']
data = outcome_bygroup_df(clean, marital, groupbyvars)
for i, (ax, group) in enumerate(zip(maritalaxes, data.columns)):
    ax = maritalaxes[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # remove spines)
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('Marital\nstatus', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)
# age
ageaxes = fig.axes[24:32]
agecats = ['out_age_1',
           'out_age_2',
           'out_age_3',
           'out_age_4',
           'out_age_5',
           'out_age_6']
data = outcome_bygroup_df(clean, agecats, groupbyvars)
for i, (ax, group) in enumerate(zip(ageaxes, data.columns)):
    ax = ageaxes[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('Age\ncategory', fontstyle='oblique')
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
fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos, total n=" +
            str(len(clean['out_age_6'].dropna()))+".", size='x-small',  ha="left", color='gray')

fig.savefig(graphs_path/'demographics_educ_marital_age.svg', bbox_inches='tight')


###household of origin rural &  lived outside. 
fig, axes = plt.subplots(nrows=3, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 4), gridspec_kw={'height_ratios': [1, 2, 3]})
titleaxes = fig.axes[:8]
# title row.
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

#out_hh_origin
originaxes=fig.axes[8:16]
hhorigin = ['out_hh_origin_1','out_hh_origin_2']
data = outcome_bygroup_df(clean, hhorigin, groupbyvars)
for i, (ax, group) in enumerate(zip(originaxes, data.columns)):
    ax = originaxes[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('hh-origin\nurbanisation', fontstyle='oblique')
    # x-axis settings
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_size('x-small')
    sns.despine(ax=ax)
#out_migrant
out_migrantaxes=fig.axes[16:24]
out_migrant = ['out_migrant_1',
'out_migrant_2',
'out_migrant_3']
data = outcome_bygroup_df(clean, out_migrant, groupbyvars)
for i, (ax, group) in enumerate(zip(out_migrantaxes, data.columns)):
    ax = out_migrantaxes[i]
    ax.set_xlim(0, 1)
    ax.barh(y=data.index, width=data[group], color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('lived outside\nhh-origin', fontstyle='oblique')
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
fig.text(0, 0, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos, total n=" +
            str(len(clean['out_migrant_3'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'rural_urban_lived_outside_hhorigin.svg', bbox_inches='tight')





###out_migrant_where
outcomes1 = ['out_migrant_where_1',
'out_migrant_where_2',
'out_migrant_where_3',
'out_migrant_where_4',
'out_migrant_where_5',
'out_migrant_where_6',
'out_migrant_where_7']

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
        ax.set_ylabel(' lived most of the time in:', fontstyle='oblique')
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
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos, total n=" +
            str(len(clean['out_migrant_where_7'].dropna()))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_migrant_where.svg', bbox_inches='tight')



def outcome_bygroup_median_df(df, outcomes, groupbyvars):
    """returns dataframe with median of groupbyvars in rows (gender, migrant, informal, total) median in columns

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
    data=pd.DataFrame()
    colselect = groupbyvars + outcomes
    bygender = df.loc[:, colselect].groupby('group_gender')[outcomes].median()
    bymigrant = df.loc[:, colselect].groupby('group_migrant')[
        outcomes].median()
    byinformal = df.loc[:, colselect].groupby('group_informal')[
        outcomes].median()
    bytotal = df.loc[:, colselect].groupby('Total')[outcomes].median()
    for df in [bygender, bymigrant,byinformal, bytotal]:
        df['labels']=df.index.values
       

    data = data.append([bygender,bymigrant, byinformal, bytotal], ignore_index=True).set_index('labels')
    #data.columns = gr_title_coldict.keys()
    #data['label'] = varlabel_df.loc[outcomes]
    #data = data.set_index('label')
    return data





##disability status

###out_migrant_where
outcomes1 =['disability_visual',
'disability_hearing',
'disability_speech',
'disability_mobility',
'disability_cognitive',
'disability_stature',
'disability_health',
'disability_no',
'disability_rta']

#recode outcomes. 
for c in outcomes1: 
    print(clean[c].value_counts(dropna=False))
    print(clean[c].cat.codes.value_counts(dropna=False))
    if clean[c].dtype=='category': 
        clean[c]=clean[c].cat.codes.replace({-1:0, 0:1})
        print(clean[c].value_counts(dropna=False))

fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 9]})
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
        ax.set_ylabel('Disability:', fontstyle='oblique')
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
#calculate total nr. 
tots=clean[outcomes1].sum()

fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos," + str(tots[:-2].sum())+" respondents reported to have disability.", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'disability_status.svg', bbox_inches='tight')




###lollipops with average for dependency in origin and destination. 
sns.set_style('white')
outcomes1=['av_dependants_origin'] ##dependents origin
outcomes2=['av_dependants_destination'] ##dependants destination
fig, axes = plt.subplots(nrows=3, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 0.5, 0.5]})

# title row.
titleaxes = fig.axes[:8]
for i, (ax, group) in enumerate(zip(titleaxes, groupnames.keys())):
    ax = titleaxes[i]
    ax.annotate(groupnames[group], **title_anno_opts, color=gr_title_coldict[group])
    # remove spines
    ax.axis('off')

outcomeaxis1=fig.axes[8:16]
data = outcome_bygroup_df(clean, outcomes1, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis1[i]
    # x-axis settings
    ax.set_xlim(0, 5)
    #ylabel
    ax.set_yticklabels(['dependants\nat origin'])

    #plot
    ax.scatter(y=data.index, x=data[group], s=8, color=gr_title_coldict[group])
    ax.hlines(y=data.index, xmin=0, xmax=data[group], color=gr_title_coldict[group])
    # labels
    # get_width pulls left or right; get_y pushes up or down
    ax.text(x=data[group][0]+0.4,y=data.index, s=str(round(data[group][0],1)), color=gr_title_coldict[group], verticalalignment='center', size='xx-small')
    
    sns.despine(ax=ax)

outcomeaxis2=fig.axes[16:24]
data = outcome_bygroup_df(clean, outcomes2, groupbyvars)
for i, (ax, group) in enumerate(zip(outcomeaxis1, data.columns)):
    ax = outcomeaxis2[i]
    # x-axis settings
    ax.set_xlim(0, 5)
    #yticklabel
    ax.set_yticklabels(['dependants\nat destination'])
    #plot
    ax.scatter(y=data.index, x=data[group], s=8, color=gr_title_coldict[group])
    ax.hlines(y=data.index, xmin=0, xmax=data[group], color=gr_title_coldict[group])
    # labels
    # get_width pulls left or right; get_y pushes up or down
    ax.text(x=data[group][0]+0.4,y=data.index, s=str(round(data[group][0],1)), color=gr_title_coldict[group], verticalalignment='center', size='xx-small')
    
    ax.set_xlabel('avg\nno.\nof people', size='x-small', color='gray')
    #ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    # ticklabels
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(0)
        label.set_size('x-small')
    sns.despine(ax=ax)
fig.align_ylabels(fig.axes)
fig.subplots_adjust(hspace = 0.1)
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos, total n=" +
            str(len(clean['av_dependants_destination'].dropna()))+".", size='x-small',  ha="left", color='gray')

fig.savefig(graphs_path/'dependants_origin_dest.svg', bbox_inches='tight')




##disability status

###out_stay_covid_1

outcomes1 = ['out_stay_covid_1',
'out_stay_covid_2',
'out_stay_covid_3',
'out_stay_covid_4',
'out_stay_covid_5']



fig, axes = plt.subplots(nrows=2, ncols=8, sharey='row', sharex='col', figsize=(
    6.25, 3), gridspec_kw={'height_ratios': [1, 5]})
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
    ax.barh(y=data.index, width=data[group],color=gr_title_coldict[group])
    # labels
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+0.05, p.get_y()+0.5, "{:.0%}".format(
            p.get_width()), color=p.get_facecolor(), verticalalignment='top', size='xx-small')
    if i == 0:
        ax.set_ylabel('During Covid-19\nstayed at:', fontstyle='oblique')
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
fig.text(0, -0.1, "Source: Socio-economic impacts of COVID-19 among (in)formal migrant workers in Laos, total n=" +
            str(len(clean['out_stay_covid_5']))+".", size='x-small',  ha="left", color='gray')
fig.subplots_adjust(wspace = 0.5)
fig.savefig(graphs_path/'out_stay_covid.svg', bbox_inches='tight')



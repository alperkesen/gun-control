# Import libraries
import re
import datetime
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

QUERIES = [
    'gun control',
    'gun reform',
    'ban gun',
    'gun laws',
    'gun restrict',
    'gun problem',
    'gun rights',
    'gun policy',
    'gun legislation',
    'gun regulation',
    'pro gun',
    'anti gun',
    'gun ownership',
    'gun politics',
]


def get_tokens(df: pd.DataFrame):
    all_tokens = ""

    for row in df.itertuples():
        text_split = re.sub("[^a-zA-Z0-9\s]+", " ", row.text).split(' ')
        tokens = [token.lower() for token in text_split]

        all_tokens += " ".join(tokens) + " "

    return all_tokens


def get_comment_counts(df,
                       start_date,
                       day_count=21):
    dates = [start_date + datetime.timedelta(days=n) for n in range(day_count + 1)]
    dates_strftime = [date.strftime("%d-%m-%Y") for date in dates]

    comment_counts = list()

    for i in range(len(dates)-1):
        df_between = df[(df.created_at > dates[i].timestamp()) &
                        (df.created_at < dates[i+1].timestamp())]
        num_comments = len(df_between)
        comment_counts.append(num_comments)

    return comment_counts, dates_strftime


def plot_popular_subreddits(df,
                            label="all shootings",
                            k=10,
                            figsize=(5, 2)):
    df['subreddit'].value_counts()[:k][::-1].plot(
        title="Popular subreddits for {}".format(label),
        kind='barh',
        figsize=figsize,
    )
    plt.savefig('plots/popular_subreddits.png', bbox_inches='tight', dpi=1000)
    plt.show()


def plot_popular_shootings(df,
                           figsize=(3, 1)):
    df['shooting'].value_counts()[::-1].plot(
        title="Popularity of all shootings",
        kind='barh',
        figsize=figsize
    )
    plt.savefig('plots/popular_shootings.png', bbox_inches='tight', dpi=1000)
    plt.show()

def plot_popular_queries(df,
                         label="all shootings",
                         k=15,
                         figsize=(15,5)):
    df['query'].value_counts()[:k][::-1].plot(
        kind='barh',
        title="Popular queries for {}".format(label),
        figsize=figsize
    )
    plt.savefig('plots/popular_queries.png', bbox_inches='tight', dpi=1000)
    plt.show()


def plot_comment_counts(comment_counts,
                        dates,
                        label='',
                        figsize=(40,20)):
    xi = list(range(len(dates[:-1])))

    plt.figure(figsize=figsize)
    plt.plot(xi, comment_counts, marker=',', linestyle='-', color='orange', label=shooting_name) 
    plt.xticks(xi, dates[:-1], rotation=60)
    plt.legend()
    plt.grid(axis='y')

    plt.title('Number of comments for {} shooting between 7 days before and 14 days later'.format(label))
    plt.xlabel('Dates')
    plt.ylabel('Number of comments')

    plt.savefig('plots/{}_comment_counts.png'.format(shooting_name), bbox_inches='tight', dpi=300)


def plot_comment_counts_dict(comment_counts_dict,
                             label='',
                             figsize=(8,4)):
    plt.figure(figsize=figsize)

    dates = ["{} days before".format(i) for i in range(7, 1, -1)]
    dates += ["1 day before", "Shooting day", "1 day after"]
    dates += ["{} days after".format(i) for i in range(2, 14)]

    xi = list(range(len(dates)))

    for shooting_name, comment_counts in comment_counts_dict.items():        
        plt.plot(xi, comment_counts, marker=',', linestyle='-', label=shooting_name)

    plt.xticks(xi, dates, rotation=60)
    plt.legend()
    plt.grid(axis='y')

    plt.title('Number of comments for different shootings between 7 days before and 14 days later'.format(label))
    plt.xlabel('Dates')
    plt.ylabel('Number of comments')

    plt.savefig('plots/{}_comment_counts_dict.png'.format(shooting_name), bbox_inches='tight', dpi=1000)


def plot_pie_chart(labels,
                   sizes,
                   explode=(0.1, 0, 0),
                   view='',
                   shooting='',
                   figsize=(0.5, 0.5)):
    plt.figure(figsize=figsize)
    fig1, ax1 = plt.subplots()
    colors = ['#ff9999', '#ffcc99', '#66b3ff']
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors,
            shadow=False, startangle=90, textprops={'fontsize': 14})
    ax1.axis('equal')

    plt.savefig('plots/{}_{}_pie.png'.format(view, shooting), bbox_inches='tight', dpi=1200)


def plot_sentiments_by_shooting(df):
    df2 = df.groupby(['shooting', 'labels']).size().reset_index(name='count')
    df3 = pd.pivot(df2, index='shooting', columns='labels', values='count').reset_index()
    df4 = df3.set_index('shooting')
    df5 = df4.div(df4.sum(axis=1), axis=0)
    df5 = df5.sort_values(['Negative'], ascending=[True])

    colors = ['#ff9999', '#ffcc99', '#66b3ff']

    df5.plot(
        kind='barh', 
        stacked=True, 
        title='Percentage of sentiments for each shooting', 
        mark_right=True,
        colors=colors,
    )

    plt.savefig('plots/sentiments_by_shooting.png', bbox_inches='tight', dpi=1200)

    plt.show()


def plot_sentiments_by_view(df):
    political_orientations = ['Liberal', 'Conservative', 'Moderate']
    dfs = list()

    for orientation in political_orientations:
        df_orientation = df[df['subreddit'].str.contains(orientation)]
        df_orientation_cap = df[df['subreddit'].str.contains(orientation.lower())]
        df_orientation = pd.concat([df_orientation, df_orientation_cap])
        df_orientation['view'] = orientation

        dfs.append(df_orientation)

    combined_df = pd.concat(dfs)

    df2 = combined_df.groupby(['view', 'labels']).size().reset_index(name='count')
    df3 = pd.pivot(df2, index='view', columns='labels', values='count').reset_index()
    df4 = df3.set_index('view')
    df5 = df4.div(df4.sum(axis=1), axis=0)
    df5 = df5.sort_values(['Negative'], ascending=[True])

    colors = ['#ff9999', '#ffcc99', '#66b3ff']

    df5.plot(
        kind='barh', 
        stacked=True, 
        title='Percentage of sentiments for each political view',
        mark_right=True,
        colors=colors,
    )
    plt.legend(loc='upper left', title ="labels")
    plt.savefig('plots/sentiments_by_view.png', bbox_inches='tight', dpi=1200)

    plt.show()


def plot_word_cloud(tokens,
                    label="",
                    figsize=(10, 10),
                    width=400,
                    height=400,
                    min_font_size=10):
    fig = plt.figure(figsize=figsize, facecolor=None)

    queries = set(sum([query.split() for query in QUERIES], []))

    stop_words = ['&gt', 'gt', 'don', 't', 's', 'people', 'thing', 'think', 'guns', 're']
    stop_words += ['firearm', 'firearms', 'u', 'one']
    
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=set(STOPWORDS).union(queries, set(stop_words)),
        min_font_size=min_font_size).generate(tokens)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('plots/{}_word_cloud.png'.format(label), bbox_inches='tight', dpi=1000)


if __name__ == '__main__':
    df = pd.read_csv('data/entire_v2.csv')

    dates_dict = {
        'Uvalde': datetime.datetime(2022, 5, 24, 0, 0) - datetime.timedelta(days=7),
        'El Paso': datetime.datetime(2019, 8, 3, 0, 0) - datetime.timedelta(days=7),
        'Las Vegas': datetime.datetime(2017, 10, 1, 0, 0) - datetime.timedelta(days=7),
        'Sandy Hook': datetime.datetime(2012, 12, 14, 0, 0) - datetime.timedelta(days=7),
    }

    shootings_dict = {
        'uvalde': 'Uvalde',
        'elpaso': 'El Paso',
        'vegas': 'Las Vegas',
        'sandyhook': 'Sandy Hook'
    }

    run_plots = [False, False, False, False, False, False, True, False]

    if run_plots[0]:
        print("1. Plot popular subreddits and queries")
        plot_popular_subreddits(df)
        plot_popular_queries(df)
        plot_popular_shootings(df)

    if run_plots[1]:
        print("2. Plot number of comments in different plots")
        for shooting_name, start_date in dates_dict.items():
            comment_counts, dates = get_comment_counts(df, start_date=start_date)
            plot_comment_counts(comment_counts, dates, label=shooting_name)

    if run_plots[2]:
        print("3. Plot number of comments in one plot")
        comment_counts = [get_comment_counts(
            df[df.shooting == shooting], start_date=dates_dict[shootings_dict[shooting]])[0]
                          for shooting in shootings_dict.keys()]
        comment_counts_dict = dict(zip(dates_dict.keys(), comment_counts))
        plot_comment_counts_dict(comment_counts_dict)

    political_orientations = ['liberal', 'conservative', 'moderate']

    if run_plots[3]:
        print("4. Plot word clouds for each political orientation")
        for orientation in political_orientations:
            df_orientation = df[df['subreddit'].str.contains(orientation)]
            df_orientation_cap = df[df['subreddit'].str.contains(orientation.capitalize())]
            df_orientation = pd.concat([df_orientation, df_orientation_cap])

            print(orientation, len(df_orientation))
            tokens = get_tokens(df_orientation)
            plot_word_cloud(tokens, label=orientation)

    if run_plots[4]:
        print("5. Plot sentiment analysis for each shooting and orientation")
        for orientation in political_orientations:
            df_orientation = df[df['subreddit'].str.contains(orientation)]
            df_orientation_cap = df[df['subreddit'].str.contains(orientation.capitalize())]
            df_orientation = pd.concat([df_orientation, df_orientation_cap])

            for shooting in shootings_dict.keys():
                df_shooting = df_orientation[df.shooting == shooting]
                
                labels = ['Negative', 'Neutral', 'Positive']
                sizes = [sum(df_shooting['labels'] == label) for label in labels]

                print(orientation, shooting, labels, sizes)
                plot_pie_chart(labels, sizes, view=orientation, shooting=shooting)

    if run_plots[5]:
        print("6. Plot sentiment analysis for each shooting")
        df['shooting'] = df.shooting.apply(lambda x: x.capitalize()).replace('Sandyhook', 'Sandy Hook').replace('Elpaso', 'El Paso')
        plot_sentiments_by_shooting(df)
        """
        for shooting in shootings_dict.keys():
            df_shooting = df[df.shooting == shooting]
        
            labels = ['Negative', 'Neutral', 'Positive']
            sizes = [sum(df_shooting['labels'] == label) for label in labels]
        
            print(shooting, labels, sizes)
            plot_pie_chart(labels, sizes, view="all", shooting=shooting)
        """

    if run_plots[6]:
        print("7. Plot sentiment analysis for each orientation")
        plot_sentiments_by_view(df)

        """
        for orientation in political_orientations:
            df_orientation = df[df['subreddit'].str.contains(orientation)]
            df_orientation_cap = df[df['subreddit'].str.contains(orientation.capitalize())]
            df_orientation = pd.concat([df_orientation, df_orientation_cap])

            labels = ['Negative', 'Neutral', 'Positive']
            sizes = [sum(df_orientation['labels'] == label) for label in labels]

            print(orientation, labels, sizes)
            plot_pie_chart(labels, sizes, view="all", shooting=orientation)
        """
    if run_plots[7]:
        print("8. Print overall sentiment")
        labels = ['Negative', 'Neutral', 'Positive']
        sizes = [sum(df['labels'] == label) for label in labels]

        plot_pie_chart(labels, sizes, view="all", shooting="all")


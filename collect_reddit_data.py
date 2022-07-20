# Import libraries

import sys
import praw
import time
import datetime
import requests
import pandas as pd
import datetime
import argparse
from json import load, dump, JSONDecodeError
from typing import List, Dict


parser = argparse.ArgumentParser(
    description="Collect reddit data"
)

parser.add_argument("--query", type=str, default=None)
parser.add_argument("--subreddit", type=str, default=None)
parser.add_argument("--hours", type=float, default=1/2)
parser.add_argument("--start_date", type=str, default="2022-5-17")
parser.add_argument("--day_count", type=int, default=21)
parser.add_argument("--filename", type=str, default="data")


example_url = "https://api.pushshift.io/reddit/search/comment/"
example_url += "?q=gun+control&subreddit=conservative&before=21d&after=31d"
example_url += "&size=100"

# Get url and search query

example_url = "https://api.pushshift.io/reddit/search/comment/"
example_url += "?q=gun+control&subreddit=conservative&before=21d&after=31d"
example_url += "&size=100"

def get_url(query: str, 
            subreddit: str,
            before: datetime.datetime,
            after: datetime.datetime,
            size=500):
    main_url = "https://api.pushshift.io/reddit/search/comment/"
    url = main_url
    url += "?q={}".format("+".join(query.split()))

    if subreddit:
        url += "&subreddit={}".format(subreddit)

    url += "&before={}".format(int(before.timestamp()))
    url += "&after={}".format(int(after.timestamp()))
    url += "&size={}".format(size)

    return url

#  end_date = datetime.datetime(2022, 3, 5, 0, 0)

def get_comments(query: str,
                 subreddit: str,
                 start_date: datetime.datetime,
                 day_count=14,
                 hours=6,
                 verbose=True):
    intervals = [start_date + datetime.timedelta(
        hours=n * hours) for n in range(int(day_count * 24 / hours))]
    comments = list()

    for i, current in enumerate(intervals):
        after = current + datetime.timedelta(hours=hours)
        url = get_url(query, subreddit, after, current)

        response = None

        try:
            response = requests.get(url).json()
        except:
            print("Error! Wait 1.5 sec...")
            time.sleep(4)
            response = requests.get(url).json()

        if verbose:
            print("Url:", url)
            print("Current:", current, "After:", after, "i:", i, "len:", len(response['data']))

        for comment in response['data']:
            comments.append((comment['id'],
                             comment['parent_id'],
                             comment['body'],
                             comment['score'],
                             comment['created_utc'],
                             comment['subreddit'],
                             query))
    return comments


def get_comments_from_all(queries: List[str],
                          subreddits: List[str],
                          start_date: datetime.datetime,
                          day_count=14,
                          hours=6,
                          verbose=True):
    comments = list()

    for subreddit in subreddits:
        for query in queries:
            comments += get_comments(query,
                                     subreddit,
                                     start_date,
                                     day_count=day_count,
                                     hours=hours,
                                     verbose=verbose)
    
    return comments


def create_df(comments: List[tuple]):
    columns = ['id', 'parent_id', 'text', 'score', 'created_at', 'subreddit', 'query']
    df = pd.DataFrame(comments, columns=columns)

    return df


def main(args):
    year, month, day = [int(val) for val in args.start_date.split('-')]
    start_date = datetime.datetime(year, month, day, 0, 0)

    queries = [
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

    if args.query:
        queries = [args.query]

    if args.subreddit:
        subreddits = [args.subreddit]
    else:
        subreddits = [None]

    comments = get_comments_from_all(queries,
                                     subreddits,
                                     start_date,
                                     day_count=args.day_count,
                                     hours=args.hours)
    df = create_df(comments)
    df.to_csv('data/{}.csv'.format(args.filename), index=False)

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)

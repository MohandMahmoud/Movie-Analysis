import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import warnings

warnings.filterwarnings('ignore')
from IPython.core.display import HTML as Center

Center(""" <style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style> """)


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%\n({v:d})'.format(p=pct, v=val)

    return my_autopct


def setting_dataframe(df):
    df = df.style.set_table_styles([{"selector": "td, th", "props": [("border", "1px solid grey !important")]},
                                    {"selector": "th", "props": [('text-align', 'center')]}])
    df = df.set_properties(**{'text-align': 'center'}).hide_index()
    return df


netflix_df = pd.read_csv("netflix_titles.csv")
netflix_df.head()
netflix_df.set_index("show_id", inplace=True)
netflix_df.head()
data_size = netflix_df.shape
print("This data has {} entries and {} features(columns).".format(data_size[0], data_size[1]))
primary_key = netflix_df.index.name
columns = netflix_df.columns
print("The primary key or the index of this data is ({}) and its columns are: ".format(primary_key))
for idx, column in enumerate(columns):
    print("{}) {}".format(idx + 1, column), end="\n")
netflix_df.head()
netflix_df["title"].tail(10)
netflix_df.rename(columns={"rating": "MPA_rating"}, inplace=True)
print(netflix_df.dtypes)
netflix_df['type'].value_counts()
netflix_df['country'].value_counts()
netflix_df['release_year'].value_counts()
netflix_df['MPA_rating'].value_counts()
netflix_df['listed_in'].value_counts()
duplicateRows = netflix_df[netflix_df.duplicated(["title"])]
print(duplicateRows)
print(netflix_df[netflix_df["MPA_rating"].isnull()])
netflix_df.loc[netflix_df['title'] == "13TH: A Conversation with Oprah Winfrey & Ava DuVernay", 'MPA_rating'] = "PG-13"
netflix_df.loc[netflix_df['title'] == "Gargantia on the Verdurous Planet", 'MPA_rating'] = "TV-14"
netflix_df.loc[netflix_df['title'] == "Little Lunch", 'MPA_rating'] = "TV-MA"
netflix_df.loc[netflix_df['title'] == "Louis C.K. 2017", 'MPA_rating'] = "TV-MA"
netflix_df.loc[netflix_df['title'] == "Louis C.K.: Hilarious", 'MPA_rating'] = "NR"
netflix_df.loc[netflix_df['title'] == "Louis C.K.: Live at the Comedy Store", 'MPA_rating'] = "NC-17"
netflix_df.loc[netflix_df['title'] == "My Honor Was Loyalty", 'MPA_rating'] = "PG-13"
netflix_df["MPA_rating"].isnull().sum()
print(netflix_df[netflix_df['date_added'].isnull()])
netflix_df.loc[netflix_df['title'] == "A Young Doctor's Notebook and Other Stories", 'date_added'] = "October 2, 2013"
netflix_df.loc[netflix_df['title'] == "Anthony Bourdain: Parts Unknown", 'date_added'] = "April 14, 2013"
netflix_df.loc[netflix_df['title'] == "Frasier", 'date_added'] = "September 23, 2003"
netflix_df.loc[netflix_df['title'] == "Friends", 'date_added'] = "September 25, 2003"
netflix_df.loc[netflix_df['title'] == "Gunslinger Girl", 'date_added'] = "January 7, 2008"
netflix_df.loc[netflix_df['title'] == "Kikoriki", 'date_added'] = "2010"
netflix_df.loc[netflix_df['title'] == "La Familia P. Luche", 'date_added'] = "July 8, 2012"
netflix_df.loc[netflix_df['title'] == "Maron", 'date_added'] = "May 4, 2016"
netflix_df.loc[netflix_df['title'] == "Red vs. Blue", 'date_added'] = "April 1, 2015"
netflix_df.loc[netflix_df['title'] == "The Adventures of Figaro Pho", 'date_added'] = "2015"
netflix_df["date_added"].isnull().sum()
null_countires = netflix_df['country'].isnull().sum()
print("The number of entries which have no country (Null) = {}\
      \nThe percentage between those entries and the total entries is {} %".format(null_countires, round(
    null_countires / data_size[0] * 100, 2)))
netflix_df = netflix_df[netflix_df['country'].notna()]

netflix_df["country"].isnull().sum()
netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'])
print(netflix_df.dtypes)
netflix_df.head()
tv_vs_movies = netflix_df['type'].value_counts()
print(tv_vs_movies)
tv_vs_movies = tv_vs_movies.to_list()
figure = plt.figure(figsize=(10, 10))
_, _, autotexts = plt.pie(tv_vs_movies,
                          labels=['Movies', 'TV-Shows'],
                          autopct=make_autopct(tv_vs_movies), colors=["indianred", "orange"],
                          textprops={"fontsize": 13, "fontname": "monospace"},
                          wedgeprops={"edgecolor": "white", 'linewidth': 1, 'antialiased': True})
for autotext in autotexts:
    autotext.set_color('white')
plt.title("Percentage of movies and TV shows", fontsize=15, fontweight="bold", fontname="monospace", y=0.945)
plt.show()
number_of_contents = netflix_df.groupby('release_year').size().to_list()
years = np.sort(netflix_df['release_year'].unique())
max_number_of_contents = max(number_of_contents)
index = number_of_contents.index(max(number_of_contents))
figure = plt.figure(figsize=(20, 8))
plt.plot(years, number_of_contents, color="indianred", marker='o', markersize=6)
plt.scatter(years[index], max_number_of_contents, s=200, color="indianred", marker='o')

plt.title("The growth of content creation over the years", fontsize=15, fontweight="bold", fontname="monospace", y=1.05)
plt.suptitle("Maximum point are represented as big {}".format(r'$\bullet$'),
             fontsize=13, fontname="monospace", y=0.915)
plt.xticks(fontsize=13, fontname="monospace")
plt.yticks(fontsize=13, fontname="monospace")
plt.show()
netflix_df['month_added'] = pd.DatetimeIndex(netflix_df['date_added']).month
netflix_df.head()
number_of_contents = netflix_df.groupby('month_added').size().to_list()
months_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
max_number_of_contents = max(number_of_contents)
index = number_of_contents.index(max(number_of_contents))
figure = plt.figure(figsize=(20, 8))
plt.plot(months_names, number_of_contents, color="indianred", marker='o', markersize=6)
plt.scatter(months_names[index], max_number_of_contents, s=200, color="indianred", marker='o')

plt.title("Number of Content Creations along the months", fontsize=15, fontweight="bold", fontname="monospace", y=1.05)
plt.suptitle("Maximum point are represented as big {}".format(r'$\bullet$'),
             fontsize=13, fontname="monospace", y=0.915)
plt.ylabel("Number of Content Creations", fontsize=14, fontname="monospace")
plt.xticks(fontsize=13, fontname="monospace")
plt.yticks(fontsize=13, fontname="monospace")
plt.ylim([0, 900])
plt.show()
oldest_tv_series = netflix_df[netflix_df['type'] == "TV Show"][['title', 'release_year']].sort_values(
    by="release_year").head(10)
setting_dataframe(oldest_tv_series.reset_index())
oldest_movies = netflix_df[netflix_df['type'] == "Movie"][['title', 'release_year']].sort_values(
    by="release_year").head(10)
setting_dataframe(oldest_movies.reset_index())


def count_countires(countries_df):
    countries_dict = dict()
    for country_entry in countries_df:
        countries_list = country_entry.split(', ')
        for country in countries_list:
            if country in countries_dict.keys():
                countries_dict[country] += 1
            else:
                countries_dict[country] = 1
    return countries_dict


movies_countries = netflix_df[netflix_df["type"] == "Movie"]["country"]
shows_countries = netflix_df[netflix_df["type"] == "TV Show"]["country"]

movies_dict = count_countires(movies_countries)
shows_dict = count_countires(shows_countries)


movies = pd.DataFrame(movies_dict.items(), columns=["Country", "Frequency"]).sort_values(by="Frequency",
                                                                                         ascending=False).reset_index(
    drop=True)

shows = pd.DataFrame(shows_dict.items(), columns=["Country", "Frequency"]).sort_values(by="Frequency",
                                                                                       ascending=False).reset_index(
    drop=True)

print(setting_dataframe(movies))

print(setting_dataframe(shows))
all_contents = movies.merge(shows, how="outer", on="Country").head(10)
all_contents.rename(columns={"Frequency_x": "Number of Movies", "Frequency_y": "Number of TV Shows"}, inplace=True)
all_contents = all_contents.astype({"Number of Movies": 'int64', "Number of TV Shows": 'int64'})
print(setting_dataframe(all_contents))

num_of_movies = all_contents["Number of Movies"]
num_of_shows = all_contents["Number of TV Shows"]

ind = np.arange(len(num_of_movies))
width = 0.4

figure = plt.figure(figsize=(15, 10))

plt.barh(ind, num_of_movies, width, color="orange", label="Movies")
plt.barh(ind + width, num_of_shows, width, color="indianred", label="TV Shows")

plt.title("The top 10 countires contributed in Movies creation VS. TV Shows creation", fontname="monospace",
          fontsize=15, fontweight="bold", y=1.07)
plt.suptitle("(The top 10 countires create {} contents out of {} on their own)"
             .format(sum(netflix_df.groupby('country').size().sort_values(ascending=False)[:10]), netflix_df.shape[0]),
             y=0.92)
plt.ylabel("Country", fontsize=14, fontname="monospace")
plt.xlabel("Number of Content Creations", fontsize=14, fontname="monospace")
plt.yticks(ind + 0.2, labels=all_contents["Country"], fontsize=13, fontname="monospace")
plt.xticks(fontsize=13, fontname="monospace")
plt.gca().invert_yaxis()
plt.legend(fontsize="medium")
plt.show()
categories_df = netflix_df.loc[netflix_df['listed_in'].notnull()]['listed_in']

categories_dict = dict()
for category_entry in categories_df:
    categories_list = category_entry.split(', ')
    for category in categories_list:
        if category in categories_dict.keys():
            categories_dict[category] += 1
        else:
            categories_dict[category] = 1

results = pd.DataFrame(categories_dict.items(), columns=["Category", "Frequency"]).sort_values(by="Frequency",
                                                                                               ascending=False).reset_index(
    drop=True)
print(setting_dataframe(results.head(10)))

movies_categories_df = netflix_df[netflix_df.loc[netflix_df['listed_in'].notnull()]["type"] == "Movie"]['listed_in']

movies_categories_dict = dict()
for category_entry in movies_categories_df:
    categories_list = category_entry.split(', ')
    for category in categories_list:
        if category in movies_categories_dict.keys():
            movies_categories_dict[category] += 1
        else:
            movies_categories_dict[category] = 1

# to see the results in a dataframe
movies = pd.DataFrame(movies_categories_dict.items(), columns=["Category", "Frequency"]).sort_values(by="Frequency",
                                                                                                     ascending=False).reset_index(
    drop=True)
print(setting_dataframe(movies.head(10)))

shows_categories_df = netflix_df[netflix_df.loc[netflix_df['listed_in'].notnull()]["type"] == "TV Show"]['listed_in']

shows_categories_dict = dict()
for category_entry in shows_categories_df:
    categories_list = category_entry.split(', ')
    for category in categories_list:
        if category in shows_categories_dict.keys():
            shows_categories_dict[category] += 1
        else:
            shows_categories_dict[category] = 1

# to see the results in a dataframe
shows = pd.DataFrame(shows_categories_dict.items(), columns=["Category", "Frequency"]).sort_values(by="Frequency",
                                                                                                   ascending=False).reset_index(
    drop=True)
print(setting_dataframe(shows.head(10)))

figure = plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)

top_ten_movies_categories = movies.head(5)

colors = ["indianred", "burlywood", "rosybrown", "orange", "wheat"]
plt.pie(top_ten_movies_categories["Frequency"], labels=top_ten_movies_categories["Category"],
        colors=colors,
        autopct=make_autopct(top_ten_movies_categories['Frequency']),
        wedgeprops={"edgecolor": "white",
                    'linewidth': 1,
                    'antialiased': True},
        textprops={"fontname": 'monospace',
                   "fontsize": 12})
plt.title(
    "The most frequent movies categories\n(Total number of contents which are categorized\nby these categories is {} out of {})"
    .format(sum(top_ten_movies_categories["Frequency"]), movies["Frequency"].sum()),
    fontname="monospace", fontsize=15, fontweight="bold")
plt.subplot(1, 2, 2)

top_ten_shows_categories = shows.head(5)

colors = ["indianred", "burlywood", "rosybrown", "orange", "wheat"]
plt.pie(top_ten_shows_categories["Frequency"], labels=top_ten_shows_categories["Category"],
        colors=colors,
        autopct=make_autopct(top_ten_shows_categories['Frequency']),
        wedgeprops={"edgecolor": "white",
                    'linewidth': 1,
                    'antialiased': True},
        textprops={"fontname": 'monospace',
                   "fontsize": 12})
plt.title(
    "The most frequent TV Shows categories\n(Total number of contents which are categorized\nby these categories is {} out of {})"
    .format(sum(top_ten_shows_categories["Frequency"]), shows["Frequency"].sum())
    , fontname="monospace", fontsize=15, fontweight="bold")
plt.ylabel("")
plt.show()
print(netflix_df["MPA_rating"].value_counts().sort_values(ascending=True)[:10])

figure = plt.figure(figsize=(20, 12))
netflix_df["MPA_rating"].value_counts().sort_values()[:10].plot(kind="barh", color="indianred")
plt.title("Number of content creations by MPA rating for top 10 MPA ratings", fontname="monospace", fontsize=15,
          fontweight="bold")
plt.xlabel("Number of Content Creations", fontsize=14, fontname="monospace")
plt.ylabel("MPA Ratings", fontsize=14, fontname="monospace")
plt.xticks(fontsize=13, fontname="monospace")
plt.show()
fav_tv_show = "Friends"
setting_dataframe(netflix_df[netflix_df['title'] == fav_tv_show])



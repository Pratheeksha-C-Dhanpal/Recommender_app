import streamlit as st
import base64
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
from ast import literal_eval
warnings.filterwarnings('ignore')
from IPython.display import HTML
from nltk.stem.snowball import SnowballStemmer
import urllib.request
import plotly.graph_objects as go

@st.cache(allow_output_mutation=True)
def load_data():
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[movies.astype(str)['genres'] != '[]']

    df=movies.copy()
    features = ['keywords', 'genres','production_companies', 'production_countries', 'spoken_languages']
    for i in features:
        df[i] = movies[i].apply(literal_eval)

    def list_genres(x):
        l = [d['name'] for d in x]
        return(l)
    df['genres'] = df['genres'].apply(list_genres)

    def list_keywords(x):
        l = [d['name'] for d in x]
        return(l)
    df['keywords'] = df['keywords'].apply(list_keywords)

    def list_companies(x):
        l = [d['name'] for d in x]
        return(l)
    df['production_companies'] = df['production_companies'].apply(list_companies)

    def list_countries(x):
        l = [d['name'] for d in x]
        return(l)
    df['production_countries'] = df['production_countries'].apply(list_countries)

    def list_languages(x):
        l = [d['name'] for d in x]
        return(l)
    df['spoken_languages'] = df['spoken_languages'].apply(list_languages)

    df1 = credits.copy()
    features = ['cast','crew']
    for i in features:
        df1[i] = credits[i].apply(literal_eval)
    def director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
    df1['crew']=df1['crew'].apply(director)
    df1.rename(columns={'crew':'director'},inplace=True)

    def list_cast(x):
        l = [d['name'] for d in x]
        return(l)
    df1['cast'] = df1['cast'].apply(list_cast)

    df1 = df1[df1.astype(str)['cast'] != '[]']
    df1.rename(columns = {'movie_id':'id'}, inplace = True)
    alldata = df.merge(df1, on = 'id')
    alldata['title_y'].equals(alldata['title_x'])
    alldata['original_title'].equals(alldata['title_x'])
    alldata['year'] = pd.DatetimeIndex(alldata['release_date']).year
    alldata.drop(columns=['status', 'original_title', 'homepage', 'id', 'spoken_languages', 'original_title', 'title_x','release_date' ], axis=1, inplace=True)
    alldata.rename(columns = {'title_y':'title'}, inplace = True)
    #Calculating mean of vote average
    c=alldata['vote_average'].mean()
    m=alldata['vote_count'].quantile(0.6)
    filtered_movies=alldata.copy().loc[alldata['vote_count']>=m]
    def imdbscore(x,m=m,c=c):
        v=x['vote_count']
        r=x['vote_average']
    #remember the imdb formula
        return (v/(v+m) * r) + (m/(m+v) * c)
    alldata['score']=filtered_movies.apply(imdbscore,axis=1)
    alldata['score'] = round(alldata['score'], 3)
    def top5(x):
        if len(x) > 3:
            x = x[:3]
        return(x)
    alldata['cast'] = alldata['cast'].apply(top5)
    new_df=alldata.copy()
    rdf=alldata.copy()
    g = rdf.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    g.name = 'genre'
    gen= rdf.drop('genres', axis=1).join(g)
    gen.transpose()
    new_df['cast'] = new_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    new_df['director'] = new_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    new_df['director'] = new_df['director'].apply(lambda x: [x,x, x])
    new_df = new_df[new_df.astype(str)['cast'] != '[]']
    new_df = new_df[new_df.astype(str)['keywords'] != '[]']
    s = new_df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]
    indices = pd.Series(new_df.index, index=new_df['title'])
    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    stemmer = SnowballStemmer('english')
    new_df['keywords'] = new_df['keywords'].apply(filter_keywords)
    new_df['keywords'] = new_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    new_df['keywords'] = new_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    new_df['allwords'] = new_df['keywords'] + new_df['cast'] + new_df['director'] + new_df['genres']
    new_df['allwords'] = new_df['allwords'].apply(lambda x: ' '.join(x))
    return new_df,alldata,imdbscore,gen

new_df,alldata,imdbscore,gen=load_data()
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(new_df['allwords'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
new_df = new_df.reset_index()
titles = new_df['title']
indices = pd.Series(new_df.index, index=new_df['title'])

@st.cache(allow_output_mutation=True)
def recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    qualified = new_df.iloc[movie_indices][['title', 'year']]
    mov = new_df.iloc[movie_indices][['vote_count', 'vote_average']]
    #qualified = mov[(mov['vote_count'] >= m) & (mov['vote_count'].notnull()) & (mov['vote_average'].notnull())]
    #qualified['vote_count'] = qualified['vote_count'].astype('int')
    #qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified.rename(columns={'title':'Title'},inplace=True)
    qualified.rename(columns={'year':'Year'},inplace=True)
    qualified['Director']=alldata['director']
    qualified['IMDBScore'] = mov.apply(imdbscore, axis=1)
    qualified = qualified.sort_values('IMDBScore', ascending=False).head(10)
    qualified['IMDBScore']=round(qualified['IMDBScore'], 2)
    #qualified['cast']=alldata['cast']

    return qualified


@st.cache(allow_output_mutation=True)
def genre_based(genre, percentile=0.85, n=10):
    df = gen[gen['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    df1 = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) &
                   (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average']]
    df['vote_count'] = df['vote_count'].astype('int')
    df['vote_average'] = df['vote_average'].astype('int')

    df['score'] = df.apply(lambda x:
                        (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
                        axis=1)
    df = df.sort_values('score', ascending=False).head(250)
    df['score']=round(df['score'],2)
    df.rename(columns={'title':'Title'},inplace=True)
    rec=df[['Title']]
    rec['Director']=df['director']
    rec['Year']=df['year']
    rec['ImdbScore']=df['score']
    return rec

@st.cache(allow_output_mutation=True)
def rating_based(score):
    rt=new_df[['title', 'director','year','score']]
    rt['Director']=alldata['director']
    rt.rename(columns={'title':'Title'},inplace=True)
    rt.rename(columns={'year':'Year'},inplace=True)
    rt.rename(columns={'score':'ImdbScore'},inplace=True)
    rt = rt.sort_values('ImdbScore', ascending=False)
    rt['ImdbScore']=round(rt['ImdbScore'],2)

    if score==8:
        result=rt.loc[(rt['ImdbScore']>=8) & (rt['ImdbScore']<=9),['Title','Director', 'Year', 'ImdbScore']]
    #result=rt[rt['score']>=8]
    elif score==7:
        result=rt.loc[(rt['ImdbScore']>=7) & (rt['ImdbScore']<=8),['Title','Director', 'Year', 'ImdbScore']]
    #result=rt[rt['score']>=7]
    elif score==6:
        result=rt.loc[(rt['ImdbScore']>=6) & (rt['ImdbScore']<=7),['Title','Director', 'Year', 'ImdbScore']]
    elif score==5:
        result=rt.loc[(rt['ImdbScore']>=5) & (rt['ImdbScore']<=6),['Title','Director', 'Year', 'ImdbScore']]
    elif score==4:
        result=rt.loc[(rt['ImdbScore']>=4) & (rt['ImdbScore']<=5),['Title','Director', 'Year', 'ImdbScore']]
    elif score==3:
        result=rt.loc[(rt['ImdbScore']>=3) & (rt['ImdbScore']<=4),['Title','Director', 'Year', 'ImdbScore']]
    elif score==2:
        result=rt.loc[(rt['ImdbScore']>=2) & (rt['ImdbScore']<=3),['Title','Director', 'Year', 'ImdbScore']]
    else:
        print('no movies')
    return result.head(30)

st.title('Hollywood Movie Recommendation :movie_camera:')
st.subheader('Recommends movies based on Movie Name, Genre and Ratings!!')



@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpg')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)



select = st.sidebar.radio("Recommendations based on:",('Movie Name', 'Genre', 'Rating'))
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#ffbabc,#001180);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)
local_css("style.css")

if select=='Movie Name':
    st.markdown('**_Type in a movie name :point_down: and I will recommend similar ones for you!!!_**')
    search_movie = st.text_input(" ", "")
    search_movie=search_movie.title()
    if st.button("Recommend"):
        try:
            res=recommendations(search_movie)
            fig = go.Figure(data=[go.Table(
            header=dict(values=list(res.columns),
                        fill_color= '#f5b5b5',
                        align='left'),
            cells=dict(values=[res.Title, res.Year, res.Director, res.IMDBScore],
                       fill_color='#e8ccfc',
                       align='left'))])
            st.write(fig)
        except:
            st.markdown('**_Enter a valid movie name!! _**')
elif select=='Genre':
    st.markdown('**_ :point_down: Select your Favorite Genre!_**')
    option = st.selectbox(' ',('Action', 'Adventure', 'Romance', 'Animation', 'Drama', 'Comedy', 'Thriller', 'Crime', 'Science Fiction', 'Horror', 'Family'))
    #st.write(genre_based(option).head(10))
    gen=genre_based(option).head(10)
    tab = go.Figure(data=[go.Table(
    columnorder = [1,2,3,4],
    columnwidth = [300,180,80,100],
    header=dict(values=list(gen.columns),
                fill_color='#f5b5b5',
                align='left'),
    cells=dict(values=[gen.Title,gen.Director, gen.Year,gen.ImdbScore],
               fill_color='#e8ccfc',
               align='left'))])

    st.write(tab)
elif select=='Rating':
    st.markdown('**_Select the rating range_**:point_down:.')
    range=st.radio('',('above 8','7-8','6-7','5-6','4-5'))
    if range=='above 8':
        r=rating_based(8)
    elif range=='7-8':
        r=rating_based(7)
    elif range=='6-7':
        r=rating_based(6)
    elif range=='5-6':
        r=rating_based(5)
    elif range=='4-5':
        r=rating_based(4)
    res = go.Figure(data=[go.Table(
    columnorder = [1,2,3,4],
    columnwidth = [300,180,80,100],
    header=dict(values=list(r.columns),
                fill_color='#f5b5b5',
                align='left'),
    cells=dict(values=[r.Title,r.Director, r.Year,r.ImdbScore],
               fill_color='#e8ccfc',
               align='left'))])
    st.write(res)


streamlit_style = """
            <style>
            footer {

            	visibility: hidden;

            	}
            footer:after {
            	content:'Made with Streamlit, by Pratheeksha C Dhanpal';
            	visibility: visible;
            	display: block;
            	position: relative;
                color: black;
                font-weight: bold;
                font-style: italic;
                font-size:1.2em;
            	#background-color: red;
            	padding: 8px;
            	top: 2px;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)





#st.success(result)
#button_clicked = st.button("OK")

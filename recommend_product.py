import numpy as np
import pandas as pd
#from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from streamlit_option_menu import option_menu
#from st_clickable_images import clickable_images
import streamlit.components.v1 as components
from PIL import Image
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from gensim import corpora, models, similarities

# 1. Read data
data = pd.read_pickle('model/product_list.pkl')
data = data.reset_index(drop=True)
lst_user = [1,2,3,4,5,6,7,10,935,4429,3769]
lst_products = data['product_name'].unique()
dictionary = pd.read_pickle('model/Dictionary.pkl')
index = pd.read_pickle('model/Index.pkl')

cols=['user_idx','recommendations']
df_new = pd.read_parquet('Recommend_User.parquet', columns=cols)

cols_idx = ['product_id','product_idx']
df_idx = pd.read_parquet('data_index.parquet', columns=cols_idx)



# TFIDF
tfidf = pickle.load(open("model/Tfidf.pkl", "rb")) 

def path_to_image_html(path):
    return '<img src="' + path + '" width="60" >'

def get_recommendations_list_idx(user_id):
    list_product_id = []
    for i in range(10):
        idx = df_new[df_new['user_idx']==user_id]['recommendations'].index.values[0]
        list_product_id.append(df_new[df_new['user_idx']==user_id]['recommendations'][idx][i]['product_idx'])
    return list_product_id

def get_product_info(id):
    product_name= data[data['product_id']==id][['product_id','product_name','link','image','price']]
    return product_name

#Create function to get top 10 relatest products
def top10_relatest_search(text,dictionary):
    data1 = data[["product_id","product_name","price","rating","image"]]
    text = text.lower().split()
    kw_vector = dictionary.doc2bow(text)
    sim = index[tfidf[kw_vector]]
    lst=[]
    dictionary = {i:round(sim[i],2) for i in range(len(sim))}
    sortedDict = sorted(dictionary.items(),key=lambda x:x[1])
    last_key = list(sortedDict)[-10:-1]
    for i in range (len(last_key)):
        lst = np.append(last_key[i][0],lst)      
    df_final =  data1.iloc[lst, :].reset_index(drop=True)
    df_final.drop_duplicates(inplace=True)
    return df_final

def show_image(df):
    lst = df['product_id'].tolist()
    for i in range(10):
        # get image and title of top 10 similar products_id
        lst_link = df[df['product_id'].isin(lst)]['image'].tolist()
        
        # get index of element in lst_link != N/A
        lst_index = [i for i, x in enumerate(lst_link) if str(x) != 'nan']
        # drop N/A value in lst_link
        lst_link = [x for x in lst_link if str(x) != 'nan']
        
        # get list title of lst_link = product_name     
        lst_title = df[df['product_id'].isin(lst)]['product_name'].tolist()
        # get title of lst_link = product_name from list lst_title with index in lst_index
        lst_title = [lst_title[i] for i in lst_index]
        
        # get list product_id of lst_link = product_id
        lst_product_id = df[df['product_id'].isin(lst)]['product_id'].tolist()
        # get product_id of lst_link = product_id from list lst_product_id with index in lst_index
        lst_product_id = [str(lst_product_id[i]) for i in lst_index]
        
        # concat lst_title and lst_product_id
        lst_title = [lst_title[i] + ' - ' + lst_product_id[i] for i in range(len(lst_title))]
        
    st.image(lst_link, 
                width=150, 
                caption=lst_title,
    )

    #return df_final

def show_by_user(id):
    product_lst = pd.unique(df_idx[df_idx['product_idx'].isin(get_recommendations_list_idx(id))]['product_id'])
    data_show=data[data['product_id'].isin(product_lst)].drop_duplicates().reset_index(drop=True)[['product_id','product_name','image','price']]
    st.dataframe(data=data_show)
    show_image(data_show)

def show_by_product_name(name):
    df_by_name = top10_relatest_search(name,dictionary)
    st.dataframe(data=df_by_name)
    show_image(df_by_name)


    


#--------------
# GUI

st.markdown("<h1 style='text-align: center; color: grey;'>Recommendation system</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Products Shopee recommendation </h2>", unsafe_allow_html=True)
st.write('\n\n')

# MENU
menu = ["1.Business Objective", "2.Recommendation"]
choice = option_menu(
    menu_title = None,
    options= menu,
    menu_icon= 'menu-up',
    orientation= 'vertical'
)


if choice == '1.Business Objective':    
    st.write("### 1. Sentiment Analysis")
    recommendation_img = Image.open("image/recommender.png")
    st.image(recommendation_img, width = 700)
    st.markdown("_A recommendation system is an artificial intelligence or AI algorithm, usually associated with machine learning, that uses Big Data to suggest or recommend additional products to consumers. These can be based on various criteria, including past purchases, search history, demographic information, and other factors._")  
    st.write("### 2. Problem/ Requirement")
    st.markdown("_Shopee is an ecosystem 'all-in-one' commercial including shopee.vn. This is a commercial website electronics ranked 1st in Vietnam South and East region South Asia._")
    st.markdown("_-->Use Machine Learning algorithms in Python for recommend customers._")
    


elif choice == '2.Recommendation':
    option = st.radio("Select type",options = ("Content based Filtering by input keyword","Collaborative Filtering from available product name","Collaborative Filtering"))
    if option == "Content based Filtering by input keyword":
        keyword = st.text_area("Input your keyword: ")
        if keyword != "":
            st.write("Some recommended products are:...")
            st.dataframe(data=top10_relatest_search(keyword,dictionary),use_container_width=False)
            show_image(top10_relatest_search(keyword,dictionary))
            #st.markdown(top10_relatest_search(keyword,dictionary).to_html(render_links=True, escape=False),unsafe_allow_html=True)    
    if option == "Collaborative Filtering from available product name":
        option2=st.selectbox("Select name of product:",options=lst_products)
        show_by_product_name(option2)
    if option == "Collaborative Filtering":
        option3=st.selectbox("Select user_id:",options=lst_user)
        show_by_user(option3)
        
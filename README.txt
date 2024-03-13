this is group IR24W-A3 6
Bobby Zhong, bobbyz2@uci.edu, 25441980
XiaoFan Lu, xiaofl14@uci.edu, 44016122
Junwei Huang, junweih4@uci.edu, 63321524
Shuang Wu, shuaw18@uci.edu, 75481078

## create index, 
1. go to index.py
2. you can use inverted_index class to create index
    the dir is the dir contain all the json file

    index = Invert_index(dir='DEV')
    index.update_index()
    index.create_partial_index()
    index.create_pageranks()
there is progress bar for executing the code 
required library for running index 
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.stem import PorterStemmer

to run the search, make sure you have the index ready, and 
change the directory, if  needed for 

id_to_doc_file = r'id_to_doc.json'
idf_file = r'idf.json'
index_distribution = {r'index1.json':'ejqxzv',
                            r'index2.json':'tuwyp',
                            r'index3.json':'rsno',
                            r'index4.json':'mlkih',
                            r'index5.json':'abcdfg',
                            r'index6.json':'0123456789'}
then write the basic_search.py, you will be prompt to type in your search query in terminal 
and type <stop> to stop the search

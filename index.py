import os
import json
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import math
from tqdm import tqdm
from nltk.stem import PorterStemmer
 



ps = PorterStemmer()
class Invert_index:
    """
    this is the class create 
    inverted index with posting of 
        tf, 
        idf,
        weight (headers, title, bold),
        pagerank,
    will finally return 6 index, where each index cover certain defined first term characters
    """
    def __init__(self,dir = r'.\IR24W-A3_6_m2\DEV') -> None:
        self.stop_word = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
        self.base_json_file = r'part'
        self.final_json_file = r'final_index.json'
        self.id_to_doc_file = r'id_to_doc.json'
        self.idf_file = r'idf.json'
        
        
        self.unique_word = set()
        self.num_index_doc = 0
        self.json_files = []
        self.processed_number = 0
        self.id_to_doc = dict()
        self.find_json_files(dir)
        
        self.urls=set()
        self.extracted_urls = []
        self.dir=dir
        self.pr_dict= {}
        
        # separate the index by letter of frequency
        self.index_distribution = {r'index1.json':'ejqxzv',
                            r'index2.json':'tuwyp',
                            r'index3.json':'rsno',
                            r'index4.json':'mlkih',
                            r'index5.json':'abcdfg',
                            r'index6.json':'0123456789'}
        self.json_file_len = len(self.json_files)
        
        print(f'total {len(self.json_files)} json file has found')

    def tokenizer(self,text_string):
        """
        accept alpha numeric 
        stem the tokens
        """
        words = re.findall(r'\b\w+\b', text_string.lower())
        stemed = [ps.stem(w) for w in words]
        # words = [w for w in words if w not in self.stop_word]
        return stemed
    
    def create_pageranks(self):
        """
        our page rank only consider incoming link,
        which is a simple version, there is not random walk implement
        return {url:# of link}
        """
        json_files=[]
        for root, _, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        for _ in tqdm(range(len(json_files))):
                if len(json_files) > 0:
                    page = json_files.pop()
                    with open(page, 'r') as f:
                        data = json.load(f)
                        content = data.get('content')
                        if content:
                            soup = BeautifulSoup(content, features='lxml')
                            try:
                                for link in soup.find_all("a"):
                                    link=link.get('href')
                                    if link in self.urls:
                                        self.pr_dict[link] = 1 + self.pr_dict.get(link, 0)
                            except:
                                print('No links found')

        sorted_dict = dict(sorted(self.pr_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        print('finish calculate page rank')
        with open('page_ranks.json', 'w') as pr_file:
            json.dump(sorted_dict, pr_file, indent=4)
        return sorted_dict
    
    def find_json_files(self,directory):
        """
        recursively finds all JSON files within a given directory.
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    self.json_files.append(os.path.join(root, file))
        return self.json_files
    
    def read_json_url_page(self, json_path):
        """
        read the json file to get
        words, headers, urls, term frequency
        and return the content in ret_dict for further index creation
        """
        ret_dict = {'url':'',
                    'tf':None,
                    'tokens':None,
                    'special_tokens':None}
        token_count = defaultdict(int)
        with open(json_path, 'r') as f:
            data = json.load(f)
            url = data.get('url')
            content = data.get('content')
            
        
            try: 
                special = {}
                soup = BeautifulSoup(content,'lxml')
                if soup.body is None:
                    print(f'nempty path: {json_path}')
                    return None
                heading_tags = ["h1", "h2", "h3",'title','b']
                for tags in soup.find_all(heading_tags):
                    special[tags.name] = tags.text.strip()
                    
                self.urls.add(url)
                for link in soup.find_all("a"):
                    self.extracted_urls.append(link.get('href'))
                raw_text = soup.body.get_text(' ', strip=True)
                tokens = self.tokenizer(raw_text)
                # print(tokens)
                for t in tokens:
                    token_count[t] +=1
                    self.unique_word.add(t)
                ret_dict["special_tokens"] = special
                ret_dict['url'] = url
                ret_dict['tf'] = token_count
                ret_dict['tokens'] = tokens
                return ret_dict
            except:
                print(f'not recongized by lxml: {json_path}')
                if url and content:
                    tokens = self.tokenizer(content)
                    for t in tokens:
                        token_count[t] +=1
                        self.unique_word.add(t)
                    ret_dict["special_tokens"] = None
                    ret_dict['url'] = url
                    ret_dict['tf'] = token_count
                    ret_dict['tokens'] = tokens
                    return ret_dict
                return None 
                
        
    def save_index_to_json(self, invert_index_dict):
        """ 
        save content of temperary invert index dictionary to json 
        """
        current_json_file = f'{self.base_json_file}{self.processed_number}.json'
        with open(current_json_file, 'w') as outfile:
            json.dump(invert_index_dict, outfile,indent=1)
        self.processed_number +=1 

    
    def update_index(self,threshold = 5600):
        """
        weight the title, headers and bold differently, all larger than normal text

        Args:
            threshold (int, optional): _description_. Defaults to 5600.
            so estimate 10 partial index file will created 
        """
        while self.json_files:
            temp_invert_index = dict()
            for _ in tqdm(range(threshold)):
                content = None
                if len(self.json_files) > 0:
                    page = self.json_files.pop()
                    content = self.read_json_url_page(page)

                    if content:
                        url, token_count_dict, tokens,specials = content['url'],content['tf'],content['tokens'],content['special_tokens']
                        # print(f'current url is {url}')
                        id = self.num_index_doc
                        self.num_index_doc +=1
                        self.id_to_doc[id] = url
                        
                        # need to keep the document id sorted 
                        for t in tokens:
                            if t not in temp_invert_index:
                                temp_invert_index[t] = {id:{'tf':token_count_dict[t],'w':1}}
                            else: 
                                temp_invert_index[t].update({id:{'tf':token_count_dict[t],'w':1}})
                            if specials:

                                for type, string in specials.items():
                                    if type == 'heads' and temp_invert_index[t][id]['w'] < 2 and t in string:
                                        temp_invert_index[t][id]['w'] = 2
                                    elif type == 'h1' and temp_invert_index[t][id]['w'] < 1.8 and t in string:
                                        temp_invert_index[t][id]['w'] = 1.8 
                                    elif type == 'h2' and temp_invert_index[t][id]['w'] < 1.6 and t in string:
                                        temp_invert_index[t][id]['w'] = 1.6
                                    elif type == 'h3' and temp_invert_index[t][id]['w'] < 1.4 and t in string:
                                        temp_invert_index[t][id]['w'] = 1.4
                                    elif type == 'b' and temp_invert_index[t][id]['w'] < 1.2 and t in string:
                                        temp_invert_index[t][id]['w'] = 1.2

                            
                else:
                    break
            self.save_index_to_json(temp_invert_index)
        with open(self.id_to_doc_file, "w") as f:
            index = json.dump(self.id_to_doc,f,indent=1)
        print(len(self.unique_word))
            # break
    
    def create_partial_index(self):
        """ 
        merge the content from raw partial index, to 6 index file that follow the 
        self.index_distribution
        """
        idf = dict()
        for path, letters in tqdm(self.index_distribution.items()):
            temp_dict = {}
            for i in range(self.processed_number):
                current_path = f'{self.base_json_file}{i}.json'
                if os.path.exists(current_path):
                    with open(current_path, 'r') as f:
                        data = json.load(f)
                        for term, url_info in data.items():
                            if term[0] in letters:
                                if term not in temp_dict:
                                    temp_dict[term] = url_info
                                else:
                                    temp_dict[term].update(url_info)
        
            for term in temp_dict.keys():
                idf[term] = math.log10(self.json_file_len/(len(temp_dict[term])+0.1))
            
            with open(path, 'w') as outfile:
                json.dump(temp_dict, outfile,indent=1)
        with open(self.idf_file, 'w') as outfile:
            json.dump(idf, outfile,indent=1)
                            

    def report(self):
        print(f'index {self.num_index_doc +1} documents')
        print(f'find total {len(self.unique_word)} unique words')
        with open(self.final_json_file, "r") as f:
            json_data = json.load(f)
        with open(self.id_to_doc_file, "w") as f:
            index = json.dump(self.id_to_doc,f,indent=1)

        json_string = json.dumps(json_data)
        json_bytes = json_string.encode("utf-8")

        # Calculate the size of the JSON file in KB.
        json_file_size_in_kb = len(json_bytes) / 1024
        print(f'the total size of index on my disk is {json_file_size_in_kb}kb')


# to create a index, need to input the directory with the json file
# then
# index.create_pageranks()
# index.update_index()
# index.create_partial_index()

index = Invert_index(dir='DEV')

index.update_index()
# index.processed_number = 10
index.create_partial_index()
index.create_pageranks()


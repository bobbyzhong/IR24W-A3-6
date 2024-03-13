import json
from nltk.stem import PorterStemmer
import math

ps = PorterStemmer()
id_to_url = dict()
idf = dict()
page_rank_dict = dict()
index = None
stop_word = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
id_to_doc_file = r'id_to_doc.json'
idf_file = r'idf.json'
page_rank_file = r'page_ranks.json'
index_distribution = {r'index1.json':'ejqxzv',
                            r'index2.json':'tuwyp',
                            r'index3.json':'rsno',
                            r'index4.json':'mlkih',
                            r'index5.json':'abcdfg',
                            r'index6.json':'0123456789'}

""" 
preload
"""
with open(id_to_doc_file, 'r') as f:
    data = json.load(f)
    for i in range(len(data)):
        id_to_url[str(i)] = data.get(str(i)) 

with open(idf_file, 'r') as f:
    data = json.load(f)
    for term, idf_value in data.items():
        idf[term] = idf_value

with open(page_rank_file, 'r') as f:
    data = json.load(f)
    for term, rank in data.items():
        page_rank_dict[term] = rank




def search_with_query(query:str, top_k=5):
    """
    1. first find the len of each query's documents
    2. merge the same doc from the smallest documents len 
    3. sorted the result docment list by the word frequency 
    4. return the top 5 doc in the sorted document list 
    """
    if len(query) == 0:
        return []
    temporary_invert_index = dict()
    query = query.lower().split()
    query = [q for q in query if q not in stop_word]
    
    if len(query) == 0:
        print(f'please consider use more informative queries, here is your current query{query}')
        return []
    query = [ps.stem(p) for p in query]
    
    
    # get a temporary index that only match to the query, and prepare to calculate score.
    for q in query: 
        for path, character_range in index_distribution.items():
            if q[0] in character_range:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if q in data:
                        documents =data.get(q)
                        temporary_invert_index[q] = documents
    
    # merge the index (term:len(docs)) from small to large
    # sorted_list = [{docs:{'tf':xx,'w':xx},}, {{docs:{'tf':xx,'w':xx}}]
    sort_list = [docs for t,docs in sorted(temporary_invert_index.items(),key=lambda x: len(x[1]))]
    if len(sort_list) == 0:
        return []
    
    ## initial same url for merge
    same_url_l = sort_list[0].keys()
    
    for i in range(len(sort_list)-1):
        same_url_l = intersect(list(same_url_l),list(sort_list[i+1].keys()))
    
    ranked_result = rank_urls(same_url_l,temporary_invert_index,idf_dict=idf,pagerank_dict=page_rank_dict)
    if len(ranked_result) > top_k:
        return ranked_result[:top_k]
    else:
        return ranked_result
    
def rank_urls(intersect_term_urls, 
                    related_urls_dic, 
                    idf_dict,
                    pagerank_dict = None,
                    intersected_weight = 1.5,
                    pagerank_weight = 0.3):
    """ 
    return urls ranked by 
    1. tf_idf score
    2. or tf_idf + pagerank
    """
    tf_idf_score = dict()
    intersect_term_urls = set(intersect_term_urls)
    for term, docs_dict in related_urls_dic.items():
        for url, info in docs_dict.items():
            tf = info['tf']
            weight = info['w']
            idf = idf_dict[term]
            
            weighted_score = (1+math.log10(tf))* weight *idf #* weight
            
            # add more weight, when all term in queries appears in a url
            if url in intersect_term_urls:
                weighted_score *= intersected_weight
            if url not in tf_idf_score:
                tf_idf_score[url] = weighted_score
            else:
                tf_idf_score[url] += weighted_score
    
    # after get top 30 tf-idf-score, add the page rank to rank calculation
    sort_result = [(url,s) for url,s in sorted(tf_idf_score.items(),key=lambda x: x[1], reverse= True)]
    
    if len(sort_result)>30:
        sort_result = sort_result[:30]
    
    if pagerank_dict:
        final_rank = dict()
        tf_idf_max, tf_idf_min = sort_result[0][1],sort_result[-1][1]
        page_rank_values = [pagerank_dict[id_to_url[url]] for url,_ in sort_result if id_to_url[url] in page_rank_dict]
        pagerank_min = min(page_rank_values)
        pagerank_max = max(page_rank_values)
        for url, tf_idf_score in sort_result:
            if id_to_url[url] in pagerank_dict:
                page_rank_value = pagerank_weight* norm(pagerank_dict[id_to_url[url]],pagerank_min,pagerank_max)
            else:
                page_rank_value = 0
            final_rank[url] = (1-pagerank_weight) * norm(tf_idf_score,tf_idf_min,tf_idf_max) +page_rank_value
        sort_result = [(url,s) for url,s in sorted(final_rank.items(),key=lambda x: x[1], reverse= True)]
    
    return [url for url,_ in sort_result]

# min max normalization
def norm(v,min,max):
    return (v-min)/(max - min)

def intersect(first_term, second_term):
    """ 
    find same documents between two different term
    """
    merged_list = []
    f, s = 0,0
    while f < len(first_term)-1 and s < len(second_term)-1:
        if first_term[f] == second_term[s]:
            merged_list.append(first_term[f])
            f, s = f+1,s+1
        if first_term[f] > second_term[s]:
            s = s+1
        else : 
            f = f+1
    return merged_list
    

if __name__ == '__main__':
    import time
    print('-'*60)
    print(' '*20,'search engine')
    print('-'*60)
    while True:
        
        query = input("type your query here: ")
        if query == '<stop>':
            print('search end')
            break
        
        start = time.time()
        result = search_with_query(query,top_k=10)
        if len(result) > 0:
            for id in result:
                print(f'{id_to_url[str(id)]}')
        print(f'search time is {time.time() - start}')
        print(f'current query : {query}')
        print('-'*60)
        print()
        print('-'*60)

from pandas import DataFrame

def build_df_from_query(q, er, n, out):
    """Turn an Event Registry API query to a pandas DataFrame
    
    Parameters
    ----------
    q : query to send to Event Registry API (eventregistry.QueryArticlesIter)
    er : session on Event Registry API (eventregistry.EventRegistry)
    n : maximum number of articles required in the iterator (int)
    out: information required on each article returned by Event Registry API (eventregistry.ReturnInfo)
    
    Returns
    -------
    pandas DataFrame containing articles details
    """
    
    id_list = []
    title_list = [] 
    body_list = []
    date_list = []
    source_list = []
    location_list = []
    
    for article in q.execQuery(er, 
                               maxItems=n, 
                               returnInfo=out,
                               sortBy='rel',
                               sortByAsc=False):
        
        id_list.append(article['uri'])
        title_list.append(article['title'])
        body_list.append(article['body'])
        date_list.append(article['date'])
        source_list.append(article['source']['title'] if article['source'] is not None else None)
        location_list.append(article['location']['label']['eng'] if article['location'] is not None else None)
    
    return DataFrame({'id': id_list,
                      'title': title_list,
                      'body': body_list,
                      'date': date_list,
                      'source': source_list,
                      'location': location_list})
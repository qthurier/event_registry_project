from pandas import DataFrame

import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

def build_df_from_query(q, er, n, out):
    """Turn an Event Registry API query into a pandas DataFrame and complete 
    with information retrieved with SpaCy NER model (trained on OntoNotes 5)
    
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
    person_list_of_list = []
    event_list_of_list = []
    
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
        
        doc = nlp(article['body'])
        person_set = set([e.text for e in doc.ents if e.label_ == 'PERSON'])
        event_set = set([e.text for e in doc.ents if e.label_ == 'EVENT'])
        person_list_of_list.append(list(person_set))
        event_list_of_list.append(list(event_set))
    
    return DataFrame({'id': id_list,
                      'title': title_list,
                      'body': body_list,
                      'date': date_list,
                      'source': source_list,
                      'location': location_list,
                      'person_list': person_list_of_list,
                      'event_list': event_list_of_list})
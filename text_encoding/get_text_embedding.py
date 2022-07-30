import numpy as np
import pandas as pd
import random

def get_text_embed(arr,dataset):
    """
        arr represents train_arr. We only consider the train_arr in each fold ignoring the test_arr relation.
        output disease text features and biomarker text features.
    """
    sent_embed = np.loadtxt('text_encoding/'+dataset+'_text_embed.txt')
    text_df = pd.read_csv('data/'+dataset+'/all_text.csv',header=None,names=['biomarker','disease','des'])
    labels = np.loadtxt('data/'+dataset+'/adj.txt')
    disease_df = pd.read_excel('data/'+dataset+'/diseases.xlsx',header=None,names=['id','disease'])
    if dataset == 'HMDD':
        biomarker_df = pd.read_excel('data/'+dataset+'/miRNAs.xlsx',header=None,names=['id','biomarker'])
    elif dataset == 'HMDAD':
        biomarker_df = pd.read_excel('data/'+dataset+'/microbes.xlsx',header=None,names=['id','microbe'])
    elif dataset == 'LncRNADisease':
        biomarker_df = pd.read_excel('data/' + dataset + '/lncRNAs.xlsx', header=None, names=['id', 'lncRNA'])
    disease_id_name = {}
    biomarker_id_name = {}
    for i in range(len(disease_df)):
        disease_id_name[disease_df.iloc[i,0]] = disease_df.iloc[i,1]
    for i in range(len(biomarker_df)):
        biomarker_id_name[biomarker_df.iloc[i,0]] = biomarker_df.iloc[i,1]

    entity_to_id = {}
    for i in range(len(disease_df)):
        id = disease_df.iloc[i,0]
        disease = disease_df.iloc[i,1]
        entity_to_id[disease] = id - 1

    for i in range(len(biomarker_df)):
        id = biomarker_df.iloc[i,0]
        biomarker = biomarker_df.iloc[i,1]
        entity_to_id[biomarker] = id - 1 + len(disease_df)

    ASS = {}
    ASS_embedding = {}
    #arr
    for i in range(len(arr)):
        disease = disease_id_name[int(labels[int(arr[i]), 0])]
        biomarker = biomarker_id_name[int(labels[int(arr[i]), 1])]
        ASS[disease+'#'+biomarker] = []

    for i in range(len(text_df)):
        disease = text_df.iloc[i,1]
        biomarker = text_df.iloc[i,0]
        relation = disease+'#'+biomarker
        if relation in ASS:
            ASS[relation].append(sent_embed[i])

    for key, values in ASS.items():
        """
            Random sample three sentence to represent ASS
        """
        if len(values) > 3:
            tmp_values = np.array(values)
            index = random.sample(range(0, tmp_values.shape[0]), 3)
            tmp_values = tmp_values[index]
            ASS_embedding[key] = np.sum(tmp_values, axis=0)
        elif len(values) > 1:
            ASS_embedding[key] = np.sum(np.array(values), axis=0)
        else:
            ASS_embedding[key] = values[0]

    entity_text_feature_dict = {}

    for key,values in ASS_embedding.items():
        disease,biomarker = key.split('#')
        if disease not in entity_text_feature_dict:
            entity_text_feature_dict[disease] = []
        if biomarker not in entity_text_feature_dict:
            entity_text_feature_dict[biomarker] = []
        entity_text_feature_dict[disease].append(values)
        entity_text_feature_dict[biomarker].append(values)

    entity_text_features = np.zeros((len(disease_df)+len(biomarker_df),768))
    for key,values in entity_text_feature_dict.items():
        if len(values) > 1:
            entity_text_features[entity_to_id[key]] = np.sum(np.array(values),axis=0)
        else:
            entity_text_features[entity_to_id[key]] = values[0]
    return entity_text_features[:len(disease_df)], entity_text_features[len(disease_df):]

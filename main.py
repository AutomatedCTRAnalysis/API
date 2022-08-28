# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from os.path import dirname, join, realpath
import numpy as np

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests,time, json
from bs4 import BeautifulSoup
from collections import defaultdict 

from collections import Counter

from fastapi import FastAPI, APIRouter
import pandas as pd
from pydantic import BaseModel
import uvicorn
from typing import List, Dict
import pickle

from fastapi.middleware.cors import CORSMiddleware

# Open Files: 
tactic_model=None
technique_model = None
tfidf_model = None
tfidf_technique = None
cv = None

def classifier(text):
    
    # Load labels:
    tactic_list = ['TA0006', 'TA0002', 'TA0040', 'TA0003', 'TA0004', 'TA0008', 'TA0005',
       'TA0010', 'TA0007', 'TA0009', 'TA0011', 'TA0001' ]
        
    technique_list = ['T1066',
    'T1047',
    'T1548',
    'T1156',
    'T1113',
    'T1067',
    'T1543', 
    'T1597',
    'T1037',
    'T1594',
    'T1595',
    'T1546', 
    'T1033',
    'T1556',
    'T1590',
    'T1534',
    'T1542',
    'T1571',
    'T1578',
    'T1003',
    'T1535',
    'T1539',
    'T1572',
    'T1592', 
    'T1593',
    'T1129',
    'T1559',
    'T1596',
    'T1599',
    'T1492',
    'T1550',
    'T1563',
    'T1570',
    'T1525',
    'T1529',
    'T1526', 
    'T1547', 
    'T1598',
    'T1566',
    'T1538', 
    'T1044',
    'T1590', 
    'T1553',
    'T1555', 
    'T1505',
    'T1171',
    'T1537',
    'T1014',
    'T1528',
    'T1530', 
    'T1557',
    'T1501',
    'T1011', 
    'T1554', 
    'T1531',
    'T1574',
    'T1123',
    'T1558',
    'T1614',
    'T1133',
    'T1608',
    'T1109',
    'T1569',
    'T1552',
    'T1099',
    'T1560',
    'T1562',
    'T1600',
    'T1567', 
    'T1647',
    'T1573',
    'T1069',
    'T1114',
    'T1163',
    'T1025',
    'T1601',
    'T1116',
    'T1093',
    'T1561', 
    'T1615',
    'T1178',
    'T1013',
    'T1565', 
    'T1606', 
    'T1568', 
    'T1489',
    'T1206',
    'T1602', 
    'T1564',
    'T1063',
    'T1080',
    'T1612', 
    'T1609', 
    'T1167',
    'T1611',
    'T1165',
    'T1137',
    'T1089',
    'T1622', 
    'T1619', 
    'T1487',
    'T1613', 
    'T1620',
    'T1621',
    'T1214',
    'T1119',
    'T1115',
    'T1103',
    'T1007',
    'T1040',
    'T1610', 
    'T1135',
    'T1120',
    'T1082',
    'T1071',
    'T1053',
    'T1162',
    'T1176',
    'T1106',
    'T1058',
    'T1202',
    'T1024',
    'T1091',
    'T1005',
    'T1140',
    'T1072',
    'T1195',
    'T1190',
    'T1219',
    'T1079',
    'T1036',
    'T1055',
    'T1205',
    'T1218',
    'T1038',
    'T1050',
    'T1010',
    'T1032',
    'T1062',
    'T1182',
    'T1029',
    'T1004',
    'T1009',
    'T1076',
    'T1131',
    'T1181',
    'T1483',
    'T1185',
    'T1021',
    'T1207',
    'T1107',
    'T1145',
    'T1112',
    'T1491',
    'T1155',
    'T1496',
    'T1217',
    'T1183',
    'T1085',
    'T1031',
    'T1092',
    'T1222',
    'T1179',
    'T1019',
    'T1042',
    'T1117',
    'T1054',
    'T1108',
    'T1193',
    'T1101',
    'T1177',
    'T1125',
    'T1144',
    'T1045',
    'T1016',
    'T1198',
    'T1087',
    'T1090',
    'T1059',
    'T1482',
    'T1175',
    'T1020',
    'T1070',
    'T1083',
    'T1138',
    'T1191',
    'T1188',
    'T1074',
    'T1049',
    'T1064',
    'T1051',
    'T1497',
    'T1102',
    'T1104',
    'T1480',
    'T1204',
    'T1196',
    'T1057',
    'T1141',
    'T1041',
    'T1060',
    'T1023',
    'T1026',
    'T1122',
    'T1015',
    'T1212',
    'T1210',
    'T1142',
    'T1199',
    'T1098',
    'T1170',
    'T1048',
    'T1097',
    'T1110',
    'T1001',
    'T1039',
    'T1078',
    'T1073',
    'T1068',
    'T1208',
    'T1027',
    'T1201',
    'T1187',
    'T1486',
    'T1488',
    'T1174',
    'T1002',
    'T1081',
    'T1128',
    'T1056',
    'T1203',
    'T1168',
    'T1495', 
    'T1100',
    'T1186',
    'T1518',
    'T1184',
    'T1095',
    'T1075',
    'T1012',
    'T1588',
    'T1030',
    'T1028',
    'T1034',
    'T1499',
    'T1065',
    'T1585', 
    'T1197',
    'T1088',
    'T1493',
    'T1132',
    'T1500',
    'T1583',
    'T1589', 
    'T1223',
    'T1213',
    'T1584', 
    'T1194',
    'T1200',
    'T1485',
    'T1130',
    'T1586', 
    'T1580', 
    'T1587', 
    'T1022',
    'T1189',
    'T1498',
    'T1158',
    'T1221',
    'T1134',
    'T1209',
    'T1111',
    'T1159',
    'T1136',
    'T1018',
    'T1046',
    'T1052',
    'T1105',
    'T1084',
    'T1160',
    'T1484',
    'T1220',
    'T1173',
    'T1008',
    'T1096',
    'T1124',
    'T1035',
    'T1006', 
    'T1086',
    'T1490',
    'T1216',
    'T1094',
    'T1043',
    'T1211',
    'T1127',
    'T1077']
    
    d_index_to_label_technique = dict(zip(technique_model.classes_, technique_list))

    # Perform TF-IDF:
    tfidf_input = tfidf_model.transform([text])
    tfidf_input = pd.DataFrame(tfidf_input.toarray(), columns = tfidf_model.get_feature_names())

    # Classify report for tactics: 
    result_tactic = tactic_model.predict_proba(tfidf_input)[0]
    result_tactic_dict = dict(zip(tactic_list, result_tactic))
    result_labels = [x for x, p in result_tactic_dict.items() if p > 0.5]

    # Classify report for techniques:
    result_technique = technique_model.predict_proba(tfidf_input)[0]
    result_technique_dict = dict(zip(technique_list, result_technique))
    result_labels_technique = [x for x, p in result_technique_dict.items() if p > 0.3]


    # Find match in report using description model:
    sents = nltk.sent_tokenize(text)
    tfidf_input_sents = tfidf_model.transform(sents)
    sent_preds = pd.DataFrame(tactic_model.predict_proba(tfidf_input_sents), columns=tactic_list)
    relevant_sents = Counter()
    
    for i, sent in enumerate(sents):
       for tactic in tactic_list:
            relevant_sents[sent] = max(sent_preds.loc[i,tactic], relevant_sents[sent])
                
    return result_labels, result_labels_technique, result_tactic_dict, result_technique_dict, [x for x, y in relevant_sents.most_common(5) if y > 0.3]

# Initialize API: 
app = FastAPI(title="2aCTI API",
    description="API for automated analysis of CTI Reports",
    version="0.1", openapi_url="/openapi.json")

api_router = APIRouter()

class InputReport(BaseModel):
    sentence: str

def mitre_scraping(overview_response):
    soup = BeautifulSoup(overview_response.content, "html5lib")
    mitre_desc = ""
    d_mitre = {}
    try:
        desc_tag = soup.find('div',{'class':'description-body'})
        for span_remover in desc_tag.findAll('span'):
            span_remover.decompose()
        pass
        mitre_desc = desc_tag.text.strip()
    except BaseException as E:
        print('Desc Error:{}'.formatE)

    pass
    try:
        technique_id = []
        main_table = soup.find('table',{'class': 'techniques-used'}).find("tbody").find_all("tr")
        for table in main_table:
            if table.find_all("td")[1].text.strip().__contains__('T'):
                technique_id.append(table.find_all("td")[1].text.strip())
        pass
    except BaseException as E:
        print(E)
    pass
    d_mitre['description'] = mitre_desc
    d_mitre['techniques'] = technique_id
    d_mitre['status'] = 'found'
    return d_mitre

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
    
# /searchattack for search bar
@api_router.post("/searchattack")
def search_attack(input_report: InputReport):
    search_keyword = input_report.sentence
    response = requests.get("https://attack.mitre.org/software/")
    d_mitre = {}
    if response.status_code == 200:
        overview_link = ""
        soup = BeautifulSoup(response.content, "html5lib")
        for view in soup.find_all('div',{'class':'sidenav-head'})[0:]:
            if view.text.lower().__contains__(search_keyword.lower()):
                overview_link = 'https://attack.mitre.org'+view.find('a').get('href')
                break

        if len(overview_link) > 0:
            overview_request = requests.get(overview_link)
            return mitre_scraping(overview_request)
        else:
           return {'status': 'not_found'}

# /classification allows to group different models together 
@api_router.post("/classification")
def read_classification(input_report: InputReport):
    if tactic_model is None:
        load()
    tactics, techniques, result_tactic_dict, result_technique_dict, relevant_sents = classifier(input_report.sentence)
    return {'tactics': tactics, 'techniques' : techniques, 'relevant_sents': relevant_sents, 'relevant_tactic_dict': result_tactic_dict, 'relevant_technique_dict': result_technique_dict} 

def load():
    global tactic_model
    global technique_model
    global tfidf_model
    global tfidf_technique
    with open('tactic_MLP_TFIDF_model.pickle', 'rb') as handle:
        tactic_model = pickle.load(handle)
    with open('technique_MLP_TFIDF_50.pickle', 'rb') as handle:
        technique_model = pickle.load(handle)
    with open('tfidf_no_lemma.pickle', 'rb') as handle:
        tfidf_model = pickle.load(handle)

app.include_router(api_router)




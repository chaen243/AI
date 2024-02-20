import json

with open('C:\prodownload\\token_dic.json', 'r') as f:
    label_dict = json.load(f)

# 불러온 딕셔너리 확인
print(label_dict)
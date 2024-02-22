import pandas as pd

# Excel 파일 읽기
excel_file = 'C:\수어 데이터셋\KETI-2017-SL-Annotation-v2_1.xlsx'
df = pd.read_excel(excel_file)

# CSV 파일로 저장
csv_file = 'C:\수어 데이터셋\KETI-2017-SL-Annotation-v2_1.csv'
df.to_csv(csv_file, index=False)
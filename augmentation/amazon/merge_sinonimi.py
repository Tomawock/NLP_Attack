import pandas as pd

csv_merge = pd.read_csv('amazon_sinonimi_part_0.csv')
for i in range(1, 66):
    try:
        csv_merge = csv_merge.append(pd.read_csv(f'amazon_sinonimi_part_{i}.csv'), ignore_index=True)
    except:
        pass
    
csv_merge.to_csv('amazon_sinonimi.csv', index=False)

import concurrent.futures
from datetime import datetime
import pandas as pd
from textattack.augmentation import WordNetAugmenter


def chunk_dataset(dataset, max_iteration=66):
    dim = len(dataset)//max_iteration
    for i in range(max_iteration):
        yield dataset[i*dim:(i+1)*dim]

    if max_iteration * dim != len(dataset):
        yield dataset[max_iteration*dim:]


def embed_chunk(chunk, chunk_number):
    my_items = []
    augmenter = WordNetAugmenter(pct_words_to_swap=0.5,
                                 transformations_per_example=1)

    for i, row in enumerate(chunk.itertuples()):
        print(f'Chunk: {chunk_number}\tSinonimo: {i}\t{datetime.now().strftime("%H:%M:%S")}')
        new_text = augmenter.augment(row.Text)[0]
        my_items.append({'Score': row.Score, 'Summary': row.Summary,
                         'Text': new_text, 'review_type': row.review_type})

    pd.DataFrame(my_items).to_csv(f'amazon_sinonimi_part_{chunk_number}.csv', index=False)


df = pd.read_csv('reducedReviews.csv')

with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, subset in enumerate(chunk_dataset(df)):
        executor.submit(embed_chunk, subset, i)

from datasets.pointar.preprocess.generate import generate

generate(
    "test", # dataset name ["train", "test"],
    index='all', # data item, 'all' for all items, 0 for data item 0
)

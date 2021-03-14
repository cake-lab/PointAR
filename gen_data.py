import multiprocessing
from datasets.pointar.preprocess.pack import pack
from datasets.pointar.preprocess.generate import generate

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    generate(
        "test",  # dataset name ["train", "test"],
        index='all',  # data item, 'all' for all items, 0 for data item 0
    )

    pack("test", index="all")

    generate(
        "train",  # dataset name ["train", "test"],
        index='all',  # data item, 'all' for all items, 0 for data item 0
    )

    pack("train", index="all")

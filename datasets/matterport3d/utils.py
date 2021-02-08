def load_matterport3d_dataset_list(dataset):
    """
    Reads the dataset list txt file from neural illumination dataset. Empty
    lines are removed.

    Parameters
    ----------
    dataset : str
        Could be "train" or "test", indicating which dataset to use.

    Returns
    -------
    dataset_list : list
        A list contains all lines in a dataset list file.
    """

    with open(f'/mnt/IRONWOLF2/yiqinzhao/Xihe/info/{dataset}list.txt', 'r') as _f:
        dataset_list = _f.readlines()
        dataset_list = [v for v in dataset_list if v]

    return dataset_list

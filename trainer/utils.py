import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def train_valid_test_split(
        train_loader_class, train_loader_init_params,
        test_loader_class, test_loader_init_params,
        normalize=False, num_workers=0, batch_size=32):

    train_loader = train_loader_class(**train_loader_init_params)
    valid_loader = train_loader_class(**train_loader_init_params)
    test_loader = test_loader_class(**test_loader_init_params)

    l = len(train_loader)
    indices = np.arange(l, dtype=np.int)
    valid_idx = np.sort(np.random.choice(l, 2500))

    valid_mask = np.zeros((l), dtype=np.bool)
    valid_mask[valid_idx] = True

    train_loader.arr_indices = indices[~valid_mask]
    valid_loader.arr_indices = indices[valid_mask]

    scaler = None

    if normalize:
        scaler = MinMaxScaler()

        t = train_loader.arr_target.reshape((-1, 3 * 9))
        scaler.fit(t[~valid_mask]) # fit only training part

        t = scaler.transform(t)
        train_loader.arr_target = np.array(t, copy=True)
        valid_loader.arr_target = np.array(t, copy=True)

        t = test_loader.arr_target.reshape((-1, 3 * 9))
        test_loader.arr_target = scaler.transform(t)

    train_loader = DataLoader(
        train_loader,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    valid_loader = DataLoader(
        valid_loader,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    test_loader = DataLoader(
        test_loader,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    return (train_loader, valid_loader, test_loader), scaler

from typing import Union, List

from torch.utils.data import DataLoader

from classification_with_embeddings.embedding.cnn.dataset import FastTextFormatDataset, FastTextFormatCompositeDataset


def get_dataloader(train_data_path: Union[str, List[str]], batch_size: int):
    if isinstance(train_data_path, str):
        dataset = FastTextFormatDataset(train_data_path)
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=FastTextFormatDataset.collate)
    elif isinstance(train_data_path, list):
        dataset = FastTextFormatCompositeDataset(train_data_path)
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=FastTextFormatCompositeDataset.collate)
    else:
        raise ValueError('Specified data path(s) should be a str or a list.')

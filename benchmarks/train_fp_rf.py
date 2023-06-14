import numpy as np

from dataclasses import asdict
from sklearn.ensemble import RandomForestRegressor

from config import Config
from data.ee import EE
from data.bde import BDE
from data.drugs import Drugs
from data.kraken import Kraken
from happy_config import ConfigLoader

from models.models_1d.utils import construct_fingerprint, construct_smiles, concatenate_smiles


if __name__ == '__main__':
    loader = ConfigLoader(model=Config, config='params/params_fp.json')
    config = loader()

    variable_name = None
    unique_variables = 1
    if config.dataset == 'Drugs':
        dataset = Drugs('datasets/Drugs', max_num_conformers=config.max_num_conformers).shuffle()
    elif config.dataset == 'Kraken':
        dataset = Kraken('datasets/Kraken', max_num_conformers=config.max_num_conformers).shuffle()
    elif config.dataset == 'BDE':
        dataset = BDE('datasets/BDE').shuffle()
        variable_name = 'is_ligand'
        unique_variables = 2
    elif config.dataset == 'EE':
        dataset = EE('datasets/EE', max_num_conformers=config.max_num_conformers).shuffle()
        variable_name = 'config_id'
        unique_variables = 2

    target_id = dataset.descriptors.index(config.target)
    labels = dataset.y[:, target_id]
    mean = labels.mean(dim=0).item()
    std = labels.std(dim=0).item()
    labels = (labels - mean) / std

    if variable_name is not None:
        smiles = concatenate_smiles(dataset, variable_name)
    else:
        smiles = construct_smiles(dataset)

    target_id = dataset.descriptors.index(config.target)
    y = dataset.y[:, target_id]
    mean = y.mean(dim=0).item()
    std = y.std(dim=0).item()
    y = (y - mean) / std
    y = y.numpy()

    print('Constructing fingerprint...')
    fingerprint = construct_fingerprint(smiles)

    train_len = int(config.train_ratio * y.shape[0])
    valid_len = int(config.valid_ratio * y.shape[0])
    test_len = y.shape[0] - train_len - valid_len

    train_x = fingerprint[:train_len]
    valid_x = fingerprint[train_len:(train_len + valid_len)]
    test_x = fingerprint[(train_len + valid_len):]

    train_y = y[:train_len]
    valid_y = y[train_len:(train_len + valid_len)]
    test_y = y[(train_len + valid_len):]

    print('Training model...')
    rf = RandomForestRegressor(random_state=config.seed, **asdict(config.modelfprf))
    rf.fit(train_x, train_y)

    test_y_hat = rf.predict(test_x)
    test_mae = np.mean(np.abs(test_y_hat * std - test_y * std))

    valid_y_hat = rf.predict(valid_x)
    valid_mae = np.mean(np.abs(valid_y_hat * std - valid_y * std))

    print(f'Validation error: {valid_mae:.4f}')
    print(f'Test error: {test_mae:.4f}')

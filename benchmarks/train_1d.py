import os
import torch
import torch.nn.functional as F

from dataclasses import asdict
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from config import Config
from data.ee import EE
from data.bde import BDE
from data.drugs import Drugs
from data.kraken import Kraken
from happy_config import ConfigLoader
from utils.early_stopping import EarlyStopping, generate_checkpoint_filename

from models.model_1d import LSTM, Transformer
from models.models_1d.utils import construct_fingerprint, construct_smiles, concatenate_smiles


class Molecules(Dataset):
    def __init__(self, smiles_ids, attention_masks, labels, fingerprint=None, input_type='smiles'):
        self.smiles_ids = smiles_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.fingerprint = fingerprint
        self.input_type = input_type

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.input_type == 'SMILES':
            smiles = self.smiles_ids[index]
            attention_mask = self.attention_masks[index]
            y = self.labels[index]
            return smiles, attention_mask, y.clone()
        else:
            fingerprint = self.fingerprint[index]
            y = self.labels[index]
            return torch.tensor(fingerprint, dtype=torch.long), y.clone()


def train(loader):
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        if config.model1d.input_type == 'SMILES':
            input_ids, attention_mask, y = batch
            if isinstance(model, Transformer):
                out = model(input_ids, attention_mask)
            else:
                out = model(input_ids)
        else:
            fingerprints, y = batch
            out = model(fingerprints)
        loss = F.mse_loss(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * y.shape[0]
        num_samples += y.shape[0]
    return epoch_loss / num_samples


@torch.no_grad()
def eval(loader):
    model.eval()
    error = 0.0
    num_samples = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        if config.model1d.input_type == 'SMILES':
            input_ids, attention_mask, y = batch
            if isinstance(model, Transformer):
                out = model(input_ids, attention_mask)
            else:
                out = model(input_ids)
        else:
            fingerprints, y = batch
            out = model(fingerprints)
        out = out.squeeze()
        error += (out * std - y * std).abs().sum().item()
        num_samples += len(y)
    return error / num_samples


if __name__ == '__main__':
    writer = SummaryWriter()
    loader = ConfigLoader(model=Config, config='params/params_1d.json')
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
    fingerprint = construct_fingerprint(smiles) if config.model1d.input_type == 'Fingerprint' else None

    tokenizer = RobertaTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
    dicts = tokenizer(smiles, return_tensors='pt', padding='longest')
    smiles_ids, attention_masks = dicts['input_ids'], dicts['attention_mask']
    vocab_size = tokenizer.vocab_size if config.model1d.input_type == 'SMILES' else fingerprint.shape[1]

    device = torch.device(config.device)
    dataset = Molecules(smiles_ids, attention_masks, labels, fingerprint, input_type=config.model1d.input_type)

    if config.model1d.model == 'LSTM':
        model = LSTM(
            vocab_size, config.hidden_dim, config.hidden_dim, 1,
            config.model1d.num_layers, config.dropout, padding_idx=tokenizer.pad_token_id)
    elif config.model1d.model == 'Transformer':
        model = Transformer(
            vocab_size, config.model1d.embedding_dim, smiles_ids.shape[1],
            config.model1d.num_heads, config.hidden_dim, 1,
            config.model1d.num_layers, config.dropout, padding_idx=tokenizer.pad_token_id)
    model = model.to(device)

    train_ratio = config.train_ratio
    valid_ratio = config.valid_ratio
    test_ratio = 1 - train_ratio - valid_ratio

    train_len = int(train_ratio * len(dataset))
    valid_len = int(valid_ratio * len(dataset))
    test_len = len(dataset) - train_len - valid_len

    train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths=[train_len, valid_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    checkpoint_path = generate_checkpoint_filename()
    early_stopping = EarlyStopping(patience=config.patience, path=checkpoint_path)
    print(f'Checkpoint path: {checkpoint_path}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, verbose=True, **asdict(config.reduce_lr_on_plateau))
    elif config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, verbose=True, **asdict(config.cosine_annealing_lr))
    elif config.scheduler == 'LinearWarmupCosineAnnealingLR':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, **asdict(config.linear_warmup_cosine_annealing_lr))
    else:
        scheduler = None

    best_val_error = None
    for epoch in range(config.num_epochs):
        loss = train(train_loader)
        if scheduler is not None:
            scheduler.step(loss)
        valid_error = eval(valid_loader)

        early_stopping(valid_error, model)
        if early_stopping.counter == 0:
            test_error = eval(test_loader)
        if early_stopping.early_stop:
            print('Early stopping...')
            break

        writer.add_scalar(f'Loss_{config.model1d.model}/{config.model1d.input_type}'
                          f'/{config.dataset}/{config.target}/train', loss, epoch)
        writer.add_scalar(f'Loss_{config.model1d.model}/{config.model1d.input_type}'
                          f'/{config.dataset}/{config.target}/valid', valid_error, epoch)
        writer.add_scalar(f'Loss_{config.model1d.model}/{config.model1d.input_type}'
                          f'/{config.dataset}/{config.target}/test', test_error, epoch)
        print(f'Progress: {epoch}/{config.num_epochs}/{loss:.5f}/{valid_error:.5f}/{test_error:.5f}')

    model.load_state_dict(torch.load(checkpoint_path))
    test_error = eval(test_loader)
    print(f'Best validation error: {-early_stopping.best_score:.7f}')
    print(f'Test error: {test_error:.7f}')

    os.remove(checkpoint_path)
    writer.close()

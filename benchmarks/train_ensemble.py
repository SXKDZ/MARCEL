import os
import torch
import torch.nn.functional as F

from dataclasses import asdict
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from config import Config
from data.ee import EE
from data.bde import BDE
from data.drugs import Drugs
from data.kraken import Kraken
from happy_config import ConfigLoader
from loaders.utils import reorder_molecule_idx
from loaders.samplers import EnsembleSampler, EnsembleMultiBatchSampler
from loaders.multibatch import MultiBatchLoader
from utils.early_stopping import generate_checkpoint_filename, EarlyStopping

from models.models_3d.painn import PaiNN
from models.models_3d.schnet import SchNet
from models.models_3d.gemnet import GemNetT
from models.models_3d.dimenet import DimeNetPlusPlus
from models.models_3d.clofnet import ClofNet
from models.models_3d.leftnet import LEFTNet
from models.model_4d import Model4D, SumPooling, MeanPooling, TransformerPooling, DeepSets, SelfAttentionPooling


def train(dataset, loader):
    model.train()
    loss_all = 0

    for data in loader:
        optimizer.zero_grad()
        if type(data) is not list:
            data = [data]
        molecule_indices = [
            reorder_molecule_idx(data[i].molecule_idx).to(device) for i in range(len(data))]
        out = model(data, molecule_indices)

        unique_raw_molecule_idx = torch.unique_consecutive(data[0].molecule_idx)
        y = dataset.y[unique_raw_molecule_idx]
        loss = F.mse_loss(out, y)
        loss.backward()
        loss_all += loss.item() * len(unique_raw_molecule_idx)
        optimizer.step()
    return loss_all / dataset.num_molecules


@torch.no_grad()
def eval(dataset, loader):
    model.eval()
    error = 0

    for data in loader:
        if type(data) is not list:
            data = [data]
        molecule_indices = [
            reorder_molecule_idx(data[i].molecule_idx).to(device) for i in range(len(data))]
        out = model(data, molecule_indices)

        unique_raw_molecule_idx = torch.unique_consecutive(data[0].molecule_idx)
        y = dataset.y[unique_raw_molecule_idx]
        error += (out * std - y * std).abs().sum().item()  # MAE
    return error / dataset.num_molecules


if __name__ == '__main__':
    writer = SummaryWriter()
    loader = ConfigLoader(model=Config, config='params/params_4d.json')
    config = loader()

    variable_name = None
    unique_variables = 1
    if config.dataset == 'Drugs':
        dataset = Drugs('datasets/Drugs', max_num_conformers=config.max_num_conformers).shuffle()
    elif config.dataset == 'Kraken':
        dataset = Kraken('datasets/Kraken', max_num_conformers=config.max_num_conformers).shuffle()
    elif config.dataset == 'BDE':
        dataset = BDE('datasets/BDE', max_num_conformers=config.max_num_conformers).shuffle()
        variable_name = 'is_ligand'
        unique_variables = 2
    elif config.dataset == 'EE':
        dataset = EE('datasets/EE', max_num_conformers=config.max_num_conformers).shuffle()
        variable_name = 'config_id'
        unique_variables = 2
    device = torch.device(config.device)

    target_id = dataset.descriptors.index(config.target)
    dataset.y = dataset.y[:, target_id]
    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = ((dataset.y - mean) / std).to(device)
    mean = mean.to(device)
    std = std.to(device)

    split = dataset.get_idx_split(
        train_ratio=config.train_ratio, valid_ratio=config.valid_ratio,
        max_num_molecules=config.max_num_molecules, seed=config.seed)
    train_dataset = dataset[split['train']]
    valid_dataset = dataset[split['valid']]
    test_dataset = dataset[split['test']]

    if variable_name is None:
        train_loader = DataLoader(train_dataset, batch_sampler=EnsembleSampler(
            train_dataset, batch_size=config.batch_size, strategy='all', shuffle=True))
        valid_loader = DataLoader(valid_dataset, batch_sampler=EnsembleSampler(
            valid_dataset, batch_size=config.batch_size, strategy='all', shuffle=False))
        test_loader = DataLoader(test_dataset, batch_sampler=EnsembleSampler(
            test_dataset, batch_size=config.batch_size, strategy='all', shuffle=False))
    else:
        train_loader = MultiBatchLoader(train_dataset, batch_sampler=EnsembleMultiBatchSampler(
            train_dataset, batch_size=config.batch_size, strategy='all', shuffle=True, variable_name=variable_name))
        valid_loader = MultiBatchLoader(valid_dataset, batch_sampler=EnsembleMultiBatchSampler(
            valid_dataset, batch_size=config.batch_size, strategy='all', shuffle=False, variable_name=variable_name))
        test_loader = MultiBatchLoader(test_dataset, batch_sampler=EnsembleMultiBatchSampler(
            test_dataset, batch_size=config.batch_size, strategy='all', shuffle=False, variable_name=variable_name))

    max_atomic_num = dataset.data.x[:, 0].max().item() + 1
    if config.model4d.graph_encoder == 'SchNet':
        graph_model_factory = lambda: SchNet(max_atomic_num=max_atomic_num, **asdict(config.model4d.schnet))
    elif config.model4d.graph_encoder == 'DimeNet++':
        graph_model_factory = lambda: DimeNetPlusPlus(
            max_atomic_num=max_atomic_num, **asdict(config.model4d.dimenetplusplus))
    elif config.model4d.graph_encoder == 'GemNet':
        graph_model_factory = lambda: GemNetT(max_atomic_num=max_atomic_num, **asdict(config.model4d.gemnet))
    elif config.model4d.graph_encoder == 'PaiNN':
        graph_model_factory = lambda: PaiNN(max_atomic_num=max_atomic_num, **asdict(config.model4d.painn))
    elif config.model4d.graph_encoder == 'ClofNet':
        graph_model_factory = lambda: ClofNet(max_atomic_num=max_atomic_num, **asdict(config.model4d.clofnet))
    elif config.model4d.graph_encoder == 'LEFTNet':
        graph_model_factory = lambda: LEFTNet(max_atomic_num=max_atomic_num, **asdict(config.model4d.clofnet))

    if config.model4d.set_encoder == 'Sum':
        set_model_factory = lambda: SumPooling()
    elif config.model4d.set_encoder == 'Mean':
        set_model_factory = lambda: MeanPooling()
    elif config.model4d.set_encoder == 'DeepSets':
        set_model_factory = lambda: DeepSets(hidden_dim=config.hidden_dim)
    elif config.model4d.set_encoder == 'Attention':
        set_model_factory = lambda: SelfAttentionPooling(hidden_dim=config.hidden_dim)
    elif config.model4d.set_encoder == 'Transformer':
        set_model_factory = lambda: TransformerPooling(
            hidden_dim=config.hidden_dim, **asdict(config.model4d.transformer))

    model = Model4D(
        hidden_dim=config.hidden_dim, out_dim=1,
        graph_model_factory=graph_model_factory, set_model_factory=set_model_factory,
        unique_variables=unique_variables, device=device).to(device)

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

    checkpoint_path = generate_checkpoint_filename()
    early_stopping = EarlyStopping(patience=config.patience, path=checkpoint_path)
    print(f'Checkpoint path: {checkpoint_path}')

    # with tqdm(total=config.num_epochs) as pbar:
    for epoch in range(config.num_epochs):
        loss = train(train_dataset, train_loader)
        if scheduler is not None:
            scheduler.step(loss)
        valid_error = eval(valid_dataset, valid_loader)

        early_stopping(valid_error, model)
        if early_stopping.counter == 0:
            test_error = eval(test_dataset, test_loader)
        if early_stopping.early_stop:
            print('Early stopping...')
            break

        writer.add_scalar(
            f'Loss_{config.model4d.graph_encoder}_{config.model4d.set_encoder}/{config.dataset}/{config.target}/train',
            loss, epoch)
        writer.add_scalar(
            f'Loss_{config.model4d.graph_encoder}_{config.model4d.set_encoder}/{config.dataset}/{config.target}/valid',
            valid_error, epoch)
        writer.add_scalar(
            f'Loss_{config.model4d.graph_encoder}_{config.model4d.set_encoder}/{config.dataset}/{config.target}/test',
            test_error, epoch)
        print(f'Progress: {epoch}/{config.num_epochs}/{loss:.5f}/{valid_error:.5f}/{test_error:.5f}')

        # pbar.update()
        # pbar.set_postfix({'loss': loss, 'valid_error': f'{valid_error:.7f}', 'test_error': f'{test_error:.7f}'})

    model.load_state_dict(torch.load(checkpoint_path))
    test_error = eval(test_dataset, test_loader)
    print(f'Best validation error: {-early_stopping.best_score:.7f}')
    print(f'Test error: {test_error:.7f}')

    os.remove(checkpoint_path)
    writer.close()

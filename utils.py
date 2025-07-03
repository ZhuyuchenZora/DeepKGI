import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import random
import os
import pickle
import math
from models.DeepKGI import DeepKGI
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from preprocess_gene import get_STRING_graph, get_predefine_cluster
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

dict_dir = './data/similarity_augment/dict/'
with open(dict_dir + "cell_id2idx_dict", 'rb') as f:
    cell_id2idx_dict = pickle.load(f)
with open(dict_dir + "drug_name2idx_dict", 'rb') as f:
    drug_name2idx_dict = pickle.load(f)
with open(dict_dir + "cell_idx2id_dict", 'rb') as f:
    cell_idx2id_dict = pickle.load(f)
with open(dict_dir + "drug_idx2name_dict", 'rb') as f:
    drug_idx2name_dict = pickle.load(f)


def train(model, loader, criterion, opt, device):
    model.train()
    total_loss = 0
    batch_count = 0

    for idx, data in enumerate(tqdm(loader, desc='Iteration')):
        drug, cell, label = data
        if isinstance(cell, list):
            drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
        else:
            drug, cell, label = drug.to(device), cell.to(device), label.to(device)
        output = model(drug, cell)

        loss = criterion(output, label.view(-1, 1).float())
        total_loss += loss.item()
        batch_count += 1

        opt.zero_grad()
        loss.backward()
        opt.step()

    avg_loss = total_loss / batch_count
    print(f'Train Loss: {avg_loss:.4f}')
    return avg_loss


def validate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            drug, cell, label = data
            if isinstance(cell, list):
                drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
            else:
                drug, cell, label = drug.to(device), cell.to(device), label.to(device)
            output = model(drug, cell)
            total_loss += F.mse_loss(output, label.view(-1, 1).float(), reduction='sum')
            y_true.append(label.view(-1, 1))
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    rmse = torch.sqrt(total_loss / len(loader.dataset))
    MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    pearson_corr = pearsonr(y_true_np, y_pred_np)[0]
    spearman_corr = spearmanr(y_true_np, y_pred_np)[0]
    return rmse, MAE, r2, pearson_corr, spearman_corr


def gradient(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args):
    cell_dict[cell_name].edge_index = torch.tensor(edge_index, dtype=torch.long)
    drug = Batch.from_data_list([drug_dict[drug_name]]).to(args.device)
    cell = Batch.from_data_list([cell_dict[cell_name]]).to(args.device)

    model.eval()
    drug_representation = model.GNN_drug(drug)
    drug_representation = model.drug_emb(drug_representation)

    cell_node, cell_representation = model.GNN_cell.grad_cam(cell)
    cell_representation = model.cell_emb(cell_representation)

    # combine drug feature and cell line feature
    x = torch.cat([drug_representation, cell_representation], -1)
    ic50 = model.regression(x)
    model.zero_grad()
    ic50.backward()
    cell_node_importance = torch.abs((cell_node * torch.mean(cell_node.grad, dim=0)).sum(dim=1))
    cell_node_importance = cell_node_importance / cell_node_importance.sum()
    _, indices = torch.sort(cell_node_importance, descending=True)
    return ic50, indices.cpu()


def inference(model, drug_dict, cell_dict, edge_index, save_name, args):
    model.eval()
    IC = pd.read_excel("./data/IC50_GDSC/GDSC2_fitted_dose_response_27Oct23.xlsx")

    # 只保留训练/验证/测试集，移除未知数据集
    if args.setup == 'known':
        train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42, stratify=IC['Cell line name'])
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42,
                                             stratify=val_test_set['Cell line name'])

    # 只处理训练/验证/测试集
    dataset = {'train': train_set, 'val': val_set, 'test': test_set}
    writer = pd.ExcelWriter(save_name)

    for dataset_name, data in dataset.items():
        data.reset_index(drop=True, inplace=True)
        IC50_pred = []
        with torch.no_grad():
            drug_name, cell_ID, cell_line_name = data['Drug name'], data["DepMap_ID"], data["stripped_cell_line_name"]
            for cell in cell_ID:
                if cell in cell_dict:
                    cell_dict[cell].edge_index = torch.tensor(edge_index, dtype=torch.long)

            drug_list = [drug_dict[name] for name in drug_name if name in drug_dict]
            cell_list = [cell_dict[name] for name in cell_ID if name in cell_dict]

            batch_size = 2048
            batch_num = math.ceil(len(drug_list) / batch_size)

            for index in range(batch_num):
                start_idx = index * batch_size
                end_idx = (index + 1) * batch_size

                drug_batch = Batch.from_data_list(drug_list[start_idx:end_idx]).to(args.device)
                cell_batch = Batch.from_data_list(cell_list[start_idx:end_idx]).to(args.device)

                y_pred = model(drug_batch, cell_batch)
                IC50_pred.append(y_pred)

            if IC50_pred:
                IC50_pred = torch.cat(IC50_pred, dim=0)
            else:
                IC50_pred = torch.tensor([])

        table = pd.concat([drug_name, cell_ID, cell_line_name], axis=1)
        table["IC50"] = data["IC50"]
        table["IC50_Pred"] = IC50_pred.cpu().numpy()
        table["Abs_error"] = np.abs(IC50_pred.cpu().numpy() - table["IC50"].values.reshape(-1, 1))
        table.to_excel(writer, sheet_name=dataset_name, index=False)
        torch.cuda.empty_cache()

    writer.close()


class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, edge_index):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        cell_name = self.Cell_line_name[index]
        if cell_name in self.cell:
            self.cell[cell_name].edge_index = self.edge_index
        return (self.drug[self.drug_name[index]], self.cell[cell_name], self.value[index])


class MyDataset_CDR(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


class MyDataset_name(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['Cell line name']
        self.value = IC['IC50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


def _collate(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = Batch.from_data_list(cells)
    return batched_drug, batched_cell, torch.tensor(labels)


def _collate_drp(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(drugs)
    cells = [torch.tensor(cell) for cell in cells]
    return batched_graph, torch.stack(cells, 0), torch.tensor(labels)


def _collate_CDR(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(drugs)
    exp = [torch.tensor(cell[0]) for cell in cells]
    cn = [torch.tensor(cell[1]) for cell in cells]
    mu = [torch.tensor(cell[2]) for cell in cells]
    return batched_graph, [torch.stack(exp, 0), torch.stack(cn, 0), torch.stack(mu, 0)], torch.tensor(labels)


def load_data(IC, drug_dict, cell_dict, edge_index, args):
    # 只保留已知设置
    if args.setup == 'known':
        train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42, stratify=IC['Cell line name'])
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42,
                                             stratify=val_test_set['Cell line name'])
    else:
        raise ValueError("仅支持 'known' 设置")

    if args.model == 'TCNN':
        Dataset = MyDataset_name
        collate_fn = None
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif args.model == 'GraphDRP':
        Dataset = MyDataset_name
        collate_fn = _collate_drp
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif args.model == 'DeepCDR':
        Dataset = MyDataset_CDR
        collate_fn = _collate_CDR
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    else:
        Dataset = MyDataset
        collate_fn = _collate
        train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4)

    return train_loader, val_loader, test_loader


class EarlyStopping():
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or 'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import networkx as nx
import matplotlib.pyplot as plot
import seaborn as sns

def get_genes_graph(genes_path, save_path, method='pearson', thresh=0.95, p_value=False):

    genes_exp_df = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
    genes_cn_df = pd.read_csv(os.path.join(genes_path, 'cn.csv'), index_col=0)
    genes_mu_df = pd.read_csv(os.path.join(genes_path, 'mu.csv'), index_col=0)
    genes_trans_df = pd.read_csv(os.path.join(genes_path, 'Cell_line_RMA_proc_basalExp.txt'), sep='\t', index_col=0)

    genes_exp_corr = genes_exp_df.corr(method=method)
    n = genes_exp_df.shape[0]
    if p_value:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)
    adj = np.where(np.abs(genes_exp_corr) > thresh, 1, 0)
    adj = adj - np.eye(adj.shape[0], dtype=int)
    edge_index = np.nonzero(adj)

    # CNV
    genes_cn_corr = genes_cn_df.corr(method=method)
    n = genes_cn_df.shape[0]
    if p_value:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)
    adj_cn = np.where(np.abs(genes_cn_corr) > thresh, 1, 0)
    adj_cn = adj_cn - np.eye(adj_cn.shape[0], dtype=int)
    edge_index_cn = np.nonzero(adj_cn)

    # Mutation
    genes_mu_corr = genes_mu_df.corr(method=method)
    n = genes_mu_df.shape[0]
    if p_value:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)
    adj_mu = np.where(np.abs(genes_mu_corr) > 0.30, 1, 0)
    adj_mu = adj_mu - np.eye(adj_mu.shape[0], dtype=int)
    edge_index_mu = np.nonzero(adj_mu)

    # Transcriptomic
    genes_trans_corr = genes_trans_df.corr(method=method)
    n = genes_trans_df.shape[0]
    if p_value:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)
    adj_trans = np.where(np.abs(genes_trans_corr) > thresh, 1, 0)
    adj_trans = adj_trans - np.eye(adj_trans.shape[0], dtype=int)
    edge_index_trans = np.nonzero(adj_trans)

    print("======")
    print(np.size(edge_index[0]), np.size(edge_index_cn[0]), np.size(edge_index_mu[0]), np.size(edge_index_trans[0]))
    edge_index = np.concatenate((edge_index, edge_index_cn, edge_index_mu, edge_index_trans), axis=1)

    np.save(os.path.join(save_path, 'edge_index_{}_{}.npy'.format(method, thresh)), edge_index)

    return n, edge_index





def ensp_to_hugo_map():
    with open('./data/9606.protein.info.v11.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map


def hugo_to_ncbi_map():
    with open('./data/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

    return hugo_map


def save_cell_graph(genes_path, save_path, edge_index):

    exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
    cn = pd.read_csv(os.path.join(genes_path, 'cn.csv'), index_col=0)
    mu = pd.read_csv(os.path.join(genes_path, 'mu.csv'), index_col=0)
    trans = pd.read_csv(os.path.join(genes_path, 'Cell_line_RMA_proc_basalExp.txt'), sep='\t', index_col=0)

    # 统一排序和索引
    index = exp.index
    columns = exp.columns
    scaler = StandardScaler()
    imp_mean = SimpleImputer(missing_values=float('nan'), strategy='mean')

    exp = pd.DataFrame(imp_mean.fit_transform(scaler.fit_transform(exp)), index=index, columns=columns)
    cn = pd.DataFrame(imp_mean.fit_transform(scaler.fit_transform(cn)), index=index, columns=columns)
    mu = pd.DataFrame(imp_mean.fit_transform(scaler.fit_transform(mu)), index=index, columns=columns)
    trans = pd.DataFrame(imp_mean.fit_transform(scaler.fit_transform(trans)), index=index, columns=columns)

    # 构造 cell_dict
    cell_dict = {}
    for i in exp.index:
        features = torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i], trans.loc[i]], dtype=torch.float).T
        cell_dict[i] = Data(x=features, edge_index=torch.tensor(edge_index, dtype=torch.long))

    torch.save(cell_dict, os.path.join(save_path, 'cell_graph.pt'))



def get_STRING_graph(genes_path, thresh=0.95):
    save_path = os.path.join(genes_path, 'edge_index_PPI_{}.npy'.format(thresh))

    if not os.path.exists(save_path):
        # gene_list
        exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)
        gene_list = exp.columns.to_list()
        gene_list = [int(gene[1:-1]) for gene in gene_list]

        # load STRING
        ensp_map = ensp_to_hugo_map()
        hugo_map = hugo_to_ncbi_map()
        edges = pd.read_csv('./data/9606.protein.links.detailed.v11.0.txt', sep=' ')

        # edge_index
        selected_edges = edges['combined_score'] > (thresh * 1000)
        edge_list = edges[selected_edges][["protein1", "protein2"]].values.tolist()

        edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                     edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]

        edge_list = [[hugo_map[edge[0]], hugo_map[edge[1]]] for edge in edge_list if
                     edge[0] in hugo_map.keys() and edge[1] in hugo_map.keys()]
        edge_index = []
        for i in edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):
                edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
                edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        edge_index = list(set(edge_index))
        edge_index = np.array(edge_index, dtype=np.int64).T

        # 保存edge_index
        # print(len(gene_list))
        # print(thresh, len(edge_index[0]) / len(gene_list))
        np.save(
            os.path.join('./data/CellLines_DepMap/CCLE_580_18281/census_706/', 'edge_index_PPI_{}.npy'.format(thresh)),
            edge_index)
    else:
        edge_index = np.load(save_path)

    return edge_index


def get_predefine_cluster(edge_index, save_path, thresh, device):
    save_path = os.path.join(save_path, 'cluster_predefine_PPI_{}.npy'.format(thresh))
    if not os.path.exists(save_path):
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(706, 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            print(len(cluster.unique()))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_path, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}

    return cluster_predefine


if __name__ == '__main__':
    gene_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    save_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    get_genes_graph(gene_path,save_path, thresh=0.55)
    save_cell_graph(gene_path, save_path)

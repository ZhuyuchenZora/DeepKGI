import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import load_data
from utils import EarlyStopping, set_random_seed
from utils import train, validate
from preprocess_gene import get_STRING_graph, get_predefine_cluster
from models.DeepKGI import DeepKGI

import argparse
import fitlog
import seaborn as sns
import matplotlib.pyplot as plot
import sys

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--model', type=str, default='DeepKGI', help='Name of the model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
    parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
    parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
    parser.add_argument('--layer', type=int, default=3, help='number of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dim for cell')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--edge', type=float, default=0.95, help='threshold for cell line graph')
    parser.add_argument('--setup', type=str, default='known', help='experimental setup')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='whether use pre-trained weights (0 for False, 1 for True')
    parser.add_argument('--weight_path', type=str, default='',
                        help='filepath for pretrained weights')
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test')
    return parser.parse_args()

def main():
    args = arg_parse()
    set_random_seed(args.seed)

    drug_dict = np.load('./data/Drugs/drug_feature_graph.npy', allow_pickle=True).item()
    cell_dict = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_all.npy',
                        allow_pickle=True).item()
    edge_index_P = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_{}.npy'.format(args.edge))
    
    NN0 = np.size(edge_index_P,0)
    NN = np.size(edge_index_P,1)
    print("NN",NN0,NN)
    """
    adj = np.eye(706)
    for i in range(NN):
        adj[edge_index_P[0][i],edge_index_P[1][i]]=1

    Adj=pd.DataFrame(adj)
    plt=sns.heatmap(Adj)
    plot.title("PPA")
    plot.show()
    
    """
    edge_index = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_pearson_0.55.npy'.format(args.edge))
    edge_index = np.concatenate([edge_index,edge_index_P],axis=1)
    
    IC = pd.read_excel('./data/IC50_GDSC/GDSC2_fitted_dose_response_27Oct23.xlsx')

    train_loader, val_loader, test_loader = load_data(IC, drug_dict, cell_dict, edge_index, args)
    print(len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print('mean degree:{}'.format(len(edge_index[0]) / 706))
    args.num_feature = cell_dict['ACH-000001'].x.shape[1]
    genes_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    edge_index = get_STRING_graph(genes_path, args.edge)
    cluster_predefine = get_predefine_cluster(edge_index, genes_path, args.edge, args.device)
    model = DeepKGI(cluster_predefine, args).to(args.device)

    if args.mode == 'train':
        if args.pretrain and args.weight_path != '':
            model.GNN_drug.load_state_dict(torch.load('./model_pretrain/{}.pth'.format(args.weight_path))['model_state_dict'])

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_folder = os.path.join(os.getcwd(), "logs", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        stopper = EarlyStopping(mode='lower', patience=args.patience)
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model, train_loader, criterion, opt, args.device)
            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)

            print('Evaluating...')
            # 更新为5个返回值，但早停只使用RMSE
            val_rmse, _, _, _, _ = validate(model, val_loader, args.device)
            print("Validation rmse:{}".format(val_rmse))
            fitlog.add_metric({'val': {'RMSE': val_rmse}}, step=epoch)

            early_stop = stopper.step(val_rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)

        # state ={'model_state_dict':model.state_dict()}
        # torch.save(state,'./weights/DeepKGI_pre.pth')

        train_rmse, train_MAE, train_r2, train_pearson, train_spearman = validate(model, train_loader, args.device)
        val_rmse, val_MAE, val_r2, val_pearson, val_spearman = validate(model, val_loader, args.device)
        test_rmse, test_MAE, test_r2, test_pearson, test_spearman = validate(model, test_loader, args.device)
        print('Train reslut: rmse:{} MAE:{} R²:{} Pearson:{} Spearman:{}'.format(
            train_rmse, train_MAE, train_r2, train_pearson, train_spearman))
        print('Val reslut: rmse:{} MAE:{} R²:{} Pearson:{} Spearman:{}'.format(
            val_rmse, val_MAE, val_r2, val_pearson, val_spearman))
        print('Test reslut: rmse:{} MAE:{} R²:{} Pearson:{} Spearman:{}'.format(
            test_rmse, test_MAE, test_r2, test_pearson, test_spearman))
        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
             "train": {
                 'RMSE': train_rmse,
                 'MAE': train_MAE,
                 'R2': train_r2,
                 'Pearson': train_pearson,
                 'Spearman': train_spearman
             },
             "valid": {
                 'RMSE': stopper.best_score,
                 'MAE': val_MAE,
                 'R2': val_r2,
                 'Pearson': val_pearson,
                 'Spearman': val_spearman
             },
             "test": {
                 'RMSE': test_rmse,
                 'MAE': test_MAE,
                 'R2': test_r2,
                 'Pearson': test_pearson,
                 'Spearman': test_spearman
             }})
    elif args.mode == 'test':
        weight = "DeepKGI_pre" if args.pretrain else "DeepKGI"  # 修正权重名称
        model.load_state_dict(
            torch.load('./weights/{}.pth'.format(weight), map_location=args.device)['model_state_dict'])
        test_rmse, test_MAE, test_r2, test_pearson, test_spearman = validate(model, test_loader, args.device)
        print('Test Results:')
        print('RMSE: {}'.format(round(test_rmse.item(), 4)))
        print('MAE: {}'.format(round(test_MAE, 4)))
        print('R²: {}'.format(round(test_r2, 4)))
        print('Pearson: {}'.format(round(test_pearson, 4)))
        print('Spearman: {}'.format(round(test_spearman, 4)))
if __name__ == "__main__":
    main()

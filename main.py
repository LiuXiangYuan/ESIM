import torch
import os
import argparse
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from time import time

from model import ESIM
from datasets import textDataset
from train import train
from infer import prediction


def get_data_dict(data, test=False):
    data_dict = dict()

    data_dict["premises"] = []
    data_dict["hypotheses"] = []
    if not test:
        data_dict["labels"] = []

    for i in range(data.shape[0]):
        text_a = [int(idx) for idx in data.text_a.iloc[i].split()]
        text_b = [int(idx) for idx in data.text_b.iloc[i].split()]

        data_dict["premises"].append(text_a)
        data_dict["hypotheses"].append(text_b)

        if not test:
            label = data.label.iloc[i]
            data_dict["labels"].append(label)

    return data_dict


def prepare_feature(train_data_path, test_data_path):
    '''
    读取数据，统计词量（暂时不使用与数据集匹配的词量，目前采用22000，多余的词量作为后续OOV嵌入）
    :param train_data_path: 训练集路径
    :param test_data_path: 测试集路径
    :return: 数据集，测试集和词量
    '''
    train_data = pd.read_table(train_data_path, names=['text_a', 'text_b', 'label'])
    test_data = pd.read_table(test_data_path, names=['text_a', 'text_b'])

    voc_num = 22000  # 0为pad

    return train_data, test_data, voc_num


def load_model(weight_path, args, device):
    '''
    加载模型
    :param weight_path: 模型权重保存路径
    :param device: 采用的设备-> cpu or gpu
    :return: 模型
    '''
    model = ESIM(vocab_size=args.vocab_size,
                 embedding_dim=args.embed_dim,
                 hidden_size=args.hidden_size,
                 num_classes=1,
                 device=args.device)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tianchi Text Matching')

    parser.add_argument("--output_dir", default='./model_output/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--predict_dir", default='./prediction_result/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for testing.")
    parser.add_argument("--learning_rate", default=0.0004, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_epochs", default=64, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--embed_dim", default=300, type=int,
                        help="The dimension of embedding layer.")
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="Hidden size of LSTM.")
    parser.add_argument("--max_gradient_norm", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--patience", default=5, type=int,
                        help="Max patience count.")
    parser.add_argument("--n_split", default=1, type=int,
                        help="K-fold.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--train", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--infer", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    train_data_path = 'gaiic_track3_round1_train_20210228/gaiic_track3_round1_train_20210228.tsv'
    test_data_path = 'gaiic_track3_round1_testA_20210228/gaiic_track3_round1_testA_20210228.tsv'

    train_data, test_data, vocab_size = prepare_feature(train_data_path, test_data_path)
    args.vocab_size = vocab_size

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.train:
        criterion = torch.nn.BCEWithLogitsLoss()
        folds = StratifiedKFold(n_splits=args.n_split,
                                shuffle=True,
                                random_state=2021).split(np.arange(train_data.shape[0]), train_data.label.values)
        kfold_best = []
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f'train fold {fold}')
            model = ESIM(vocab_size=args.vocab_size,
                         embedding_dim=args.embed_dim,
                         hidden_size=args.hidden_size,
                         num_classes=1,
                         device=args.device)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)
            fold_train_data = get_data_dict(train_data.loc[train_idx, :].reset_index(drop=True))
            train_dataset = textDataset(fold_train_data)
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=0)

            fold_val_data = get_data_dict(train_data.loc[val_idx, :].reset_index(drop=True))
            val_dataset = textDataset(fold_val_data)
            val_loader = DataLoader(val_dataset,
                                    batch_size=args.val_batch_size,
                                    shuffle=False,
                                    num_workers=0)
            args.fold = fold
            best_loss = train(model, args, train_loader, val_loader, criterion, optimizer, lr_scheduler)
            kfold_best.append(best_loss)

        print('local cv:', kfold_best, np.mean(kfold_best))

    if args.infer:
        model_list = []
        for i in range(args.n_split):
            model_list.append(load_model(args.output_dir + 'fold_' + str(i + 1) + '_best.pth', args, device))

        if not os.path.exists(args.predict_dir):
            os.makedirs(args.predict_dir)

        fold_test_data = get_data_dict(test_data, test=True)
        test_dataset = textDataset(fold_test_data, test=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        start_time = time()
        predict = prediction(model_list, test_loader, device)
        end_time = time()
        print('Spend time:', (end_time - start_time) // 60)

        pd.DataFrame(predict).to_csv(args.predict_dir + "result.csv", index=False, header=False)

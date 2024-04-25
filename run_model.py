
__auther__ = 'dovkess@gmail.com'

import argparse
import cProfile
import io
import time
import datetime
import os
import pickle
from contextlib import ExitStack

import torch
from matplotlib import pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from file_system_storage import FS
from model import GCPNet
from torch.utils.data import Subset

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


EPOC_STEP = 50
CHECK_POINT_PATH = '/models/dov/'
DATA_SET_PATH = 'data/dov2/'

def save_model(epoc, model, acc, loss, opt, val_acc, prev_best=None):
    prev_best = prev_best or -1
    obj = {
            'epoc': epoc,
            'model': model.state_dict(),
            'acc': acc,
            'loss': loss,
            'optimizer_state_dict': opt.state_dict(),
            'test_acc': val_acc,
            'prev_best': prev_best
        }
    name = 'checkpoint_no_V_{epoc}.pt'
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    FS().upload_data(buffer, os.path.join(CHECK_POINT_PATH, name.format(epoc=epoc)))
    if prev_best < val_acc:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)
        FS().upload_data(buffer, os.path.join(CHECK_POINT_PATH, name.format(epoc="best")))
    return max(prev_best, val_acc)

def train(batch_data, train=True):
    matrixs = [torch.tensor(batch_data[i][0]) for i in range(len(batch_data))]
    cn = torch.tensor([batch_data[i][1] for i in range(len(batch_data))])
    edges = [batch_data[i][2] for i in range(len(batch_data))]
    split = torch.tensor([[len(mat), len(mat)] for mat in matrixs])
    colors_mat = [torch.tensor([[1] * cn[i]] * len(matrixs[i])) for i in range(len(batch_data))]
    labels = torch.randint(0, 2, size=(len(matrixs),)).float()
    for i in range(len(matrixs)):
        if labels[i] == 0:
            v, u = edges[i]
            matrixs[i][v][u] = matrixs[i][u][v] = 1

    with ExitStack() as stack:
        prev_train = gcp.training
        if train==False:
            gcp.eval()
            stack.enter_context(torch.no_grad())
        elif train==True:
            gcp.train()

        M_vv = torch.block_diag(*matrixs).float()
        M_vc = torch.block_diag(*colors_mat).float()
        opt.zero_grad()
        split = split[:, 0].tolist()
        pred = gcp(M_vv.to(device=device), M_vc.to(device=device), split)
        l = loss(pred, labels.reshape(-1, 1).to(device=device))
        if train!=False:
            l.backward()
            opt.step()
        gcp.train(prev_train)

    return l.detach(), ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).float().mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default='./train_jsons')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epocs', type=int, default=3500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--number_of_batches', type=int, default=128)

    args = parser.parse_args()

    files = FS().list_items(DATA_SET_PATH)  #list from wasabi
    cpu_device = 'cpu'
    device = args.device
    embedding_size = args.embedding_size
    tmax=args.tmax
    number_of_batches = args.number_of_batches
    batch_size = args.batch_size
    graph_dir = args.graph_dir

    gcp = GCPNet(embedding_size, tmax=tmax, device=device)
    gcp.to(device)
    opt, loss = gcp.optimizer_and_loss()


    # ds = GraphDataSet(graph_dir , batch_size=batch_size)
    # ss = Subset(ds, torch.randperm(len(ds))[:number_of_batches])
    # dl = DataLoader(ss, batch_size=number_of_batches, shuffle=True)


    epoc_loss = []
    epoc_loss_val = []
    epoc_acc = []
    epoc_acc_val = []

    split_test_train = int(len(files)*0.8)
    files_train = files[:split_test_train]
    files_val = files[split_test_train:]
    prev_best = None
    for epoch_i in range(332):
        plot_loss = 0
        plot_loss_val = 0
        plot_acc = 0
        plot_acc_val = 0
        t1 = time.perf_counter()
        print('Running epoc: {}'.format(epoch_i))

        train_count = 0
        for f_i, f in enumerate(files_train):
            f_data = FS().get_data(f)
            f_data = pickle.load(io.BytesIO(f_data))
            for batch_i, batch_data in enumerate(chunk(f_data,16)):
                train_count += len(batch_data)
                loss_train, acc_train = train(batch_data)
                plot_loss += loss_train
                plot_acc += acc_train

        val_count = 0
        for f_i, f in enumerate(files_val):
            f_data = FS().get_data(f)
            f_data = pickle.load(io.BytesIO(f_data))
            for batch_i, batch_data in enumerate(chunk(f_data,16)):
                val_count += len(batch_data)
                loss_val, acc_val = train(batch_data, train=False)
                plot_loss_val += loss_val
                plot_acc_val += acc_val

        t2 = time.perf_counter()
        print('Time: t2-t1={}'.format(t2-t1))
        plot_acc /= train_count if train_count!=0 else 1
        plot_loss /= train_count if train_count!=0 else 1
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)
        plot_acc_val /= val_count if val_count!=0 else 1
        plot_loss_val /= val_count if val_count!=0 else 1
        epoc_acc_val.append(plot_acc_val)
        epoc_loss_val.append(plot_loss_val)

        print('Saving model')
        prev_best = save_model(epoch_i, gcp, epoc_acc, epoc_loss, opt, plot_acc_val, prev_best=prev_best)

        print('Train Accuracy: {}\tLoss: {}'.format(plot_acc, plot_loss),
              'Val Accuracy: {}\tLoss: {}'.format(plot_acc_val, plot_loss_val))
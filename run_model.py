
__auther__ = 'dovkess@gmail.com'

import argparse
import cProfile
import time
import datetime
import os
import pickle

import torch
from matplotlib import pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from model import GCPNet


EPOC_STEP = 50
CHECK_POINT_PATH = '/content/drive/MyDrive/project_MSC/checkpoints/'
DATA_SET_PATH = '/content/pickles/pickles/'  # '/content/drive/MyDrive/project_MSC/train_jsons'  #
DEVICE =  'cpu'  # 'cuda'  #

def save_model(epoc, model, acc, loss, opt, test_acc):
    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    torch.save(
        {
            'epoc': epoc,
            'model': model.state_dict(),
            'acc': acc,
            'loss': loss,
            'optimizer_state_dict': opt.state_dict(),
            'test_acc': test_acc
        },
        '{}/checkpoint_no_V_{}_{}.pt'.format(CHECK_POINT_PATH, dt, epoc)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default='./train_jsons')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epocs', type=int, default=3500)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)
    ds = GraphDataSet(args.graph_dir, batch_size=args.batch_size)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    opt = torch.optim.Adam(gcp.parameters(), lr=2e-5, weight_decay=1e-10)
    loss = torch.nn.BCELoss()

    plot_loss = 0
    plot_acc = 0
    epoc_loss = []
    epoc_acc = []
    epoch_size = 128

    v_normal = torch.normal(mean=0., std=1., size=(1, embedding_size))/8.
    for i in range(args.epochs):
        t1 = time.perf_counter()
        print('Running epoc: {}'.format(i))
        for j, b in enumerate(dl):
            # print('Running batch: {}'.format(j))
            if j == epoch_size:
                break
            M_vv, labels, cn, split, M_vc = b
            # M_vv = M_vv.squeeze()
            labels = labels.squeeze()
            cn = cn.squeeze()
            split = split.squeeze()
            # M_vc = M_vc.squeeze()
            split = [int(s) for s in split]
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()

            # Added new init
            v_size = M_vv.shape[0]
            V = v_normal.expand(v_size, embedding_size)
            cn = cn.squeeze()
            C = c_rand.expand(int(sum(cn)), embedding_size)
            opt.zero_grad()

            # Move to Cuda
            M_vv = M_vv.to(device=DEVICE)
            M_vc = M_vc.to(device=DEVICE)
            V = V.to(device=DEVICE)
            C = C.to(device=DEVICE)
            # end move to cuda

            pred, V_ret, C_ret = gcp.forward(M_vv, M_vc, V, C, split)
            l = loss(pred, torch.Tensor(labels).to(device=DEVICE))
            plot_loss += l.detach()
            plot_acc += sum((pred.detach().cpu() > 0.5) == torch.Tensor(labels))/float(len(cn))
            l.backward()
            opt.step()
        t2 = time.perf_counter()
        print('Time: t2-t1={}'.format(t2-t1))
        print(v_normal)
        plot_acc /= epoch_size
        plot_loss /= epoch_size
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)
        if (i % EPOC_STEP) == 0:
            print('Saving model')
            save_model(i, gcp, epoc_acc, epoc_loss, opt)
        print('Accuracy: {}\tLoss: {}'.format(plot_acc, plot_loss))

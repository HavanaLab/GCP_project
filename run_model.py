
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
# CHECK_POINT_PATH = './checkpoints/second_fix'
CHECK_POINT_PATH = './checkpoints/first_fix'
DATA_SET_PATH = '/content/pickles/pickles/'  # '/content/drive/MyDrive/project_MSC/train_jsons'  #
DEVICE =  'cpu'  # 'cuda'  #

def save_model(epoc, model, acc, loss, opt, test_acc, best):
    new_best = best<test_acc

    best = max(best, test_acc)
    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(CHECK_POINT_PATH, exist_ok=True)
    save_obj = {
            'epoc': epoc,
            'model': model.state_dict(),
            'acc': acc[-1],
            'loss': loss[-1],
            'optimizer_state_dict': opt.state_dict(),
            'test_acc': test_acc,
            "best": best,
        }
    torch.save(save_obj, '{}/checkpoint_no_V_{}_{}.pt'.format(CHECK_POINT_PATH, dt, epoc))
    if new_best: torch.save(save_obj, '{}/best.pt'.format(CHECK_POINT_PATH))
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default='./train_jsons')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--check_path', type=str, default=None)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)
    ds = GraphDataSet(args.graph_dir, batch_size=args.batch_size)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    test_ds = GraphDataSet(os.path.join("/", *args.graph_dir.split("/")[:-1], "test"), batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    opt = torch.optim.Adam(gcp.parameters(), lr=2e-5, weight_decay=1e-10)
    loss = torch.nn.BCELoss()

    plot_loss = 0
    plot_acc = 0
    epoc_loss = []
    epoc_acc = []
    epoch_size = 128
    accumulated_size = 0
    start_epoc = 0
    best = -1
    # load checkpoint
    if args.check_path is not None:
        checkpoint = torch.load(args.check_path)
        gcp.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if "best" in checkpoint:
            best = checkpoint["best"]

        epoc_acc = checkpoint['acc'] if hasattr(epoc_acc, '__iter__') else epoc_acc.append(checkpoint['acc'])

        start_epoch_string = args.check_path.split("/")[-1].split("_")[-1].split(".")[0]
        if "epoc" in checkpoint:
            start_epoc = checkpoint['epoc']
        if start_epoch_string.isdigit():
            start_epoc = int(start_epoch_string)

        start_epoc = checkpoint['epoc']
        # gcp.set_ch(checkpoint['C_h'])
        # gcp.set_vh(checkpoint['V_h'])

    for i in range(start_epoc, args.epochs):
        t1 = time.perf_counter()
        print('Running epoc: {}'.format(i))
        for j, b in enumerate(dl):
            # print('Running batch: {}'.format(j))
            if j == epoch_size: break
            M_vv, labels, cn, split, M_vc = b
            # M_vv = M_vv.squeeze()
            labels = labels.squeeze()
            # cn = cn.squeeze()
            split = split.squeeze(0)
            # M_vc = M_vc.squeeze()
            split = [int(s) for s in split]
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()

            # Added new init
            v_size = M_vv.shape[0]
            # cn = cn.squeeze()
            opt.zero_grad()

            # Move to Cuda
            M_vv = M_vv.to(device=args.device)
            M_vc = M_vc.to(device=args.device)

            # end move to cuda

            pred, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn)
            l = loss(pred.to(DEVICE), torch.Tensor(labels).to(device=DEVICE))
            plot_loss += l.detach()
            plot_acc += ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum()/float(cn.shape[1])
            # print(cn.shape[1])
            # accumulated_size += 1 if len(pred.shape) == 0 else pred.shape[0]

            if not args.test:
                l.backward()
                opt.step()
        t2 = time.perf_counter()
        print('Time: t2-t1={}'.format(t2-t1))
        plot_acc /= j
        plot_loss /= j
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)


        # Test
        if not args.test:
            test_acc = 0
            for j, b in enumerate(test_dl):
                if j == epoch_size: break
                M_vv, labels, cn, split, M_vc = b
                labels = labels.squeeze()
                split = split.squeeze(0)
                split = [int(s) for s in split]
                M_vv = M_vv.squeeze()
                M_vc = M_vc.squeeze()
                v_size = M_vv.shape[0]
                M_vv = M_vv.to(device=args.device)
                M_vc = M_vc.to(device=args.device)
                pred, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn)
                test_acc += ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum()/float(cn.shape[1])
            test_acc /= j
        else:
            test_acc = -1
        if best < test_acc or (i % EPOC_STEP) == 0 and not args.test:
            print('Saving model')
            best = save_model(i, gcp, epoc_acc, epoc_loss, opt, test_acc, best)
        print('Accuracy: {}\tLoss: {}\t Test: {}\t Max: {}'.format(plot_acc, plot_loss, test_acc, best))

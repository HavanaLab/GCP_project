
__auther__ = 'dovkess@gmail.com'

import argparse
import cProfile
import time

import torch
from matplotlib import pyplot as plt
from torch.backends import cudnn
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from model import GCPNet


EPOC_STEP = 5
CHECK_POINT_PATH = 'C:/Users/Nehama/DovProject/checkpoints/'


def save_model(epoc, model, acc, loss, opt):
    torch.save(
        {
            'epoc': epoc,
            'model': model.state_dict(),
            'acc': acc,
            'loss': loss,
            'vh': model.get_vh(),
            'ch': model.get_ch(),
            'optimizer_state_dict': opt.state_dict()
        },
        '{}/checkpoint_8_batch_{}.pt'.format(CHECK_POINT_PATH, epoc)
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

    opt = torch.optim.Adam(gcp.parameters(), lr=2e-5)
    loss = torch.nn.BCELoss()

    plot_loss = 0
    plot_acc = 0
    epoc_loss = []
    epoc_acc = []

    # # profiling
    # pr = cProfile.Profile()
    # pr.enable()

    for i in range(args.epocs):
        t1 = time.perf_counter()
        print('Running epoc: {}'.format(i))
        for j, b in enumerate(dl):
            # print('Running batch: {}'.format(j))
            M_vv, labels, cn, split, M_vc = b
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()
            v_size = M_vv.shape[0]
            V = torch.rand(v_size, embedding_size)
            C = torch.rand(int(sum(cn)), embedding_size)
            opt.zero_grad()
            # Move to Cuda
            M_vv = M_vv.to(device=args.device)
            M_vc = M_vc.to(device=args.device)
            V = V.to(device=args.device)
            C = C.to(device=args.device)
            # end move to cuda
            pred = gcp.forward(M_vv, M_vc, V, C, split)
            l = loss(pred, torch.Tensor(labels).to(device=args.device))
            plot_loss += l.detach()
            plot_acc += sum((pred.detach().cpu() > 0.5) == torch.Tensor(labels))/float(len(cn))
            l.backward()
            opt.step()
        t2 = time.perf_counter()
        print('Time: t2-t1={}'.format(t2-t1))
        plot_acc /= len(dl)
        plot_loss /= len(dl)
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)
        if (i % EPOC_STEP) == 0:
            print('Saving model')
            save_model(i, gcp, epoc_acc, epoc_loss, opt)
        print('Accuracy: {}\tLoss: {}'.format(plot_acc, plot_loss))

    # END profiling
    # pr.disable()
    # pr.dump_stats('{}/profile.pstat'.format(CHECK_POINT_PATH))

    plt.plot(plot_loss)
    plt.plot(plot_acc)
    plt.show()

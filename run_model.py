
__auther__ = 'dovkess@gmail.com'

import argparse
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from model import GCPNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default='/home/dov/openu/Project/pytorch_GNN/GCP_project/adversarial-training')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epocs', type=int, default=50)
    args = parser.parse_args()

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax)
    ds = GraphDataSet(args.graph_dir, batch_size=args.batch_size)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    opt = torch.optim.Adam(gcp.parameters(), lr=2e-5)
    loss = torch.nn.BCELoss()

    plot_loss = 0
    plot_acc = 0
    epoc_loss = []
    epoc_acc = []

    for i in range(args.epocs):
        print('Running epoc: {}'.format(i))
        for j, b in enumerate(dl):
            # print('Running batch: {}'.format(j))
            M_vv, labels, cn, split, M_vc = b
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()
            v_size = M_vv.shape[0]
            V = torch.rand(v_size, embedding_size)
            C = torch.rand(int(sum(cn)), embedding_size)
            # M_vc = torch.zeros([v_size, int(sum(cn))])  # error. this should be ones only where there are nodes (mostly sparce)
            opt.zero_grad()
            pred = gcp.forward(M_vv, M_vc, V, C, split)
            l = loss(pred, torch.Tensor(labels))
            plot_loss += l.detach()
            plot_acc += sum((pred.detach() > 0.5) == torch.Tensor(labels))/float(len(cn))
            l.backward()
            opt.step()
        plot_acc /= len(dl)
        plot_loss /= len(dl)
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)
        print('Accuracy: {}\tLoss: {}'.format(plot_acc, plot_loss))

    plt.plot(plot_loss)
    plt.plot(plot_acc)
    plt.show()

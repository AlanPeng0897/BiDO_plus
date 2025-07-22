import seaborn as sns
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from utils import anom_utils, utils
from models.classifier import Classifier
from metrics.accuracy import Accuracy


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results


def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    print("len of all results", float(len(all_results)))
    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results


def get_energy_score(model, val_loader):
    model.eval()
    init = True
    in_energy = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            images = images.cuda()
            # labels = labels.cuda().float()
            _, outputs = model(images)
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy()
            in_energy.update(e_s.mean())  #DEBUG

            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))

    print('Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(in_energy=in_energy))
    return -1 * sum_energy


def get_msp_score(model, val_loader):
    init = True
    in_energy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i > 48:
                break
            if images.shape[0] == 1:
                continue
            _, outputs = model(images.cuda())

            nclasses = 1000
            # outputs = outputs[:, :nclasses]
            soft_label = F.softmax(outputs, dim=1).detach().cpu().numpy()
            in_soft_label = soft_label[:, :nclasses]
            scores = np.max(in_soft_label, axis=1)

            in_energy.update(scores.mean())

            if init:
                sum_scores = scores
                init = False
            else:
                sum_scores = np.concatenate((sum_scores, scores))

    print('Min Conf: ', np.min(sum_scores), 'Max Conf: ', np.max(sum_scores), 'Average Conf: ', np.mean(sum_scores))

    return sum_scores


def get_score(model, val_loader, mode='msp'):
    if mode == "energy":
        score = get_energy_score(model, val_loader)

    elif mode == "msp":
        score = get_msp_score(model, val_loader)

    return score


def main():
    metric = Accuracy
    OE_args=loaded_args["outlier_exp"]
    bido_args=loaded_args["bido"]

    model_config = loaded_args["model"]
    target_model = Classifier(**model_config, bido_args=bido_args, OE_args=OE_args, weights=args.path_T)

    # path_T = args.path_T

    # ckp_T = torch.load(path_T)
    # target_model.load_state_dict(ckp_T['model_state_dict'])
    test_acc = target_model.dry_evaluate(testloader, metric)

    print(f"test_acc:{test_acc:.4f}")

    mode = 'msp' #'msp' #'energy'
    ood_sum_score = get_score(target_model, testloaderOut, mode)
    id_sum_score = get_score(target_model, testloader, mode)

    auroc, aupr, fpr = anom_utils.get_and_print_results(id_sum_score, ood_sum_score, "CelebA", "100")
    results = cal_metric(known=id_sum_score, novel=ood_sum_score, method = "msp sum")

    plt.style.use('seaborn-darkgrid')
    custom_palette = sns.color_palette(["#000080", "#7CFC00"])

    if args.enable_bido and args.enable_OE:
        title = "BiDO Model"
        save_path = f"plot_figures/BiDO_OE_{mode}.pdf"

    elif args.enable_bido:
        title = "BiDO Model"
        save_path = f"plot_figures/BiDO_{mode}.pdf"

    else:
        title = "Regular Model"
        save_path = f"plot_figures/Reg_{mode}.pdf"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    fontsize = 23
    if mode == 'energy':
        plt.xlabel('Free Energy Score', fontweight="bold", fontsize=fontsize)

    elif mode == 'msp':
        plt.xlabel('Softmax Score', fontsize=fontsize)

    data = {
        "MSP": np.concatenate((id_sum_score, ood_sum_score)),
        "data": ["ID"] * len(id_sum_score) + ["OOD"] * len(ood_sum_score)
    }

    ax = sns.kdeplot(id_sum_score, label="ID", multiple="stack", common_norm=True)
    ax = sns.kdeplot(ood_sum_score, label="OOD", multiple="stack", common_norm=True)

    plt.tick_params(labelsize=18)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylabel('Density', fontsize=fontsize)
    ax.set_title(title, fontweight="bold", fontsize=fontsize)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cxr')

    parser.add_argument('--enable_bido', action='store_true', help='multi-layer constraints')
    parser.add_argument('--enable_OE',action='store_true', help='OE constraints')

    parser.add_argument('--path_T', '-t',  default='results/celeba/HSIC/0_0/20230618_172529/Classifier_0.8958.pth', help='')
    parser.add_argument('--config', '-c', default='', help='')
    parser.add_argument('--save_path', default='./results', help='')

    args = parser.parse_args()

    loaded_args = utils.load_json(json_file=args.config)

    loaded_args['bido']['enable_bido'] = args.enable_bido
    loaded_args['outlier_exp']['enable_OE'] = args.enable_OE

    in_dataset_file = loaded_args['dataset']['test_file']

    _, testloader = utils.init_dataloader(loaded_args, in_dataset_file, mode="test")
    _, testloaderOut = utils.init_dataloader(loaded_args, mode="test_ood")


    main()


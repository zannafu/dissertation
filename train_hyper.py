import math
import higher
import transformers

import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset

import Metrics
import helpers
from utils import *
import lossFunc
import hypergrad as hg


class Task:
    """
    Handles the train and validation loss for a single task
    """

    def __init__(self, model, train_loader, val_loader, unlabeled_loader, args, deter_val_loader=None):
        device = next(model.parameters()).device
        self.args = args
        # stateless version of model
        self.fmodel = higher.monkeypatch(model, device=device, copy_initial_weights=True)

        self.n_params = len(list(model.parameters()))
        self.train_loader, self.val_loader, self.unlabeled_loader = train_loader, val_loader, unlabeled_loader
        self.train_bsize, self.val_bsize = self.train_loader.batch_size, self.val_loader.batch_size
        self.val_loss, self.val_acc = None, None
        self.threshold = 0.
        self.prior = 0.
        self.val_prior = 0.

        self.deter_val_loader = deter_val_loader

        self.train_meters = helpers.AverageMeterSet()

    def train_loss_f(self, params, hparams):
        self.fmodel.train()
        data, labels, true_labels = next(iter(self.train_loader))
        if self.args.cuda:
            data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()
        net_out = self.fmodel(data, params=params)

        # unlabeled data with linear weight
        probs = torch.sigmoid(net_out)
        weights = 1.0 - probs.detach().clone() / self.threshold
        weights = torch.where(weights > 0., weights, torch.tensor(0., dtype=torch.float, device=weights.device))
        # positive data with weight=1.0
        weights = torch.where(labels == 1, torch.tensor(1., dtype=torch.float, device=weights.device), weights)

        # loss w.r.t. pseudo labels
        if self.args.loss == 'bce':
            loss = lossFunc.bce_loss(net_out, labels, weights)
        else:
            loss = getattr(lossFunc, f'{self.args.loss}_loss')(net_out, labels, self.prior, weights)
        # TODO: don't use reg?
        reg = sum([(param ** 2).sum() for param in params])
        loss += 0.5 * 0.001 * reg

        with torch.no_grad():
            # loss w.r.t. true labels
            true_loss = lossFunc.bce_loss(net_out, labels, weights)
            # acc w.r.t. true labels
            acc = Metrics.accuracy(net_out, true_labels)

        self.train_meters.update('pseudo_loss', loss.item(), labels.size(0))
        self.train_meters.update('true_loss', true_loss.item(), labels.size(0))
        self.train_meters.update('accuracy', acc, labels.size(0))
        return loss

    def val_loss_f(self, params, hparams):
        self.fmodel.eval()
        data, labels = next(iter(self.val_loader))
        if self.args.cuda:
            data, labels = data.cuda(), labels.cuda()
        net_out = self.fmodel(data, params=params)
        if self.args.loss == 'bce':
            loss = lossFunc.bce_loss(net_out, labels)
        else:
            loss = getattr(lossFunc, f'{self.args.loss}_loss')(net_out, labels, self.val_prior)
        acc = Metrics.accuracy(net_out, labels)

        self.val_loss = loss.item()  # avoid memory leaks
        self.val_acc = acc

        return loss

    def evaluate(self, params, hparams):
        self.fmodel.eval()
        acc_meter = helpers.AverageMeter()
        loss_meter = helpers.AverageMeter()
        val_loader = self.deter_val_loader if self.deter_val_loader is not None else self.val_loader
        with torch.no_grad():
            for data, labels in val_loader:
                if self.args.cuda:
                    data, labels = data.cuda(), labels.cuda()
                net_out = self.fmodel(data, params=params)
                loss = lossFunc.bce_loss(net_out, labels)
                acc = Metrics.accuracy(net_out, labels)
                acc_meter.update(acc, labels.size(0))
                loss_meter.update(loss.item(), labels.size(0))
        return loss_meter.avg, acc_meter.avg

    def estimate_prior(self, params, hparams):
        self.fmodel.eval()
        probs = []
        true_labels = []
        with torch.no_grad():
            for data, labels in self.unlabeled_loader:
                if self.args.cuda:
                    data, labels = data.cuda(), labels.cuda()
                prob = torch.sigmoid(self.fmodel(data, params=params))
                if prob.dim() == 0:
                    prob = prob.reshape(1)
                probs.append(prob)
                if labels.dim() == 0:
                    labels = labels.reshape(1)
                true_labels.append(labels)
            probs = torch.cat(probs, dim=0)
            true_labels = torch.cat(true_labels, dim=0)
            weights = 1.0 - probs / self.threshold
            weights = torch.where(weights > 0., weights, torch.tensor(0., dtype=torch.float, device=weights.device))
            # prior = weights[probs >= 0.5].sum().item() / max(weights.sum().item(), 1e-5)
            self.prior = weights[true_labels == 1].sum().item() / max(weights.sum().item(), 1e-5)
            # self.prior = self.args.prior
        probs = []
        with torch.no_grad():
            for data, label in self.unlabeled_loader:
                if self.args.cuda:
                    data, label = data.cuda(), label.cuda()
                prob = torch.sigmoid(self.fmodel(data, params=params))
                if prob.dim() == 0:
                    prob = prob.reshape(1)
                probs.append(prob)
            probs = torch.cat(probs, dim=0)
            self.val_prior = (probs >= 0.5).float().sum().item() / probs.size(0)

    def reset_meters(self):
        self.train_meters.reset()


def train_hyper(model, positive_dataset, unlabeled_dataset, val_dataset, args):
    deter_val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)

    o_alpha = torch.tensor(-math.log(1.0 / args.alpha - 1)).requires_grad_(True)
    o_eta = torch.tensor(math.log(args.eta - 1.0)).requires_grad_(True)
    hyper_params = [o_alpha, o_eta]
    outer_opt = torch.optim.Adam(lr=args.outer_lr, params=hyper_params)

    positive_data, positive_labels = positive_dataset.X, positive_dataset.y
    unlabeled_data, unlabeled_labels = unlabeled_dataset.X, unlabeled_dataset.y
    train_data = torch.cat((positive_data, unlabeled_data), dim=0)
    train_labels = torch.cat((positive_labels, -torch.ones_like(unlabeled_labels, dtype=unlabeled_labels.dtype)), dim=0)
    train_true_labels = torch.cat((positive_labels, unlabeled_labels), dim=0)
    train_loader = DataLoader(TensorDataset(train_data, train_labels, train_true_labels), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    # for prior estimation
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size_val, shuffle=False)

    history_val_loss = []
    history_val_acc = []

    last_params = None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True)
    task = Task(model, train_loader, val_loader, unlabeled_loader, args, deter_val_loader=deter_val_loader)

    for hyper_iter in range(args.outer_iters):
        alpha = torch.sigmoid(hyper_params[0])
        eta = torch.exp(hyper_params[1]) + 1.0
        helpers.prYellow(f'Hyper-Iter [{hyper_iter + 1} / {args.outer_iters}]   alpha = {alpha.item()}    eta = {eta.item()}')
        outer_opt.zero_grad()
        params = [p.detach().clone().requires_grad_(True) for p in model.parameters()]
        inner_opt = optim.Adam(params, lr=args.lr)

        scheduler = transformers.get_constant_schedule_with_warmup(inner_opt, num_warmup_steps=args.n_warmup)

        val_best_acc = 0
        val_best_index = -1
        val_best_param = [p.detach().clone().requires_grad_(True) for p in model.parameters()]

        for epoch in range(args.epochs):
            max_upd = math.floor(math.log(1.0 / alpha.item(), eta.item()))
            task.threshold = alpha.item() * math.pow(eta.item(), min(epoch, max_upd))
            # calculate prior
            task.estimate_prior(params, hyper_params)

            for inner_epoch in range(args.inner_epochs):
                for train_iter in range(len(task.train_loader)):
                    inner_opt.zero_grad()
                    loss = task.train_loss_f(params, hyper_params)
                    loss.backward()
                    inner_opt.step()
            scheduler.step()

            val_loss, val_acc = task.evaluate(params, hyper_params)
            print(
                f'Epoch [{epoch + 1} / {args.epochs}]   Pseudo_Loss: {task.train_meters["pseudo_loss"].avg:.5f}   True_Loss: {task.train_meters["true_loss"].avg:.5f}    True_Acc: {task.train_meters["accuracy"].avg * 100.0:.5f}   val_loss: {val_loss:.5f}    val_acc: {val_acc * 100.0:.5f}')
            task.reset_meters()
            # Early stop
            if val_acc > val_best_acc:
                val_best_acc = val_acc
                val_best_index = epoch
                val_best_param = [p.detach().clone() for p in params]
            else:
                if epoch - val_best_index >= args.patience:
                    params = val_best_param
                    break

        max_upd = math.floor(math.log(1.0 / alpha.item(), eta.item()))
        task.threshold = alpha * torch.pow(eta, min(val_best_index, max_upd))
        cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=1.)
        hg.CG(params, hyper_params, K=args.hyper_K, fp_map=cg_fp_map, outer_loss=task.val_loss_f, stochastic=False, tol=1e-12)
        outer_opt.step()
        history_val_loss.append(task.val_loss)
        history_val_acc.append(task.val_acc)
        last_params = params

        torch.cuda.empty_cache()

        # warm start
        if args.warm_start:
            state_dict = model.state_dict()
            for idx, k in enumerate(state_dict.keys()):
                state_dict[k] = last_params[idx]
            model.load_state_dict(state_dict)

    state_dict = model.state_dict()
    for idx, k in enumerate(state_dict.keys()):
        state_dict[k] = last_params[idx]
    model.load_state_dict(state_dict)
    history = {'val_loss': history_val_loss, 'val_acc': history_val_acc}
    return history

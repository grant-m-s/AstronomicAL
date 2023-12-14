'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

import torch.nn.functional as F
import numpy as np
from .strategy import Strategy
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from sklearn.metrics import f1_score


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda


LR = 0.1
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4
device_global = 'cuda'

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()


    def forward(self, x):
        return x.view(x.size(0), -1)

# 硬改成根据输入层数设计lossnet
class LossNet(nn.Module):
    def __init__(self, features):
        # feature_sizes=[32, 16, 8, 4], num_channels=[16, 32, 64, 128]
        super(LossNet, self).__init__()
        self.num_layers = len(features)

        interm_dim = 128
        feature_sizes = []
        num_channels = []
        for f in features:
            num_channels.append(f.size(1))
            feature_sizes.append(f.size(2))
        
        self.GAP_list = []
        self.FC_list = []
        for num in range(self.num_layers):
            self.GAP_list.append(nn.AvgPool2d(feature_sizes[num]).to(device_global) )
            self.FC_list.append(nn.Linear(num_channels[num], interm_dim).to(device_global))

        self.linear = nn.Linear(self.num_layers * interm_dim, 1)


    def forward(self, features):
        out_list = []

        for num in range(self.num_layers):
            out = self.GAP_list[num](features[num])
            out = out.view(out.size(0), -1)
            out = self.FC_list[num](out)
            out = F.relu(out)
            out_list.append(out)
        # print(torch.cat(out_list, 1).size())
        out = self.linear(torch.cat(out_list, 1))
        return out


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
        return

    return loss

class LearningLoss(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(LearningLoss, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        global device_global
        device_global = self.device
        
        

    def ll_train(self, epoch, loader_tr, optimizers,criterion):
        self.clf.train()
        self.loss_module.train()
        accFinal = 0.
        accLoss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.to(self.device) ), Variable(y.to(self.device) )
            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()
            scores, e1, features = self.clf(x,intermediate = True)
            target_loss = criterion(scores, y)
            if epoch > 120:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                for feature in features:
                    feature = feature.detach()
            pred_loss = self.loss_module(features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = LossPredLoss(
                pred_loss, target_loss, margin=MARGIN)
            loss = m_backbone_loss + WEIGHT * m_module_loss


            loss.backward()
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            for p in filter(lambda p: p.grad is not None, self.loss_module.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            
            optimizers['backbone'].step()
            optimizers['module'].step()



            accFinal += torch.sum((torch.max(scores,1)[1] == y).float()).data.item()
            accLoss += loss.item()
            # if batch_idx % 10 == 0:
            #     print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr.dataset.X), accLoss

    def train(self,alpha=0, n_epoch=80, qs_dict=None):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        transform = self.args.transform_tr
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)                      

        # n_epoch = self.args.n_epoch']
        self.clf = self.net.apply(weight_reset).to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(self.clf.parameters(), lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        # sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

        print("current:",len(self.X[idxs_train]))
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.to(self.device) ), Variable(y.to(self.device) )
            scores, e1, features = self.clf(x,intermediate = True)
            break
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)  

        # for f in features:
        #     print(f.size())
        self.loss_module = LossNet(features).to(self.device) 
        optim_module = optim.SGD(self.loss_module.parameters(), lr=self.args.lr,
                                 momentum=MOMENTUM, weight_decay=WDECAY)
        optimizers = {'backbone': optim_backbone, 'module': optim_module}

        
        

        epoch = 0
        accCurrent = 0.
        best_acc = 0
        best_test_f1 = 0.
        while epoch < n_epoch:
            # ts = time.time()
            # schedulers['backbone'].step()
            # need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
            # need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
            accCurrent,accLoss = self.ll_train(epoch, loader_tr, optimizers, criterion)
            test_acc= self.predict(self.X_te, self.Y_te)
            test_f1 = f1_score(self.Y_te, self.get_prediction(self.X_te,self.Y_te).cpu(),average="macro")
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                print(epoch,": ", test_f1)
                self.save_model(qs_dict=qs_dict)
            # if test_acc > best_acc:
            #     best_acc = test_acc
            #     print(epoch,": ", best_acc)
            #     self.save_model(qs_dict=qs_dict)
            # recorder.update(epoch, accLoss, accCurrent, 0, test_acc)
            epoch += 1
            current_learning_rate, _ = adjust_learning_rate(optimizers['backbone'], epoch, self.args.gammas, self.args.schedule, self.args)
            adjust_learning_rate(optimizers['module'], epoch, self.args.gammas, self.args.schedule, self.args)

                
        # if self.args.save_model:
        #     self.save_model()

    def get_uncertainty(self,models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).to(self.device) 

        with torch.no_grad():
            for (inputs, labels,idx) in unlabeled_loader:
                inputs = inputs.to(self.device) 
                # labels = labels.to(self.device) 
                scores, e1, features = models['backbone'](inputs,intermediate = True)
                pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        return uncertainty.cpu()

    def get_uncertainty_probs(self,models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).to(self.device) 

        with torch.no_grad():
            for (inputs, labels,idx) in unlabeled_loader:
                inputs = inputs.to(self.device) 
                labels = labels.to(self.device) 
                scores, e1, features = models['backbone'](inputs,intermediate = True)

                uncertainty = torch.cat((uncertainty, scores), 0)

        return uncertainty.cpu()

    def get_loss(self, X_tr, Y_tr):

        # b = np.zeros((Y_tr.size, Y_tr.max() + 1))
        # b[np.arange(Y_tr.size), Y_tr] = 1

        # print("Y_tr:" Y_tr[:30])
        # print("one hot:", b[:30])

        unlabeled_loader = DataLoader(
            self.handler(X_tr, torch.Tensor(Y_tr.numpy()).long(),
                         transform=self.args.transform_te), shuffle=False, 
            **self.args.loader_tr_args)
        models = {'backbone': self.clf, 'module': self.loss_module}
        preds = self.get_uncertainty_probs(models, unlabeled_loader)

        # print(type(preds))
        # print(preds.shape)

        # print(type(Y_tr))
        # print(Y_tr.shape)

        loss = F.cross_entropy(preds, Y_tr)

        return loss

    def query(self, n, quest=None):
         
        unlabeled_loader = DataLoader(
            self.handler(self.X, torch.Tensor(self.Y.numpy()).long(),
                         transform=self.args.transform_te), shuffle=False, 
            **self.args.loader_tr_args)
        models = {'backbone': self.clf, 'module': self.loss_module}
        uncertainty = self.get_uncertainty(models, unlabeled_loader)

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # print("idxs_unlabeled: ", idxs_unlabeled[:250])
        np.random.shuffle(idxs_unlabeled)
        
        # pick random 10000
        idxs_unlabeled = idxs_unlabeled[:min(10000,len(idxs_unlabeled))]

        if quest is not None:
            
            seed = 5

            idxs = quest.query_by_segments(self, n, quest.tessellations,seed, 1000, subset=idxs_unlabeled, uncertainty=uncertainty) 
            # arg = reversed(np.argsort(uncertainty))
            inters = quest.combined[quest.combined[:,0].argsort()]
            inters = inters[idxs_unlabeled]
            inters = self.sort_informativeness(inters)
            # print("not used: ", not_used[:250,0])
            combined = quest.combined[quest.combined[:,0].argsort()]
            not_used = combined[inters[:,0].astype(int)][:,0].astype(int)
                        # print("\n\n\n\n\n")
            # arg = idxs_unlabeled[reversed(np.argsort(uncertainty[idxs_unlabeled]))]
            print("orig: ", not_used[:50])
            print("update: ", idxs[:50])

            assert len(idxs) == n

            # assert False
            return idxs, not_used[:n]
        # print("uncert:", uncertainty)
        else:
        # Index in ascending order
            arg = reversed(np.argsort(uncertainty[idxs_unlabeled]))

        # print(uncertainty[arg[:n]])
        # assert False

            return idxs_unlabeled[arg[:n]]

    def get_informativeness(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        transform = self.args.transform_tr
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), 
                                    shuffle=False,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data)
                                    generator=self.g,
                                    **self.args.loader_tr_args)                      

        # n_epoch = self.args.n_epoch']
        self.clf = self.net.apply(weight_reset).to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(self.clf.parameters(), lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        # sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        print("current:",len(self.X[idxs_train]))
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.to(self.device) ), Variable(y.to(self.device) )
            scores, e1, features = self.clf(x,intermediate = True)
            break

        unlabeled_loader = DataLoader(
            self.handler(self.X, torch.Tensor(self.Y.numpy()).long(),
                         transform=self.args.transform_te), shuffle=True,
            **self.args.loader_tr_args)

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), 
                                    shuffle=False,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    generator=self.g,
                                    **self.args.loader_tr_args)  

        # for f in features:
        #     print(f.size())
        self.loss_module = LossNet(features).to(self.device) 

        models = {'backbone': self.clf, 'module': self.loss_module}
        U = self.get_uncertainty(models, unlabeled_loader)

        # print(U[~self.idxs_lb][:20])
        # assert False

        return U
    
    def sort_informativeness(self, combine):

        return combine[list(reversed(combine[:, 4].argsort()))]


def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    "Add by YU"
    lr = args.lr
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu
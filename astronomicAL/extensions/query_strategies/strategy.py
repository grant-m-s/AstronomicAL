from joblib.externals.cloudpickle.cloudpickle import instance
import numpy as np
import random
from sklearn import preprocessing
from torch import nn
import sys, os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from .utils import adjust_learning_rate
import time
from tqdm import tqdm
from .util import AugMixDataset
from sklearn.metrics import pairwise_distances, f1_score
class Strategy:
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        self.X = X  # vector
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te

        self.idxs_lb = idxs_lb # bool type
        self.handler = handler
        self.args = dict(args)

        print(self.args)
        self.n_pool = len(idxs_lb)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net.to(self.device)
        self.clf = deepcopy(net.to(self.device))

        # for reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(self.args['seed'])

    def seed_worker(self, worker_id):
        """
        To preserve reproducibility when num_workers > 1
        """
        # https://pytorch.org/docs/stable/notes/randomness.html
        worker_seed = self.args['seed']
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device) 
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            # exit()
            optimizer.zero_grad()

            out, e1 = self.clf(x)
            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())
                
            loss = F.cross_entropy(out, y)

            train_loss += loss.item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            
            loss.backward()
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            
            # if batch_idx % 10 == 0:
            #     print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr.dataset.X), train_loss

    
    def train(self, alpha=0.1, n_epoch=10, qs_dict=None, orig=None):
        self.clf =  deepcopy(self.net)
        # if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
                                                        # find_unused_parameters=True,
                                                        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr = self.args['lr'], weight_decay=5e-4, momentum=self.args['momentum'])

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        # epoch_time = AverageMeter()
        # recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        best_test_acc = -0.1
        best_test_f1 = -0.1

        last_update = 0
        
        if idxs_train.shape[0] != 0:
            transform = self.args["transform_tr"]

            train_data = self.handler(self.X[idxs_train], 
                                torch.Tensor(self.Y[idxs_train]).long() if type(self.Y) is np.ndarray else  torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform)

            loader_tr = DataLoader(train_data, 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args["loader_tr_args"])
            for epoch in tqdm(range(n_epoch)):
                if epoch - last_update > 150:
                    print("stopping early at epoch ", epoch)
                    break
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args["gammas"], self.args["schedule"], self.args)
                
                # Display simulation time
                # need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                # need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr, optimizer)

                if "candels" in self.args['dataset']:
                    preds = self.get_prediction(self.X_te, self.Y_te)

                    test_f1 = f1_score(self.Y_te.cpu().numpy(), preds.cpu().numpy())
                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        total_1s = np.sum(preds.cpu().numpy())
                        total_0s = len(preds) - total_1s                        
                        print(epoch,": ", best_test_f1, f"{total_0s}|{total_1s}")
                        last_update = epoch
                        self.save_model(qs_dict=qs_dict, orig=orig)
                else:
                    test_acc = self.predict(self.X_te, self.Y_te)

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        print(epoch,": ", best_test_acc, orig)
                        last_update = epoch
                        self.save_model(qs_dict=qs_dict, orig=orig)

                # if self.args.save_model and test_acc > best_test_acc:
                #     best_test_acc = test_acc
                #     print(epoch,": ", best_test_acc)
                #     self.save_model()
            # recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module

        # best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc                


    def predict(self, X):
        transform=self.args["transform_te"]
        loader_te = DataLoader(self.handler(X,Y=torch.zeros((len(X),1)), transform=transform), pin_memory=True, 
                        shuffle=False, **self.args["loader_te_args"])
        P = torch.zeros(len(X)).long().to(self.device)
        
        self.clf.eval()

        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs,labels, idx) in loader_te:
                inputs = inputs.cuda()

                scores, _ =self.clf(inputs)

                pred = scores.max(1)[1]     
                P[idx] = pred
        
        return P

    def get_prediction(self, X, Y):
        transform=self.args["transform_te"]
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args["loader_te_args"])

        P = torch.zeros(len(X)).long().to(self.device)

        self.clf.eval()

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                pred = out.max(1)[1]     
                P[idxs] = pred
   
        return P

    def predict_prob(self, X, Y):
        transform = self.args["transform_te"]
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args["loader_te_args"])

        self.clf.eval()

        probs = torch.zeros([len(Y), self.args["n_class"]])
        print(probs.shape)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        transform = self.args['transform_te']
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()

        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        transform = self.args['transform_te']
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()

        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_loss(self, X_tr, Y_tr):
        preds = self.predict_prob(X_tr, Y_tr)
        loss = F.cross_entropy(preds, Y_tr)

        return loss

    def get_embedding(self, X, Y, return_probs=False):
        """ get last layer embedding from current model"""
        transform = self.args['transform_te']
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()

        if torch.is_tensor(Y):
            y_max = Y.max().cpu().item()
        else:
            y_max = np.max(Y)
        embedding = torch.zeros([len(Y), 512])
        probs = torch.zeros([len(Y), y_max+1])

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                probs[idxs] = out.cpu().float()
                embedding[idxs] = e1.data.cpu().float()
        if not return_probs:
            return embedding
        else:
            return embedding, probs


    def get_grad_embedding(self, X, Y):
        """ gradient embedding (assumes cross-entropy loss) of the last layer"""
        transform = self.args['transform_te'] 

        model = self.clf
        if isinstance(model, nn.DataParallel):
            model = model.module
        embDim = 512
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
    
    def save_model(self, qs_dict=None, orig=None):
        # save model and selected index
        # save_path = os.path.join(self.args.save_path,self.args.dataset+'_checkpoint')
        # if not os.path.isdir(save_path):
            # os.makedirs(save_path)
        # labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        # labeled_percentage = '%.1f'%float(100*labeled/len(self.X))
        # torch.save(self.clf, os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args['seed'])+'.pkl'))
        # print('save to ',os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args['seed'])+'.pkl'))
        # path = os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args['seed'])+'.npy')
        if qs_dict is not None:

            path = f"{qs_dict['Dataset']}/{qs_dict['Embedding']}/{qs_dict['Tessellations']}/{self.args['nQuery']}/{qs_dict['QS']}_{qs_dict['method']}_{qs_dict['sub']}_{len(self.X[self.idxs_lb])}_{self.args['seed']}"
            if orig is not None:
                path = path + f"_{orig}"
        
        else:
            path = f"{qs_dict['Dataset']}/{qs_dict['Embedding']}/{self.args['nQuery']}_{self.args['nStart']}_{len(self.X[self.idxs_lb])}_{self.args['seed']}"
        
        print("saving to: ", path)
        torch.save(self.clf.state_dict(),"models/"+path)


    def load_model(self, net, qs_dict=None, orig=None):

        best_model = self.get_best_model(net, qs_dict=qs_dict, orig=orig)

        self.clf = best_model


    def get_best_model(self, net, qs_dict=None, orig=None):
        if qs_dict is not None:
            path = f"{qs_dict['Dataset']}/{qs_dict['Embedding']}/{qs_dict['Tessellations']}/{self.args['nQuery']}/{qs_dict['QS']}_{qs_dict['method']}_{qs_dict['sub']}_{len(self.X[self.idxs_lb])}_{self.args['seed']}"
            if orig is not None:
                path = path + f"_{orig}"

            
            print(path)
                
        
        else:
            path = f"{qs_dict['Dataset']}/{qs_dict['Embedding']}/{self.args['nQuery']}_{self.args['nStart']}_{len(self.X[self.idxs_lb])}_{self.args['seed']}"
        
        print("loading from: ", path)
        try:        
            net.load_state_dict(torch.load("models/"+path))
        except:
            from collections import OrderedDict
            state_dict = torch.load("models/"+path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            net.load_state_dict(new_state_dict)


        return deepcopy(net.to(self.device))

    def replace_model(self, net):
        self.clf = deepcopy(net.to(self.device))

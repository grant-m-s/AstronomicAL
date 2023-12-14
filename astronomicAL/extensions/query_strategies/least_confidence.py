import numpy as np
from .strategy import Strategy
import pdb
import torch

class LeastConfidence(Strategy):
    def __init__(self, X, Y,  X_te, Y_te, idxs_lb, net, handler, args):
        super(LeastConfidence, self).__init__(X, Y, X_te, Y_te,  idxs_lb, net, handler, args)

    def query(self, n, quest=None):
        probs = self.predict_prob(self.X, np.asarray(self.Y))
        print("probs: ", probs.shape)
        U = probs.max(1)[0]
        print("U: ", U.shape)

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        print("idxs_unlabeled: ", idxs_unlabeled.shape)


        if quest is not None:
            print("still quest")
            # assert False
            seed = 5

            idxs = quest.query_by_segments(self, n, quest.tessellations,seed, 1000, uncertainty=U)

            print("\n\n",)
            print("orig: ", idxs_unlabeled[U[idxs_unlabeled].sort()[1][:50]])
            print("update: ", idxs[:50])

            assert idxs_unlabeled[U[idxs_unlabeled].sort()[1][0]] == idxs[0], f"Not the same!"
            print("\n\n")
            
            # assert False
            return idxs, idxs_unlabeled[U[idxs_unlabeled].sort()[1][:n]]
        else:
            print("U[idxs_unlabeled]: ", U[idxs_unlabeled].shape)
            print("U[idxs_unlabeled].sort()[1]: ", U[idxs_unlabeled].sort()[1].shape)
            print(n)
            print("U[idxs_unlabeled].sort()[1][:n]: ", U[idxs_unlabeled].sort()[1][:n].shape)





            return idxs_unlabeled[U[idxs_unlabeled].sort()[1][:n]]
    
    # def get_informativeness(self):
    #     idxs_unlabeled = np.arange(self.n_pool)
    #     probs_all = self.predict_prob(self.X, np.asarray(self.Y))
    #     probs = probs_all[~self.idxs_lb]
    #     U = probs.max(1)[0]

    #     informativeness = U.sort()

    #     return informativeness[0], idxs_unlabeled[U.sort()[1]]


    def get_informativeness(self):
        # idxs_unlabeled = np.arange(self.n_pool)
        print(self.X.shape)
        print(self.Y.shape)

        probs = self.predict_prob(self.X, np.asarray(self.Y))


        # probs = probs_all[~self.idxs_lb]
        U = probs.max(1)[0]

        # informativeness = U.sort()

        return U
    
    def sort_informativeness(self, combine):

        return combine[combine[:, 4].argsort()]
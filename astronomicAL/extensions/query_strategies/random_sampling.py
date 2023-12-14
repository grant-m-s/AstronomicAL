import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(RandomSampling, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def query(self, n, quest=None):
        inds = np.arange(len(self.X))
        inds_n = inds[~self.idxs_lb]

        chosen = inds[np.random.permutation(len(self.X))]


        not_used = chosen[~self.idxs_lb]
        not_used = inds_n[not_used.argsort()]

        if quest is not None:
            seed = 5

            idxs = quest.query_by_segments(self, n, quest.tessellations, seed, uncertainty=chosen)
            print("\n\n",)
            print("orig: ", not_used[:50])
            print("update: ", idxs[:50])

            assert not_used[0] == idxs[0], f"Not the same! - {not_used[:5]} vs {idxs[:5]}"

            return idxs, not_used[:n]

        else:
    
            return not_used[:n]

    def get_informativeness(self):

        idxs_unlabeled = np.arange(self.n_pool)
        chosen = idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))]

        return chosen

    def sort_informativeness(self, combine):

        print("sort informativeness")

        return combine[combine[:,4].astype(int).argsort()]
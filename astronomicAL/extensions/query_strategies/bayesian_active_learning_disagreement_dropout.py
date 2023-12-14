import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
	def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args, n_drop=10):
		super(BALDDropout, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
		self.n_drop = n_drop

	def query(self, n, quest=None):

		idxs_all = np.arange(self.n_pool)

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(
						self.X, self.Y.numpy(),  self.n_drop)

		# print(probs.shape)
		# assert False
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1

		assert len(U) == len(self.X)
	
		idxs_all = idxs_all[~self.idxs_lb]
		chosen_lb = -U[~self.idxs_lb].cpu().numpy()

		print(idxs_all.shape)
		print(chosen_lb.shape)



		cb = np.array([idxs_all,chosen_lb]).T

		cb = cb[cb[:,1].argsort()][:,0]
		# print("not sorted U ", not_used[:10])
		# print("\n\n")
		# print("sorted U ", not_used.sort()[:10])

		# print(not__used.argsort())

		if quest is not None:
			seed = 5

			idxs = quest.query_by_segments(self, n, quest.tessellations, seed, uncertainty=U)

			print("\n\n",)
			print("orig: ", cb[:50])
			print("update: ", idxs[:50])


			assert cb[0] == idxs[0], f"Not the same! - {cb[:5]} vs {idxs[:5]}"
			print("\n\n")

			return idxs.astype(int), cb[:n].astype(int)
		else:
			return cb[:n].astype(int)

	def get_informativeness(self):
		probs = self.predict_prob_dropout_split(
						self.X, self.Y.numpy(),  self.n_drop)
		# print(probs.shape)
		# assert False
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1

		# print("U shape", U.shape)

		return U
	
	def sort_informativeness(self, combine):

		# print("sort info")
		# print(combine[:20])

		# print(combine[(-combine[:, 4]).argsort()][:20])

		return combine[(-combine[:, 4]).argsort()]


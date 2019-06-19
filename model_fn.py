import torch
from torch import nn
import torch.distributions
from tqdm import tqdm
import os
import numpy as np


class WNTM_pLSA:
    def __init__(self,
                 n_topics,
                 vocab_size,
                 doc_count,
                 context_size,
                 batch_size,
                 batch_steps,
                 num_collection_passes,
                 num_documents_passes,
                 device,
                 dtype,
                 phi_smooth_sparse_tau=.0,
                 theta_smooth_sparse_tau=.0,
                 vocab_stat=None,
                 mode='v1',
                 dump_phi_freq=None,
                 dump_phi_path=None
                 ):
        """

        :param n_topics:
        :param vocab_size:
        :param doc_count:
        :param context_size:
        :param batch_size:
        :param batch_steps:
        :param num_collection_passes:
        :param num_documents_passes:
        :param device:
        :param dtype:
        :param phi_smooth_sparse_tau:
        :param theta_smooth_sparse_tau:
        :param vocab_stat:
        :param mode: v1/v2; v1 - e-step for all batches, m-step after all
                            v2 - em-step on each batch
        """
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.doc_count = doc_count
        self.context_size = context_size
        self.num_collection_passes = num_collection_passes
        self.num_documents_passes = num_documents_passes

        self.phi_smooth_sparse_tau = phi_smooth_sparse_tau
        self.theta_smooth_sparse_tau = theta_smooth_sparse_tau

        self.batch_size = batch_size
        self.batch_steps = batch_steps
        self.device = device
        self.dtype = dtype

        self.__init_matrices()
        self.__init_aux_vars()
        if self.phi_smooth_sparse_tau is not None and\
                self.theta_smooth_sparse_tau is not None:
            self.vocab_stat = vocab_stat
            self.__init_regularizers()

        self.phi_log = []

        if mode == 'v1':
            self.run = self.run
            self.__rectify = self.__rectify
        elif mode == 'v2':
            self.run = self.run_v2
            self.__rectify = self.__rectify_v2
        else:
            raise NotImplementedError

        if dump_phi_freq:
            self.dump_phi_freq = dump_phi_freq
            self.dump_phi_path = dump_phi_path
        else:
            self.dump_phi_freq = None

        self.steps_trained = 0

    def __init_matrices(self):
        with torch.cuda.device(self.device):
            dist = torch.distributions.Normal(loc=torch.Tensor([0.5]),
                                              scale=torch.Tensor([0.5]))  # 0.5

            # dist = torch.distributions.Uniform(0, 1)
            # dist = torch.distributions.Bernoulli(torch.tensor([0.3]))
            self.phi = dist.sample((self.vocab_size, self.n_topics))
            self.phi = self.phi.view((self.vocab_size, self.n_topics))
            self.theta = dist.sample((self.n_topics, self.doc_count))
            self.theta = self.theta.view((self.n_topics, self.doc_count))

            # self.phi = torch.randn(
            #     size=(self.vocab_size, self.n_topics),
            #     device=self.device,
            #     dtype=self.dtype,
            #     requires_grad=False
            # )
            # self.theta = torch.randn(
            #     size=(self.n_topics, self.doc_count),
            #     device='cuda:1',
            #     dtype=self.dtype,
            #     requires_grad=False
            # )

            # default
            # self.phi = torch.where(self.phi <= torch.zeros_like(self.phi),
            #                        torch.zeros_like(self.phi), self.phi)
            # self.phi = torch.where(self.phi >= torch.ones_like(self.phi),
            #                        torch.ones_like(self.phi) - 0.001, self.phi)
            # self.theta = torch.where(self.theta <= torch.zeros_like(self.theta),
            #                          torch.zeros_like(self.theta), self.theta)
            # self.theta = torch.where(self.theta >= torch.ones_like(self.theta),
            #                          torch.ones_like(self.theta) - 0.001,
            #                          self.theta)

            #
            cpu_device = 'cpu:0'
            self.zero = torch.tensor(.0, dtype=self.dtype, device=cpu_device)
            self.one = torch.tensor(.1, dtype=self.dtype, device=cpu_device)
            self.phi = torch.where(self.phi < self.zero,
                                   self.zero, self.phi)
            self.phi = torch.where(self.phi > self.one,
                                   self.one - 0.001, self.phi)
            self.theta = torch.where(self.theta < self.zero,
                                     self.zero, self.theta)
            self.theta = torch.where(self.theta > self.one,
                                     self.one - 0.001,
                                     self.theta)
            self.zero = self.zero.cuda(self.device)
            self.one = self.one.cuda(self.device)

            # it works but with warning
            # self.phi = torch.tensor(self.phi, device=self.device,
            #                         dtype=self.dtype)
            self.phi = self.phi.cuda(self.device)
            # self.theta = torch.tensor(self.theta, device=self.device,
            #                           dtype=self.dtype)
            self.theta = self.theta.cuda(self.device)

            # doesn't work
            # self.theta.to(self.device)
            # self.phi.clone().detach()
            # self.theta.clone().detach()
            # self.phi.cuda()
            # self.theta.cuda(self.device)

            self.context_size = torch.LongTensor([self.context_size]).to(
                self.device)

            self.unk_inx = torch.tensor([self.vocab_size-1],
                                        dtype=self.dtype,
                                        device=self.device)

    def __init_aux_vars(self):
        self.n_wt = torch.zeros((self.vocab_size, self.n_topics),
                                device=self.device, dtype=self.dtype)
        self.n_td = torch.zeros((self.n_topics, self.doc_count),
                                device=self.device, dtype=self.dtype)
        self.n_t = torch.zeros(self.n_topics, device=self.device,
                               dtype=self.dtype)
        self.n_d = torch.zeros(self.doc_count, dtype=self.dtype,
                               device=self.device)

    def __init_regularizers(self):
        self.beta_w = torch.tensor(self.vocab_stat, device=self.device,
                                   dtype=self.dtype)\
            .view(-1, 1)
        # self.beta_w = torch.tensor([1/self.vocab_size]*self.vocab_size,
        #                            device=self.device,
        #                            dtype=self.dtype)\
        #     .view(-1, 1)
        # self.alpa_t = torch.tensor([1/self.n_topics]*self.n_topics,
        #                            device=self.device, dtype=self.dtype)\
        #     .view(-1, 1)

        self.alpa_t = torch.tensor(1/self.n_topics,
                                   device=self.device, dtype=self.dtype)\
            .view(-1, 1)

    def __dump_phi(self, epoch):
        print(f'Dump phi: epoch — {epoch}')
        if not os.path.exists(self.dump_phi_path):
            os.mkdir(self.dump_phi_path)
        savepath = os.path.join(self.dump_phi_path, f"epoch{epoch}.npy")
        np.save(savepath, self.get_phi())

    def run(self, batch_generator):
        for i in tqdm(range(self.num_collection_passes),
                      total=self.num_collection_passes,
                      desc='Passing through collection: '):
            # for n_dw, doc_inxs, batch in \
            #         tqdm(batch_generator,
            #              total=self.batch_steps,
            #              desc='Batch generator: ',
            #              leave=False):
            for n_dw, doc_inxs, batch, context_len in batch_generator:
                for _ in range(self.num_documents_passes):
                    self.e_step(n_dw, doc_inxs, batch, context_len)

            # torch.max(self.theta), torch.min(self.theta), torch.max(
            #     self.phi), torch.min(self.phi), \
            # torch.isnan(self.theta.cpu()).sum(), self.theta.shape

            if self.phi_smooth_sparse_tau is not None and\
                    self.theta_smooth_sparse_tau is not None:
                self.m_step_smoothed_sparsed()
            else:
                self.m_step()
            self.__init_aux_vars()

            self.steps_trained += 1
            print(f'Phi norm: {self.phi_log[self.steps_trained-1]}; '
                  f'step: {self.steps_trained}')

            if self.dump_phi_freq and \
                    (self.steps_trained) % self.dump_phi_freq == 0:
                self.__dump_phi(self.steps_trained)

            # phi_log: [array(2968932.5, dtype=float32), array(1.5804731e-05, dtype=float32), array(5.1386894e-05, dtype=float32), array(0.00021857, dtype=float32), array(0.00107529, dtype=float32)]

    def run_v2(self, batch_generator):
        for _ in tqdm(range(self.num_collection_passes),
                      total=self.num_collection_passes,
                      desc='Passing through collection: '):
            old_phi = self.phi.cpu()
            for n_dw, doc_inxs, batch, context_len in batch_generator:
                for _ in range(self.num_documents_passes):
                    self.em_step(n_dw, doc_inxs, batch, context_len)

            # self.theta = self.__rectify(self.theta)
            assert not np.any(torch.sum(torch.isnan(self.phi.cpu())).numpy() > 0)

            phi_norm = \
                torch.sum((self.phi.cpu().float() - old_phi.float()) ** 2)
            self.phi_log.append(phi_norm.cpu().numpy())
            self.__init_aux_vars()

            self.steps_trained += 1
            print(f'Phi norm: {self.phi_log[self.steps_trained-1]}; '
                  f'step: {self.steps_trained}')

            if self.dump_phi_freq and \
                    (self.steps_trained) % self.dump_phi_freq == 0:
                self.__dump_phi(self.steps_trained)

    def em_step(self, n_dw, doc_inxs, batch, context_len):
        self.e_step(n_dw, doc_inxs, batch, context_len)
        if self.phi_smooth_sparse_tau is not None and \
                self.theta_smooth_sparse_tau is not None:
            self.m_step_smoothed_sparsed_v2()
        else:
            self.m_step()

    def e_step_naive(self, n_dw, doc_inxs, batch):
        """
        :param n_dw: freq of term 'w' occurrence in doc 'd'
                     [[1, 1, 2, 1, 2] - for each word in a doc, ...] —
                     [batch_size, context_size]
        :param doc_inxs: Tensor of doc inxs with shape [batch_size]
        :param batch: Tensor of word inxs with shape [batch_size, context_size]
        :return:
        """

        # phi_theta = self.phi.mm(self.theta)

        batch_size = batch.shape[0]
        # E-step
        # for i in range(batch_size):
        for i in torch.arange(0, self.batch_size):
            d = doc_inxs[i]
            doc = batch[i]
            for w in doc:  # w -- term inx
                w = w.int()
                # n_dw = torch.LongTensor([1.0]).float()
                phi_w = self.phi[w, :]
                theta_d = self.theta[:, d]
                # numerator = n_dw * phi_w * theta_d
                numerator = phi_w * theta_d
                denominator = torch.sum(self.phi[w, :] * self.theta[:, d])  # phi_theta[w, d]
                n_tdw = numerator / denominator

                self.n_wt[w, :] += n_tdw
                self.n_td[:, d] += n_tdw
                self.n_t += n_tdw

    def e_step(self, n_dw, doc_inxs, context_batch, context_len):
        """
        :param n_dw: freq of term 'w' occurrence in doc 'd'
                     [[1, 1, 2, 1, 2] - for each word in a doc, ...] —
                     [batch_size, context_size]
        :param doc_inxs: Tensor of doc inxs with shape [batch_size]
        :param context_batch: Tensor of word inxs with shape [batch_size, context_size]
        :return:
        """
        with torch.cuda.device(self.device):
            # phi_theta = self.phi.mm(self.theta)

            context_batch = context_batch.cuda(self.device)
            batch_size = context_batch.shape[0]
            context_size = context_batch.shape[1]
            # E-step
            # [batch_size, context_size, n_topics]
            phi_w = self.phi[context_batch.long()]  # .to(self.device)
            # [batch_size, n_topics]
            theta_d = torch.t(self.theta)[doc_inxs.long()]  # .to(self.device)
            # theta_d = theta_d.view(batch_size, -1, 1)

            # To zeroize <UNK> token:
            mask = context_batch == self.unk_inx

            phi_w = torch.masked_fill(phi_w,
                                      mask.view(batch_size, context_size, -1),
                                      self.zero)

            # [1, batch_size, context_size * n_topics]
            theta_d = theta_d.repeat(1, 1, context_size)
            # [batch_size, context_size, n_topics]
            theta_d = theta_d.view(batch_size, context_size, -1)

            # [batch_size, context_size, n_topics]
            numerator = phi_w * theta_d
            # [batch_size, context_size, context_size]
            # denominator = torch.bmm(phi_w, torch.transpose(theta_d, 1, 2))
            # [batch_size, context_size, 1]
            denominator = torch.sum(phi_w * theta_d, dim=2, keepdim=True)
            # denominator = denominator
            # [batch_size, context_size, n_topics]
            n_tdw = numerator / denominator  # n_dw * numerator / denominator
            # n_tdw[torch.isnan(n_tdw)] = self.zero
            n_tdw = torch.where(n_tdw != n_tdw, self.zero, n_tdw)

            # [batch_size*context_size]
            context_1d_mask = context_batch.view(-1)
            # [batch_size*context_size, n_topics]
            n_tdw_context = n_tdw.view(-1, self.n_topics)
            # [batch_size, n_topics]
            n_tdw_doc = torch.sum(n_tdw, dim=1, keepdim=False)
            # [n_topics]
            n_tdw_t = n_tdw.sum(1).sum(0)
            n_tdw_d = n_tdw.sum(2).sum(1)

            # self.n_wt[context_1d_mask.long()] += n_tdw_context  #doesn't work
            # too slow; doesn't work too
            # for i in torch.arange(batch_size*context_size):
            #     self.n_wt[context_1d_mask.long()[i]] += n_tdw_context[i]
            # torch.unique(context_1d_mask.long(), return_counts=True)
            # [unique_count, max_unique_count, n_topics]
            wt_index = context_1d_mask.long().cuda(self.device)
            n_wt_update, wt_index = self._group_by_with_index_mapping(
                wt_index, n_tdw_context)
            self.n_wt[wt_index] += n_wt_update

            self.n_td[:, doc_inxs.long()] += n_tdw_doc.t()  # t_() - inplace t
            self.n_t += n_tdw_t
            self.n_d[doc_inxs.long()] += n_tdw_d

    # @staticmethod
    def _group_by_with_index_mapping(self, true_labels, samples):
        """
        TODO: read Notes of reproducibility
        :param true_labels: indices for initial embedding matrix
            [100, 100, 200, 200, 0] =>
            [0, 100, 200], [1, 1, 2, 2, 0], [1, 2, 2]
        :param samples: 2D-tensor with vectors to agg(sum)
            [[0.1, .0], [-0.1, 0.2], [...], [...], [...]]
        :return: agg(sum): [[...], [.0, 0.1], [...]],
                    index: [0, 100, 200]
        """
        with torch.cuda.device(self.device):
            true_unique_labels, ordering_index = true_labels.unique(dim=0,
                return_counts=False, return_inverse=True)
            ordering_labels = ordering_index.view(ordering_index.size(0), 1)\
                .expand(-1, samples.size(1))
            ordering_unique_labels, ordering_count =\
                ordering_labels.unique(dim=0, return_counts=True,
                                       return_inverse=False)
            grouped_res = torch.zeros_like(ordering_unique_labels,
                                           dtype=torch.float,
                                           device=self.device)
            # grouped_res = grouped_res.cuda(self.device)
            grouped_res = grouped_res.scatter_add_(0,
                ordering_labels.cuda(self.device), samples.cuda(self.device))
            grouped_res = grouped_res / \
                          ordering_count.float().cuda().unsqueeze(1)
            return grouped_res, true_unique_labels

    def m_step(self):
        # M-step
        # for w in torch.arange(0, self.vocab_size):
        #     self.phi[w] = self.n_wt[w] / self.n_t
        #
        # for d in torch.arange(0, self.doc_count):
        #     self.theta[:, d] = self.n_td[:, d] / self.context_size
        with torch.cuda.device(self.device):
            new_phi = self.n_wt / self.n_t.view(-1, self.n_topics)
            # phi_norm = torch.sum((self.phi - new_phi.to('cpu'))**2)
            phi_norm = torch.sum((self.phi - new_phi)**2)
            self.phi_log.append(phi_norm.cpu().numpy())
            self.phi = new_phi
            # self.theta = self.n_td / self.context_size.float().to(self.device)
            self.theta = self.n_td / self.n_d.view(-1, self.doc_count)

    def __rectify(self, t):
        # return torch.where(t < torch.zeros_like(t), torch.zeros_like(t), t)

        # r, c = t.shape
        # # TODO: refactor these magic numbers
        # if r * c > 1.5e+06*400:
        #     t = t.cpu()
        #     t = torch.where(t < self.zero.cpu(), self.zero.cpu(), t)
        #     # t[torch.isnan(t)] = self.zero.cpu()
        #     t = torch.where(t != t, self.zero.cpu(), t)
        #     return t.cuda()
        # else:
        #     t = torch.where(t < self.zero, self.zero, t)
        #     t = torch.where(t != t, self.zero, t)
        # return t

        t = t.cpu()
        t = torch.where(t < self.zero.cpu(), self.zero.cpu(), t)
        t[torch.isnan(t)] = self.zero.cpu()
        # t = torch.where(t != t, self.zero.cpu(), t)
        return t.cuda()


    # @staticmethod
    def __rectify_v2(self, t):
        t = torch.where(t < self.zero, self.zero, t)
        t = torch.where(t != t, self.zero, t)
        return t

    def m_step_smoothed_sparsed(self):
        with torch.cuda.device(self.device):
            alpha = torch.tensor([self.theta_smooth_sparse_tau],
                                 device=self.device, dtype=self.dtype)
            beta = torch.tensor([self.phi_smooth_sparse_tau],
                                device=self.device, dtype=self.dtype)
            old_phi = self.phi.cpu()
            # phi_norm = torch.sum((self.phi - new_phi.to('cpu'))**2)
            # self.phi = (self.n_wt + beta)

            # default
            self.phi = (self.n_wt + beta*self.beta_w)

            # beta = torch.tensor(-.1, device=self.device, dtype=self.dtype)
            # beta_reg = beta*((self.phi**2).mean(dim=1, keepdim=True)**1/2)
            # # beta_reg = beta*self.phi.mean(dim=1, keepdim=True)
            # beta_reg[torch.isnan(beta_reg)] = self.zero
            # self.phi = (self.n_wt + beta_reg)

            # self.phi = (self.n_wt / self.n_t.view(-1, self.n_topics) + beta)
            self.phi /= torch.sum(self.phi, dim=0, keepdim=True)
            self.phi = self.__rectify(self.phi)
            # self.phi[-1] = torch.zeros(size=(self.n_topics,), dtype=self.dtype,
            #                            device=self.device)  # -> nan
            # self.theta = (self.n_td + alpha) /\
            #             torch.sum(self.n_td + alpha, dim=0, keepdim=True)\
            # self.theta = (self.n_td + alpha)

            # default
            self.theta = (self.n_td + alpha*self.alpa_t)  # doesn't work!

            # alpha = torch.tensor(-.01, device=self.device, dtype=self.dtype)
            # alpha_reg = alpha/((self.theta**2).mean(1, keepdim=True)**(1/2))
            # alpha_reg = alpha*self.theta.mean(1, keepdim=True)
            # alpha_reg[torch.isnan(alpha_reg)] = self.zero
            # self.theta = (self.n_td + alpha_reg)

            # self.theta = (self.n_td + alpha/self.theta.sum(dim=1, keepdim=True))
            # self.theta = (self.n_td / self.context_size.float().to(self.device)
            #               + alpha)

            # to avoid nans
            # self.theta = self.theta.cpu()
            # self.theta[torch.isnan(self.theta)] = .0
            # self.theta = self.theta.cuda(self.device)

            # self.theta =
            self.theta /= torch.sum(self.theta, dim=0, keepdim=True)
            self.theta = self.__rectify(self.theta)

            phi_norm = \
                torch.sum((self.phi.cpu().float() - old_phi.float()) ** 2)
            self.phi_log.append(phi_norm.cpu().numpy())

    def m_step_smoothed_sparsed_v2(self):
        with torch.cuda.device(self.device):
            alpha = torch.tensor([self.theta_smooth_sparse_tau],
                                 device=self.device, dtype=self.dtype)
            beta = torch.tensor([self.phi_smooth_sparse_tau],
                                device=self.device, dtype=self.dtype)
            # phi_norm = torch.sum((self.phi - new_phi.to('cpu'))**2)
            # self.phi = (self.n_wt + beta)
            self.phi = (self.n_wt + beta*self.beta_w)
            # self.phi = (self.n_wt + beta/self.phi.sum(dim=1, keepdim=True))
            # self.phi = (self.n_wt / self.n_t.view(-1, self.n_topics) + beta)
            self.phi /= torch.sum(self.phi, dim=0, keepdim=True)
            self.phi = self.__rectify(self.phi)
            # self.theta = (self.n_td + alpha) /\
            #             torch.sum(self.n_td + alpha, dim=0, keepdim=True)\
            # self.theta = (self.n_td + alpha)
            self.theta = (self.n_td + alpha*self.alpa_t)
            # self.theta = (self.n_td + alpha/self.theta.sum(dim=1, keepdim=True))
            # self.theta = (self.n_td / self.context_size.float().to(self.device)
            #               + alpha)
            self.theta /= torch.sum(self.theta, dim=0, keepdim=True)
            self.theta = self.__rectify(self.theta)

    def get_phi(self):
        return self.phi.cpu().numpy()


if __name__ == '__main__':
    print(torch.__version__)

    import torch.distributions
    g = torch.distributions.Normal(loc=torch.Tensor([0.5]),
                                   scale=torch.Tensor([0.25]))
    s = g.sample((2, 3))
    print(s)

    # print(s - torch.Tensor([100]))
    print(s[0, 1])
    print(s[torch.LongTensor([.0]), torch.LongTensor([1.])])

    for d in range(torch.LongTensor([5])):
        print(d)

    for d in torch.arange(0, torch.LongTensor([5])):
        print(d)
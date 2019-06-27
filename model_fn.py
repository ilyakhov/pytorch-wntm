import torch
from torch import nn
import torch.distributions
from tqdm import tqdm
import os
import numpy as np
import logging

GPU_FLOAT32_LIMIT = 1.5e+06*400


class pLSA:
    def __init__(self,
                 n_topics,
                 vocab_size,
                 doc_count,
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
                 dump_phi_path=None,
                 log_perplexity=False,
                 log_matrix_norms=False
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
        :param vocab_stat: TF for phi sparse/smooth reg.
        :param mode: v1/v2; v1 - e-step for all batches, m-step after all
                            v2 - em-step on each batch
        """
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.doc_count = doc_count
        self.num_collection_passes = num_collection_passes
        self.num_documents_passes = num_documents_passes

        self.phi_smooth_sparse_tau = phi_smooth_sparse_tau
        self.theta_smooth_sparse_tau = theta_smooth_sparse_tau

        self.batch_size = batch_size
        self.batch_steps = batch_steps
        self.device = device
        self.dtype = dtype

        self.log_perplexity = log_perplexity
        self.log_matrix_norms = log_matrix_norms

        self.__init_matrices()
        self.__init_aux_vars()
        if self.phi_smooth_sparse_tau is not None and\
                self.theta_smooth_sparse_tau is not None:
            self.vocab_stat = vocab_stat
        self.__init_regularizers()

        self.phi_log = []
        self.theta_log = []
        self.perplexity_log = []

        self.run = self.run
        self.__rectify = self.__rectify

        if dump_phi_freq:
            self.dump_phi_freq = dump_phi_freq
            self.dump_phi_path = dump_phi_path
        else:
            self.dump_phi_freq = None

        self.steps_trained = 0

        if mode == 'v1':
            self.run = self.run
            self.__rectify = self.__rectify
        elif mode == 'v2':
            self.run = self.run_v2
            self.__rectify = self.__rectify_v2
        else:
            raise NotImplementedError

    def __init_matrices(self):
        with torch.cuda.device(self.device):
            # _dist = 'normal'
            # loc = 0.5
            # scale = 0.1
            # logging.info(f'{_dist}: loc — {loc}; scale — {scale}')
            #
            # dist = torch.distributions.Normal(loc=torch.Tensor([loc]),
            #                                   scale=torch.Tensor([scale]))
            # 0.5

            _dist = 'uniform: [0,1)'
            dist = torch.distributions.Uniform(0, 1)
            logging.info(f'{_dist}')

            self.phi = dist.sample((self.vocab_size, self.n_topics))
            self.phi = self.phi.view((self.vocab_size, self.n_topics))
            self.theta = dist.sample((self.n_topics, self.doc_count))
            self.theta = self.theta.view((self.n_topics, self.doc_count))

            cpu_device = 'cpu:0'
            self.zero = torch.tensor(.0, dtype=self.dtype, device=cpu_device)
            self.one = torch.tensor(1., dtype=self.dtype, device=cpu_device)
            self.phi = torch.where(self.phi < self.zero,
                                   self.zero, self.phi)
            self.phi = torch.where(self.phi > self.one,
                                   self.one - 0.001, self.phi)
            self.theta = torch.where(self.theta < self.zero,
                                     self.zero, self.theta)
            self.theta = torch.where(self.theta > self.one,
                                     self.one - 0.001,
                                     self.theta)

            # norm
            self.phi /= self.phi.sum(dim=0, keepdim=True)
            self.theta /= self.theta.sum(dim=0, keepdim=True)

            # drop nans
            # self.phi = torch.where(self.phi != self.phi, self.zero, self.phi)
            # self.theta = torch.where(self.theta != self.theta, self.zero,
            #                          self.theta)
            # assert torch.sum(torch.isnan(self.phi)) == 0
            # assert torch.sum(torch.isnan(self.theta)) == 0

            assert not np.any(
                torch.sum(torch.isnan(self.phi)).numpy() > 0)
            assert not np.any(
                torch.sum(torch.isnan(self.theta)).numpy() > 0)

            self.phi = self.phi.cuda(self.device)
            self.theta = self.theta.cuda(self.device)
            self.zero = self.zero.cuda(self.device)
            self.one = self.one.cuda(self.device)

            # crook: it needs for perplexity evaluation
            # TODO: optimize it for WNTM
            if self.log_perplexity is True:
                self.n_dw = torch.zeros((self.vocab_size, self.doc_count),
                                        device='cpu:0', dtype=self.dtype)

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
        if self.vocab_stat is not None:
            # using TF of the words to sparse/smooth reg
            self.beta_w = torch.tensor(self.vocab_stat, device=self.device,
                                       dtype=self.dtype).view(-1, 1)
        else:
            # uniform beta for phi sparse reg
            self.beta_w = torch.tensor(1/self.vocab_size,
                                       device=self.device,
                                       dtype=self.dtype).view(-1, 1)

        self.alpa_t = torch.tensor(1/self.n_topics,
                                   device=self.device,
                                   dtype=self.dtype).view(-1, 1)

    def run(self, batch_generator):
        gather_ndw = True if self.log_perplexity else False
        for _ in tqdm(range(self.num_collection_passes),
                      total=self.num_collection_passes,
                      desc='Passing through collection: '):
            for n_dw, doc_inxs, batch in batch_generator:
                for _ in range(self.num_documents_passes):
                    self.e_step(n_dw, doc_inxs, batch,
                                gather_ndw=gather_ndw)
                    gather_ndw = False
            if self.log_matrix_norms is True:
                old_phi = self.phi.cpu()
                old_theta = self.theta.cpu()
            self.m_step_smoothed_sparsed()

            # phi/theta norms logging
            if self.log_matrix_norms is True:
                phi_norm = \
                    (torch.sum((self.phi.cpu().float() - old_phi.float()) ** 2)
                     ** 1/2)
                theta_norm = \
                    (torch.sum((self.theta.cpu().float() -
                                old_theta.float()) ** 2)
                     ** 1/2)
                self.phi_log.append(phi_norm.numpy())
                self.theta_log.append(theta_norm.numpy())

            # perplexity logging
            if self.log_perplexity is True:
                self.perplexity_log.append(self.perplexity())

                logging.info(f'Phi norm: {self.phi_log[self.steps_trained]}; '
                             f'step: {self.steps_trained}')
                logging.info(
                    f'Theta norm: {self.theta_log[self.steps_trained]}; '
                    f'step: {self.steps_trained}')
                logging.info(f'Perplexity: '
                             f'{self.perplexity_log[self.steps_trained]}; '
                             f'step: {self.steps_trained}')

            self.steps_trained += 1
            if self.dump_phi_freq and \
                    (self.steps_trained) % self.dump_phi_freq == 0:
                self.__dump_phi(self.steps_trained)

    def perplexity(self):
        """
        Full:
        exp(-1/n_m * sum(n_dw * ln(mm(self.phi, self.theta))))
        :return:
        """
        # TODO: optimize it for WNTM
        phi = self.phi.cpu()
        theta = self.theta.cpu()
        n_m = torch.sum(self.n_dw)

        one = torch.tensor(1, dtype=self.dtype, device='cpu:0')
        mm = torch.mm(phi, theta)
        mm = torch.where(mm == self.zero.cpu(), 1e-4*one, mm)
        mm = mm.log()
        one = -1 * one
        perp = torch.exp(one/n_m * torch.sum(self.n_dw * mm))
        return perp.numpy()

    def e_step(self, n_dw, doc_inxs, context_batch, gather_ndw=False):
        """
        :param n_dw: freq of term 'w' occurrence in doc 'd'
                     [[1, 1, 2, 1, 2] - for each word in a doc, ...] —
                     [batch_size, context_size]
        :param doc_inxs: Tensor of doc inxs with shape [batch_size]
        :param context_batch: Tensor of word inxs with shape
        [batch_size, context_size]
        :return:
        """
        with torch.cuda.device(self.device):
            # phi_theta = self.phi.mm(self.theta)
            if gather_ndw is True:
                self.n_dw[:, doc_inxs.long()] = n_dw.t()

            n_dw = torch.unsqueeze(n_dw, 2).cuda(self.device)

            context_batch = context_batch.cuda(self.device)
            batch_size = context_batch.shape[0]
            context_size = context_batch.shape[1]
            # E-step
            # [batch_size, context_size, n_topics]
            phi_w = self.phi[context_batch.long()]  # .to(self.device)
            # [batch_size, n_topics]
            theta_d = torch.t(self.theta)[doc_inxs.long()]  # .to(self.device)
            # theta_d = theta_d.view(batch_size, -1, 1)

            # [1, batch_size, context_size * n_topics]
            theta_d = theta_d.repeat(1, 1, context_size)
            # [batch_size, context_size, n_topics]
            theta_d = theta_d.view(batch_size, context_size, -1)

            # [batch_size, context_size, n_topics]
            numerator = phi_w * theta_d
            # [batch_size, context_size, context_size]
            # [batch_size, context_size, 1]
            denominator = torch.sum(phi_w * theta_d, dim=2, keepdim=True)
            # [batch_size, context_size, n_topics]
            n_tdw = n_dw * numerator / denominator
            # n_tdw = torch.where(n_tdw != n_tdw, self.zero, n_tdw)
            n_tdw[torch.isnan(n_tdw)] = self.zero

            # [batch_size*context_size, n_topics]
            n_tdw_context = n_tdw.view(-1, self.n_topics)
            # [batch_size, n_topics]
            n_tdw_doc = torch.sum(n_tdw, dim=1, keepdim=False)
            # [n_topics]
            n_tdw_t = n_tdw.sum(1).sum(0)
            n_tdw_d = n_tdw.sum(2).sum(1)

            # [batch_size*context_size]
            context_1d_mask = context_batch.view(-1)
            wt_index = context_1d_mask.long().cuda(self.device)
            n_wt_update, wt_index = self._group_by_with_index_mapping(
                wt_index, n_tdw_context)
            self.n_wt[wt_index] += n_wt_update

            self.n_td[:, doc_inxs.long()] += n_tdw_doc.t()  # t_() - inplace t
            self.n_t += n_tdw_t
            self.n_d[doc_inxs.long()] += n_tdw_d

    def _group_by_with_index_mapping(self, true_labels, samples):
        """
        TODO: implement stuff from "Notes of reproducibility"
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
            grouped_res = grouped_res.scatter_add_(0,
                ordering_labels.cuda(self.device), samples.cuda(self.device))
            grouped_res = grouped_res / \
                          ordering_count.float().cuda().unsqueeze(1)
            return grouped_res, true_unique_labels

    def __rectify(self, t):
        # If matrix has more elements then 1.5kk*400,
        # the next computations are being done on cpu
        r, c = t.shape
        if r * c > GPU_FLOAT32_LIMIT:
            t = t.cpu()
            t = torch.where(t < self.zero.cpu(), self.zero.cpu(), t)
            # filter nans: 2 ways
            # Need this in case of empty sentences!
            # t[torch.isnan(t)] = self.zero.cpu()
            # t = torch.where(t != t, self.zero.cpu(), t)

            assert not np.any(
                torch.sum(torch.isnan(t)).numpy() > 0)
            return t.cuda()
        else:
            t = torch.where(t < self.zero, self.zero, t)
            # t = torch.where(t != t, self.zero, t)
            assert not np.any(
                torch.sum(torch.isnan(t)).cpu().numpy() > 0)
            return t

    def m_step_smoothed_sparsed(self):
        with torch.cuda.device(self.device):
            alpha = torch.tensor([self.theta_smooth_sparse_tau],
                                 device=self.device, dtype=self.dtype)
            beta = torch.tensor([self.phi_smooth_sparse_tau],
                                device=self.device, dtype=self.dtype)

            self.phi = (self.n_wt + beta*self.beta_w)
            self.phi /= torch.sum(self.phi, dim=0, keepdim=True)
            self.phi = self.__rectify(self.phi)

            self.theta = (self.n_td + alpha*self.alpa_t)
            self.theta /= torch.sum(self.theta, dim=0, keepdim=True)
            self.theta = self.__rectify(self.theta)

    def get_phi(self):
        return self.phi.cpu().numpy()

    def __dump_phi(self, epoch):
        logging.info(f'Dump phi: epoch — {epoch}')
        if not os.path.exists(self.dump_phi_path):
            os.mkdir(self.dump_phi_path)
        savepath = os.path.join(self.dump_phi_path, f"epoch{epoch}.npy")
        np.save(savepath, self.get_phi())

    def m_step(self):
        """
        Rational EM. Not tested enough!
        :return:
        """
        with torch.cuda.device(self.device):
            new_phi = self.n_wt / self.n_t.view(-1, self.n_topics)
            phi_norm = (torch.sum((self.phi - new_phi)**2)**1/2)
            self.phi_log.append(phi_norm.cpu().numpy())
            self.phi = new_phi
            self.theta = self.n_td / self.n_d.view(-1, self.doc_count)

    def run_v2(self, batch_generator):
        """
        M-step after each E-step. Not tested enough!
        :param batch_generator:
        :return:
        """
        for _ in tqdm(range(self.num_collection_passes),
                      total=self.num_collection_passes,
                      desc='Passing through collection: '):
            old_phi = self.phi.cpu()
            for n_dw, doc_inxs, batch, context_len in batch_generator:
                for _ in range(self.num_documents_passes):
                    self.em_step(n_dw, doc_inxs, batch, context_len)

            # self.theta = self.__rectify(self.theta)
            assert not np.any(
                torch.sum(torch.isnan(self.phi.cpu())).numpy() > 0)

            phi_norm = \
                (torch.sum((self.phi.cpu().float() - old_phi.float()) ** 2)
                 ** 1 / 2)
            self.phi_log.append(phi_norm.cpu().numpy())
            self.__init_aux_vars()

            self.steps_trained += 1
            logging.info(f'Phi norm: {self.phi_log[self.steps_trained-1]}; '
                         f'step: {self.steps_trained}')

            if self.dump_phi_freq and \
                    (self.steps_trained) % self.dump_phi_freq == 0:
                self.__dump_phi(self.steps_trained)

    def __rectify_v2(self, t):
        """
        Rectification on each step is expensive operation if
        data are being copied on cpu. For train_mode='v2' no data copy
        to 'cpu'(RAM) has being used.
        :param t:
        :return:
        """
        t = torch.where(t < self.zero, self.zero, t)
        t = torch.where(t != t, self.zero, t)
        return t

    def em_step(self, n_dw, doc_inxs, batch, context_len):
        self.e_step(n_dw, doc_inxs, batch, context_len)
        self.m_step_smoothed_sparsed()


class WNTM_pLSA(pLSA):
    def __init__(self,
                 n_topics,
                 vocab_size,
                 doc_count,
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
                 dump_phi_path=None,
                 log_perplexity=False,
                 log_matrix_norms=False
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
        :param vocab_stat: TF for phi sparse/smooth reg.
        :param mode: v1/v2; v1 - e-step for all batches, m-step after all
                            v2 - em-step on each batch
        """
        super(WNTM_pLSA, self).__init__(
            n_topics=n_topics,
            vocab_size=vocab_size,
            doc_count=doc_count,
            batch_size=batch_size,
            batch_steps=batch_steps,
            num_collection_passes=num_collection_passes,
            num_documents_passes=num_documents_passes,
            device=device,
            dtype=dtype,
            phi_smooth_sparse_tau=phi_smooth_sparse_tau,
            theta_smooth_sparse_tau=theta_smooth_sparse_tau,
            vocab_stat=vocab_stat,
            mode=mode,
            dump_phi_freq=dump_phi_freq,
            dump_phi_path=dump_phi_path,
            log_perplexity=log_perplexity,
            log_matrix_norms=log_matrix_norms
        )
        # super(WNTM_pLSA, self).e_step = self.e_step

        self.__init_const()

    def __init_const(self):
        self.unk_inx = torch.tensor([self.vocab_size - 1],
                                    dtype=self.dtype,
                                    device=self.device)

    def e_step(self, n_dw, doc_inxs, context_batch, gather_ndw=False):
        """
        :param n_dw: freq of term 'w' occurrence in doc 'd'
                     [[1, 1, 2, 1, 2] - for each word in a doc, ...] —
                     [batch_size, context_size]
        :param doc_inxs: Tensor of doc inxs with shape [batch_size]
        :param context_batch: Tensor of word inxs with shape
        [batch_size, context_size]
        :param first: 'first' iteration over documents
                      to gather self.n_dw matrix
        :return:
        """
        with torch.cuda.device(self.device):
            if gather_ndw is True:
                self.n_dw[:, doc_inxs.long()] = n_dw.t()

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
            # [batch_size, context_size, 1]
            denominator = torch.sum(phi_w * theta_d, dim=2, keepdim=True)
            # [batch_size, context_size, n_topics]
            n_tdw = numerator / denominator
            # n_tdw = torch.where(n_tdw != n_tdw, self.zero, n_tdw)
            n_tdw[torch.isnan(n_tdw)] = self.zero

            # [batch_size*context_size, n_topics]
            n_tdw_context = n_tdw.view(-1, self.n_topics)
            # [batch_size, n_topics]
            n_tdw_doc = torch.sum(n_tdw, dim=1, keepdim=False)
            # [n_topics]
            n_tdw_t = n_tdw.sum(1).sum(0)
            n_tdw_d = n_tdw.sum(2).sum(1)

            # [batch_size*context_size]
            context_1d_mask = context_batch.view(-1)
            wt_index = context_1d_mask.long().cuda(self.device)
            n_wt_update, wt_index = self._group_by_with_index_mapping(
                wt_index, n_tdw_context)
            self.n_wt[wt_index] += n_wt_update

            self.n_td[:, doc_inxs.long()] += n_tdw_doc.t()  # t_() - inplace t
            self.n_t += n_tdw_t
            self.n_d[doc_inxs.long()] += n_tdw_d
import time
import torch
import torch.nn.functional as F
from src.data_loader import create_dataloder
from backpack import backpack
from backpack.extensions import (BatchGrad, BatchL2Grad)
from tqdm import tqdm
from src.utils import update_dataset

INF = 100000000

class NTK_active_memo:
    def __init__(self, args):
        self.args = args
        self.topk = int(args.budget_num_per_query)
        self.total_epoch = args.epochs
        self.total_budget = self.topk * args.total_query_times
        self.query_g_idx = []

    def get_batch_grad(self, pesudo_labels, model, idxs, images, nrom_sq):
        if model.device == 'cuda':
            images = images.to(model.device)
        outputs = model(images)
        labels = pesudo_labels[idxs]
        if model.MSE:
            labels = torch.nn.functional.one_hot(labels)
            loss_pool = model.compute_MSELoss(outputs, labels)
        else:
            loss_pool = model.compute_CELoss(outputs, labels)

        model.zero_grad()
        with backpack(BatchGrad(), BatchL2Grad()):
            loss_pool.backward()
            batch_grad_cache = []
            grad_square_cache = None
            for name, param in model.named_parameters():
                if 'bn' in name:
                    batch_pool_grad = param.grad.reshape(1, -1).expand(idxs.shape[0], -1) / idxs.shape[0]
                else:
                    try:
                        batch_pool_grad = param.grad_batch.reshape(param.grad_batch.shape[0], -1)
                    except:
                        print(name)
                        exit(1)

                batch_grad_cache.append(batch_pool_grad * idxs.shape[0])

                if nrom_sq:
                    if grad_square_cache is None:
                        grad_square_cache = param.batch_l2 * (idxs.shape[0]**2)
                    else:
                        if 'bn' in name:
                            grad_square_cache = grad_square_cache + torch.square(param.grad / idxs.shape[0]).sum() * (idxs.shape[0]**2)
                        else:
                            grad_square_cache = grad_square_cache + param.batch_l2 * (idxs.shape[0]**2)

            del outputs
            torch.cuda.empty_cache()
            # TODO: check if del batch_grad_cache will influence batch_grad
            batch_grad = torch.hstack(batch_grad_cache)
            del batch_grad_cache
            torch.cuda.empty_cache()
        return batch_grad, grad_square_cache



    def get_pesudo_label(self, model, pool_loader, type='max'):
        idx_tensor = []
        pesudo_labels = []
        with torch.no_grad():
            for idxs, images, pool_label in pool_loader:
                if model.device == 'cuda':
                    images = images.to(model.device)
                outputs = model(images)
                if type == 'sample':
                    sampler = torch.distributions.Categorical(F.softmax(outputs, dim=1))
                    predicted = sampler.sample()
                elif type == 'max':
                    _, predicted = torch.max(outputs.data, 1)
                elif type == 'min':
                    _, predicted = torch.min(outputs.data, 1)
                elif type == 'truth':
                    predicted = pool_label.to(model.device)
                else:
                    predicted = None
                pesudo_labels.append(predicted)
                idx_tensor.append(idxs)
        pesudo_labels = torch.cat(pesudo_labels)
        return pesudo_labels


    def get_train_grad(self, model, train_loader):
        # NOTE: don't call zero_grad in this loop. need to agg all train grad!:
        for idxs, inputs, labels in train_loader:
            if model.device == 'cuda':
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
            outputs = model(inputs)
            if model.MSE:
                labels = torch.nn.functional.one_hot(labels)
                loss_train = model.compute_MSELoss(outputs, labels)
            else:
                loss_train = model.compute_CELoss(outputs, labels)
            loss_train.backward()
        train_grad = model.collect_grad().reshape(1,-1)
        return train_grad


    def query(self, args, model, train_loader, pool_loader, train_data, pool_data):
        # 1.get pesudo label for pool_loader
        model.eval()
        pesudo_labels = self.get_pesudo_label(model, pool_loader)
        model.zero_grad()
        torch.cuda.empty_cache()

        # 2.get train grad
        temp_train_dataloader = create_dataloder(args, train_data, shuffle=False, certain_batch_size=1)
        train_grad = self.get_train_grad(model, temp_train_dataloader)

        model.zero_grad()
        torch.cuda.empty_cache()

        # 3.compute inner product between train grad and pool grad, l2-norm of pool grad
        small_pool_loader = create_dataloder(args, pool_data, shuffle=False, certain_batch_size=args.small_bz)
        sim_with_train = []
        sq_sum_of_pool = []
        t = time.time()
        for idxs, images, _ in tqdm(small_pool_loader):
            batch_grad_cache, grad_square_cache = self.get_batch_grad(pesudo_labels, model, idxs, images, nrom_sq=True)
            torch.cuda.empty_cache()
            sim_with_train.append((train_grad * batch_grad_cache).sum(dim=1, keepdim=True) )
            del batch_grad_cache
            torch.cuda.empty_cache()
            sq_sum_of_pool.append(grad_square_cache.reshape(-1,1))

        sq_sum_of_pool = torch.vstack(sq_sum_of_pool)
        sim_with_train = torch.vstack(sim_with_train)

        if args.wo_train_sim:
            self_term = (sq_sum_of_pool).reshape(-1,)
        else:
            self_term = (args.train_sim_times*sim_with_train + sq_sum_of_pool).reshape(-1,)

        print('t1:{}'.format(time.time() - t))
        model.zero_grad()

        # 4.inner product between pool grad and query example grad
        queried_idxs = []
        queried_grads = torch.zeros_like(train_grad)
        if not args.wo_de_inf:
            de_inf_grad = torch.zeros_like(train_grad)

        t= time.time()
        iter_num = self.topk//args.grad_update_size
        assert self.topk % args.grad_update_size == 0

        for i in tqdm(range(iter_num), total=iter_num):
            if args.wo_de_inf or i == 0:
                updated_self_term = self_term.clone()
            else:
                de_inf = []
                for idxs, images, _ in tqdm(small_pool_loader):
                    batch_grad_cache, _ = self.get_batch_grad(pesudo_labels, model, idxs, images, nrom_sq=False)
                    torch.cuda.empty_cache()

                    if args.norm_inf:
                        de_inf_grad = queried_grads / (i*args.grad_update_size)
                    else:
                        de_inf_grad = queried_grads

                    de_inf.append((batch_grad_cache * de_inf_grad).sum(dim=1))
                    del batch_grad_cache
                    torch.cuda.empty_cache()

                updated_self_term = self_term - torch.cat(de_inf)

            updated_self_term[queried_idxs] = -INF
            sorted_idxs = updated_self_term.argsort().cpu().numpy()
            query_idxs_in_pool = sorted_idxs[-args.grad_update_size:][::-1].copy()

            if not args.wo_de_inf:
                pool_x, pool_t, pool_idx = pool_data
                temp_data = (pool_x[query_idxs_in_pool], pool_t[query_idxs_in_pool])
                temp_dataloader = create_dataloder(args, temp_data, shuffle=False, certain_batch_size=query_idxs_in_pool.shape[0])
                curr_query_grad = []

                model.zero_grad() # temp_dataloader has only one iter, no need to call zero_grad in the loop
                for idxs, images, _ in tqdm(temp_dataloader):
                    if model.device == 'cuda':
                        images = images.to(model.device)
                    outputs = model(images)
                    labels = pesudo_labels[query_idxs_in_pool[idxs]]

                    if model.MSE:
                        labels = torch.nn.functional.one_hot(labels)
                        loss_pool = model.compute_MSELoss(outputs, labels)
                    else:
                        loss_pool = model.compute_CELoss(outputs, labels)

                    loss_pool.backward()
                    for name, param in model.named_parameters():
                        curr_query_grad.append(param.grad.reshape(1, -1))

                curr_query_grad = torch.hstack(curr_query_grad)
                model.zero_grad()
                torch.cuda.empty_cache()

                if args.inf_type == 'residual':
                    queried_grads += curr_query_grad.to(queried_grads.device) - de_inf_grad * args.grad_update_size
                elif args.inf_type == 'full':
                    queried_grads += curr_query_grad.to(queried_grads.device)

            queried_idxs.extend(query_idxs_in_pool.tolist())

        print('t2{}'.format(time.time() - t))


        if len(self.query_g_idx) == 0:
            self.query_g_idx.append(list(train_data[2]))
            self.query_g_idx.append(list(pool_loader.dataset.g_idx[queried_idxs]))
        else:
            self.query_g_idx.append(list(pool_loader.dataset.g_idx[queried_idxs]))


        ###### update dataloader ######
        pool_all_idxs = set(list(range(pool_data[1].shape[0])))
        query_idxs_set = set(queried_idxs)
        remain_idxs = list(pool_all_idxs - query_idxs_set)

        train_data_, pool_data_ = update_dataset(train_data, pool_data, queried_idxs, remain_idxs)

        train_dataloader_ = create_dataloder(args, train_data_, shuffle=True)
        pool_dataloader_ = create_dataloder(args, pool_data_, shuffle=False)

        return train_dataloader_, pool_dataloader_, train_data_, pool_data_
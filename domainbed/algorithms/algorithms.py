# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

import math
import torch.nn.init as init

import time
def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer =  torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
      

    def update(self, x, y, **kwargs):
        time_b = time.time()
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        self.optimizer.step()
        time_e = time.time()
        # print("memory_allocated_GB", torch.cuda.memory_allocated() / (1024 ** 3))
        # print("memory_allocated_GBmax:",torch.cuda.max_memory_allocated()/ (1024 ** 3))

        return {"loss": loss.item(),"memory_M": torch.cuda.max_memory_allocated()/ (1024 ** 3),'memory1':memory1,'time':time_e-time_b}

    def predict(self, x):
        return self.network(x)


class ERM_linear(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_linear, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer =  torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
      

    def update(self, x, y, **kwargs):
        time_b = time.time()
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        self.optimizer.step()
        time_e = time.time()
        # print("memory_allocated_GB", torch.cuda.memory_allocated() / (1024 ** 3))
        # print("memory_allocated_GBmax:",torch.cuda.max_memory_allocated()/ (1024 ** 3))

        return {"loss": loss.item(),"memory_M": torch.cuda.max_memory_allocated()/ (1024 ** 3),'memory1':memory1,'time':time_e-time_b}

    def predict(self, x):
        return self.network(x)


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}




class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape # 我们会把系数展开，所以只有一个维度的长度
        self.b = nn.Parameter(torch.full(b_shape, init)) # 给予其一个初始化

    def forward(self, x):
        return F.softplus(self.b) + self.eps

class Pre_layer (ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.alpha = hparams['alpha'] # 平衡ERM和预训练损失的参数
        self.layer_lr = hparams['layer_lr'] # 层方差估计用的神经网络的学习率
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pre_model = self.featurizer.parameters()
        self.pre_model = self.pre_model.to(self.device)
        self.pre_model_list = []
        self.var_list = []
        for layer in self.pre_model:
            layer_tensor = layer.reshape(-1)
            self.pre_model_list.append(layer_tensor)
            self.var_list.append(VarianceEncoder(shape=len(layer_tensor))) # 初始化方差
        self.var_list = nn.ModuleList(self.var_list)
        parameters = [
            {"params": self.network.parameters()},
            {"params": self.var_list.parameters(), "lr": self.layer_lr},
        ]
        self.optimizer =  get_optimizer(hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"])
        

    def update(self, x, y, **kwargs):
        # 这里好像没有转移到显卡上？
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        # 训练
        z = self.featurizer(all_x)
        predict = self.classifier(z)
        loss = F.cross_entropy(predict,all_y)
        # 计算预训练损失
        layer_list = self.classifier.parameters()
        for layer, pre_layer ,var_e in zip(layer_list,self.pre_model_list,self.var_list):
            layer = layer.reshape(-1)
            var = var_e(layer)
            vlb = (layer-pre_layer).pow(2).div(var) + var.log()
            reg_loss += vlb.mean()/2
        loss += self.alpha*reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "reg_loss": reg_loss.item()}

    
    def predict(self, x):
        return self.network(x)  


from models.resnet_domain import  resnet50 as domainresnet50

from torch import optim

import random
def get_optim_and_scheduler(model, network, epochs, lr, train_all=True, nesterov=False):
    if train_all:
        params = model.parameters()
    else:
        params = model.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_style(style_net, epochs, lr, nesterov=False, step_radio=0.8):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * step_radio)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_optim_and_scheduler_layer_joint(style_net, epochs, lr, train_all=None, nesterov=False):
    optimizer = optim.SGD(style_net, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * 1.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d for style net" % step_size)
    return optimizer, scheduler


def get_model_lr(name, model, fc_weight=1.0):
    if 'resnet' in name:
        return [
            (model.conv1, 1.0),     # 0
            (model.bn1, 1.0),       # 1
            (model.layer1, 1.0),    # 2
            (model.domain_discriminators[1], 1.0),   # 3
            (model.layer2, 1.0),    # 4
            (model.domain_discriminators[2], 1.0),   # 5
            (model.layer3, 1.0),    # 6
            (model.domain_discriminators[3], 1.0),    # 7
            (model.layer4, 1.0),    # 8
            (model.domain_discriminators[4], 1.0),    # 9
            (model.classifier, 1.0 * fc_weight)   # 10
        ]
    elif name == 'alexnet':
        return [
            (model.layer0, 1.0),  # 0
            (model.layer1, 1.0),  # 1
            (model.layer2, 1.0),  # 2
            (model.feature_layers, 1.0),  # 3
            (model.fc, 1.0 * fc_weight),  # 4
        ]
    else:
        raise NotImplementedError


def get_optimizer(model, init_lr, momentum=.9, weight_decay=.0005, nesterov=False):
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay,
                          nesterov=nesterov)
    return optimizer


def get_optim_and_scheduler_scatter(model, network, epochs, lr, momentum=.9, weight_decay=.0005, nesterov=False, step_radio=0.8):
    model_lr = get_model_lr(name=network, model=model, fc_weight=1.0)
    optimizers = [get_optimizer(model_part, lr * alpha, momentum, weight_decay, nesterov)
                  for model_part, alpha in model_lr]
    step_size = int(epochs * step_radio)
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=step_size) for opt in optimizers]
    return optimizers, schedulers


class DomainDropout(Algorithm):
   

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super(DomainDropout, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.network =domainresnet50(pretrained=True,
                device=self.device,
                classes=num_classes,
                domains=num_domains,
                network='resnet50',
                domain_discriminator_flag=hparams['domain_discriminator_flag'],
                grl=hparams['grl'],
                lambd=hparams['lambd'],
                drop_percent=hparams['drop_percent'],
                wrs_flag=hparams['filter_WRS_flag'],
                recover_flag=hparams['recover_flag'],)
        self.discriminator_layers = self.hparams['discriminator_layers']
        self.optimizer_scatter, self.scheduler_scatter = get_optim_and_scheduler_scatter(self.network,
                                                                                         network='resnet50',
                                                                                         epochs=5000, # 这里前面都不变，后面跑domainnet改成20000
                                                                                         lr=hparams['lr'],
                                                                                         nesterov=False)
        self.layer_wise_prob = hparams['layer_wise_prob']
        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self.domain_discriminator_flag = hparams['domain_discriminator_flag']
        self.domain_loss_flag = hparams['domain_loss_flag']
        self.discriminator_layers = hparams['discriminator_layers']





    def select_layers(self, layer_wise_prob):
        # layer_wise_prob: prob for layer-wise dropout
        layer_index = np.random.randint(len(self.hparams['discriminator_layers']), size=1)[0]

        layer_select = self.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select - 1] = 1
        return layer_drop_flag
    
    def compute_kl_loss(self, p, q, pad_mask=None, T=10):
        p_T = p / T
        q_T = q / T
        p_loss = F.kl_div(F.log_softmax(p_T, dim=-1), F.softmax(q_T, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q_T, dim=-1), F.softmax(p_T, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        # p_loss = p_loss.sum()
        # q_loss = q_loss.sum()

        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss


    def update(self,x, y, **kwargs ):
            all_x = torch.cat([xi for xi in x])
            class_l = torch.cat([yi for yi in y])
            minibatches = to_minibatch(x, y)
            domain_l = torch.cat(
                [
                    torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                    for i, (x, y) in enumerate(minibatches)
                ]
            )
            self.network.train()
            
            CE_loss = 0.0
            batch_num = 0.0
            class_right = 0.0
            class_total = 0.0

            CE_domain_loss = [0.0 for i in range(5)]
            domain_right = [0.0 for i in range(5)]
            CE_domain_losses_avg = 0.0
            KL_loss = 0.0


            layer_drop_flag = self.select_layers(layer_wise_prob=self.layer_wise_prob)
            optimizer = self.optimizer_scatter

            class_logit, domain_logit = self.network(x=all_x, domain_labels=domain_l, layer_drop_flag=layer_drop_flag)
            class_loss = self.criterion(class_logit, class_l)
            CE_loss += class_loss
            domain_losses_avg = torch.tensor(0.0).to(device=self.device)

            if self.domain_discriminator_flag == 1:
                domain_losses = []
                for i, logit in enumerate(domain_logit):
                    domain_loss = self.domain_criterion(logit, domain_l)
                    domain_losses.append(domain_loss)
                    CE_domain_loss[i] += domain_loss
                domain_losses = torch.stack(domain_losses, dim=0)
                domain_losses_avg = domain_losses.mean(dim=0)
            CE_domain_losses_avg += domain_losses_avg

            loss = 0.0
            loss += class_loss
            if self.domain_loss_flag == 1:
                loss += domain_losses_avg
            if self.hparams['KL_Loss'] == 1:
                batch_size = int(class_logit.shape[0] / 2)
                class_logit_1 = class_logit[:batch_size]
                class_logit_2 = class_logit[batch_size:]
                kl_loss = self.compute_kl_loss(class_logit_1, class_logit_2, T=self.hparams['KL_Loss_T'])
                loss += self.hparams['KL_Loss_weight'] * kl_loss
                KL_loss += kl_loss

            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                opt.step()

            _, class_pred = class_logit.max(dim=1)
            class_right_batch = torch.sum(class_pred == class_l.data)
            class_right += class_right_batch

            domain_right_batch = [torch.tensor(0.0).cuda() for i in range(5)]
            if self.domain_discriminator_flag == 1:
                for i, logit in enumerate(domain_logit):
                    _, domain_pred = logit.max(dim=1)
                    domain_right_batch[i] = torch.sum(domain_pred == domain_l.data)
                    domain_right[i] += domain_right_batch[i]
            batch_num += 1
            return {"CE_LOSS":CE_loss.item(),"KL_loss":KL_loss.item()}

    def predict(self, x):
        y, _ = self.network(x=x, layer_drop_flag=[0, 0, 0, 0])  
        return y


# Vmamba backbone
from collections import OrderedDict

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class ERM_V(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


from .shuffle import random_swap_blocks_corners,random_swap_blocks_across_images,random_blur_blocks,random_zero_blocks,random_rotate_blocks_corners,add_learnable_vectors

class ERM_V_shuffle(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_Vss(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_V_shuffle2(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_swap_blocks_across_images(all_x,k=self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)

class ERM_V_shuffle3(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_blur_blocks(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_V_shuffle4(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_zero_blocks(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)

class ERM_V_shuffle5(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_Vs(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']) # 也许需要额外固定参数
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)




class ERM_V_shuffle6(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,self.k)
        all_x = random_swap_blocks_across_images(all_x,k=self.k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)



class ERM_V_shuffle7(Algorithm): # 全训
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
        self.lv = torch.nn.Parameter(torch.randn(4, self.k, 3, 4, 4)) # 后面两个跟patch大小有关
        self.lv = torch.nn.Parameter(self.lv.to('cuda'))
        self.optimizer2= torch.optim.SGD(
            [self.lv],
            lr=self.hparams["lr2"],
            weight_decay=self.hparams['weight_decay'])


      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_x = add_learnable_vectors(all_x,self.lv)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer2.step()
        self.optimizer.step()
        

        return {"loss": loss.item()}

    def predict(self, x):
        x = add_learnable_vectors(x,self.lv)
        return self.Tnetwork.network(x)



class ERM_V_shuffle8(Algorithm): #只训分类头
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
        self.lv = torch.nn.Parameter(torch.randn(4, self.k, 3, 4, 4)) # 后面两个跟patch大小有关
        self.lv = torch.nn.Parameter(self.lv.to('cuda'))
        self.optimizer2= torch.optim.SGD(
            [self.lv],
            lr=self.hparams["lr2"],
            weight_decay=self.hparams['weight_decay'])

      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_x = add_learnable_vectors(all_x,self.lv)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer2.step()
        self.optimizer.step()
        

        return {"loss": loss.item()}

    def predict(self, x):
        x = add_learnable_vectors(x,self.lv)
        return self.Tnetwork.network(x)



class ERM_V_shuffle9(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,self.k)
        all_x = random_swap_blocks_across_images(all_x,k=self.k)
        all_x = random_zero_blocks(all_x,self.k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)

# 越加越少 震荡叠加（算两个）


class ERM_V_shuffle10(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.count = 0
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        count = self.count / 1000
        count = 1 if count < 1 else count 
        k = round(self.k/count)
        k = int(k)
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,k=k)
        all_x = random_swap_blocks_across_images(all_x,k=k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)



def oscillate(count, k):
    return (math.sin(count) + 1) / 2 * k



def linear_oscillate(count, k):
    period = k * 2
    x = count % period
    if x < k:
        return x
    else:
        return period - x



class ERM_V_shuffle11(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.count = 0
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
       
        k = int(oscillate(self.count,self.k))
        # k = 1 if k<1 else k
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,k=k)
        all_x = random_swap_blocks_across_images(all_x,k=k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_V_shuffle12(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.count = 0
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
       
        k =int( linear_oscillate(self.count,self.k))
        k = 1 if k<1 else k
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,k=k)
        all_x = random_swap_blocks_across_images(all_x,k=k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def predict(self, x):
        return self.Tnetwork.network(x)



class ERM_V_shuffle13(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_x = random_rotate_blocks_corners(all_x,k=self.k)
        all_x = random_zero_blocks(all_x,k=self.k)
        all_x = random_swap_blocks_across_images(all_x,k=self.k)
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)



class ERM_V_shuffle14(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']) # 只训练分类器看看

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.layer_shapes = []
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.gumble_linear_list = nn.ModuleList(CustomNet(h) for h in self.h_list)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.001,weight_decay=self.hparams['weight_decay'])

       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_across_images(all_x,k=self.k)
        # all_x = random_rotate_blocks_corners(all_x,k=self.k)
        # all_x = random_zero_blocks(all_x,k=self.k)
        
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)
    
    def hook_fn(self, module, input, output):
        input_shape = input[0].shape
        output_shape = output.shape
        self.layer_shapes.append((input_shape, output_shape))


    def register_hooks(self,model):
        for layer in model.children():
            layer.register_forward_hook(self.hook_fn)
            if len(list(layer.children())) > 0:
                self.register_hooks(layer)
    
    def register(self):
        self.Tnetwork.network = self.Tnetwork.network.to('cuda')
        self.register_hooks(self.Tnetwork.network)
        input_tensor = torch.rand(1, 3, 224, 224)  # 根据模型的输入形状调整
        input_tensor = input_tensor.to('cuda')
        output = self.Tnetwork.network(input_tensor)
        for idx, (input_shape, output_shape) in enumerate(self.layer_shapes):
            print(f"Layer {idx}: Input shape: {input_shape}, Output shape: {output_shape}")
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x+x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

class CustomNet(nn.Module):
    def __init__(self, h):
        super(CustomNet, self).__init__()
        # 定义一个线性层，输入和输出大小为 hxw
        self.linear = nn.Linear(h * h, h * h)
        self.binary_concrete_layer = BinaryConcrete(tau=0.1)
    
    def forward(self, x):
        # 获取输入张量的形状信息
        shape = x.shape
        
        # 前两个维度合并，后两个维度合并
        x = x.reshape(-1, shape[-2] * shape[-1])
        
        # 通过线性层
        x = self.linear(x)
        x = x.reshape(shape[0], shape[1], shape[2], shape[3])
        x_s = self.binary_concrete_layer(x) # 这个操作不对，我们的想法应该是一张28*28的表的比如说每一行的概率和要为1，这里明显不对
        x = x*x_s
        
        # 恢复到原始形状
       #  x = x.reshape(shape[0], shape[1], shape[2], shape[3])
        return x
        

class BinaryConcrete2(torch.nn.Module):
    def __init__(self, tau=0.1):
        super(BinaryConcrete2, self).__init__()
        self.tau = tau

    def forward(self, logits):
        # Reshape logits to apply Binary Concrete across the last two dimensions
        shape = logits.shape
        logits = logits.view(-1, shape[-2], shape[-1])
        
        # Sample from Gumbel(0, 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        
        # Apply the Binary Concrete formula
        y = logits + gumbel_noise
        y = torch.sigmoid(y / self.tau)
        
        # Reshape back to the original shape
        y = y.view(*shape)
        
        return y


class BinaryConcrete(torch.nn.Module):
    def __init__(self, tau=0.1):
        super(BinaryConcrete, self).__init__()
        self.tau = tau

    def forward(self, logits):
        # Reshape logits to apply Binary Concrete across the last two dimensions
        shape = logits.shape
        logits = logits.reshape(-1, shape[-2], shape[-1])
        
        # Sample from Gumbel(0, 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        
        # Apply the Binary Concrete formula
        y = logits + gumbel_noise
        y = torch.sigmoid(y / self.tau)
        
        # Apply the straight-through estimator
        y_hard = torch.round(y)
        y = (y_hard - y).detach() + y  # Straight-through estimator
        
        # Reshape back to the original shape
        y = y.reshape(*shape)
        y_hard = y_hard.reshape(*shape)
        # 这里有更改！ 现在在测试的时候返回的是离散值，训练的时候返回的是正常值！
        if self.training:
            return y  # During training, return the continuous version
        else:
            return y_hard  # During evaluation, return the discrete version



class ConvLayer(nn.Module):
    def __init__(self, in_channels,tau=0.1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels, # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.binary_concrete_layer = BinaryConcrete(tau=tau)

    def forward(self, x):
        x = self.conv(x)
        x_s = self.binary_concrete_layer(x)
        x = x_s*x
        return x

class ConvLayer2(nn.Module): # 不含离散的测试
    def __init__(self, in_channels):
        super(ConvLayer2, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels, # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.binary_concrete_layer = BinaryConcrete(tau=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)
        # x_s = self.binary_concrete_layer(x)
        # x = x_s*x
        return x


class ConvLayer2s(nn.Module): # 不含离散的测试
    def __init__(self, in_channels,out_channels):
        super(ConvLayer2s, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels, # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.binary_concrete_layer = BinaryConcrete(tau=0.1)

    def forward(self, x):
        x = self.conv(x)
        # x = F.relu(x)
        x_s = self.binary_concrete_layer(x)
        x = x_s*x
        return x

class ConvLayer3(nn.Module): # 不含离散的测试
    def __init__(self, in_channels,tau=0.1):
        super(ConvLayer3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels, # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.binary_concrete_layer = BinaryConcrete(tau=tau)

    def forward(self, x):
        x = self.conv(x)
        x_s = self.binary_concrete_layer(x)
        return x_s



class ConvLayer4(nn.Module):   # /home/home_node7/wzb/自己算法/miro-main2/domainbed/lib/fast_data_loader.py 出的 FastDataLoader drop_last=True才能用 相当于丢掉了最后的一个batch
    def __init__(self, in_channels, tau=0.1):
        super(ConvLayer4, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels, # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.in_channels = in_channels
        self.binary_concrete_layer = BinaryConcrete(tau=tau)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        # print(batch_size)
        # print(self.in_channels)

        assert batch_size % self.in_channels == 0, "batch size must be a multiple of in_channels"

        # 将 batch 维度重新 reshape，使得每个分块大小为 in_channels
        num_splits = batch_size // self.in_channels
        x = x.reshape(num_splits, self.in_channels, in_channels, height, width)  # 形状变为 [num_splits, in_channels, in_channels, height, width]

        # 在每个分块上交换 batch 和 channel 维度
        x = x.permute(0, 2, 1, 3, 4)  # [num_splits, in_channels, in_channels, height, width] -> [num_splits, in_channels, in_channels, height, width]
        
        # 将每个分块分别进行卷积操作
        x = self.conv(x.reshape(-1, self.in_channels, height, width))  # 卷积操作后，形状为 [num_splits * in_channels, in_channels, height, width]

        # 经过 BinaryConcrete 层
        x = self.binary_concrete_layer(x)

        # 还原形状，交换回 batch 和 channel 维度
        x = x.reshape(num_splits, in_channels, self.in_channels, height, width)  # 将形状恢复为 [num_splits, in_channels, in_channels, height, width]
        x = x.permute(0, 2, 1, 3, 4)  # 从 [num_splits, in_channels, in_channels, height, width] -> [num_splits, in_channels, in_channels, height, width]

        # 将结果 reshape 回原来的 batch size
        x = x.reshape(batch_size, in_channels, height, width)
        
        return x



class ConvLayer5(nn.Module):
    def __init__(self, in_channels, tau=0.1):
        super(ConvLayer5, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels,  # 输出通道数（与输入通道数一致）
            kernel_size=3,             # 卷积核大小 3x3
            stride=1,                  # 步长为1
            padding=1                  # 填充1，以保证输出尺寸不变
        )
        self.binary_concrete_layer = BinaryConcrete(tau=tau)

    def forward(self, x):
        batch_size, total_channels, height, width = x.shape
        assert total_channels % self.in_channels == 0, "Input channels must be a multiple of in_channels"

        # 将通道维度划分为多个块，每块大小为 in_channels
        num_splits = total_channels // self.in_channels
        x_split = torch.split(x, self.in_channels, dim=1)  # 按通道分割

        # 对每个分块单独进行卷积操作
        outputs = []
        for x_part in x_split:
            x_conv = self.conv(x_part)
            x_conv = self.binary_concrete_layer(x_conv)
            outputs.append(x_conv)

        # 将所有分块沿通道维度重新拼接
        x_out = torch.cat(outputs, dim=1)

        return x_out





class CustomNet3(nn.Module): # 简单线性层
    def __init__(self, h):
        super(CustomNet, self).__init__()
        # 定义一个线性层，输入和输出大小为 hxw
        self.linear = nn.Linear(h * h, h * h)
        self.binary_concrete_layer = BinaryConcrete(tau=0.1)
    
    def forward(self, x):
        # 获取输入张量的形状信息
        shape = x.shape
        
        # 前两个维度合并，后两个维度合并
        x = x.reshape(-1, shape[-2] * shape[-1])
        
        # 通过线性层
        x = self.linear(x)
        x = x.reshape(shape[0], shape[1], shape[2], shape[3])
        # x_s = self.binary_concrete_layer(x) # 这个操作不对，我们的想法应该是一张28*28的表的比如说每一行的概率和要为1，这里明显不对
        # x = x*x_s
        
        # 恢复到原始形状
       #  x = x.reshape(shape[0], shape[1], shape[2], shape[3])
        return x



class ERM_V_shuffle15(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']) # 只训练分类器看看

        self.k = self.hparams['k']
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.layer_shapes = []
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.gumble_linear_list = nn.ModuleList(CustomNet(h) for h in self.h_list)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])

       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_across_images(all_x,k=self.k)
        # all_x = random_rotate_blocks_corners(all_x,k=self.k)
        # all_x = random_zero_blocks(all_x,k=self.k)
        
        
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)
    
    def hook_fn(self, module, input, output):
        input_shape = input[0].shape
        output_shape = output.shape
        self.layer_shapes.append((input_shape, output_shape))


    def register_hooks(self,model):
        for layer in model.children():
            layer.register_forward_hook(self.hook_fn)
            if len(list(layer.children())) > 0:
                self.register_hooks(layer)
    
    def register(self):
        self.Tnetwork.network = self.Tnetwork.network.to('cuda')
        self.register_hooks(self.Tnetwork.network)
        input_tensor = torch.rand(1, 3, 224, 224)  # 根据模型的输入形状调整
        input_tensor = input_tensor.to('cuda')
        output = self.Tnetwork.network(input_tensor)
        for idx, (input_shape, output_shape) in enumerate(self.layer_shapes):
            print(f"Layer {idx}: Input shape: {input_shape}, Output shape: {output_shape}")
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x+x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x



class ERM_Vss1(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(CustomNet(h) for h in self.h_list)
        self.conv = ConvLayer(96)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x = x + x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x+x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





class ERM_Vss2(Algorithm): # 卷积版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ConvLayer(h) for h in self.in_c)
        self.conv = ConvLayer(96)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x = x + x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x+x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





class ERM_Vss2t(Algorithm): # 卷积版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ConvLayer(h) for h in self.in_c)
        self.conv = ConvLayer(96)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.Adam(self.conv.parameters(),
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])
        # print(f"Parameters: {count_parameters(self.conv)}")
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x = x + x1
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            if i < 2:
                x2 = gumble_linear(x)
                x = x+x2 # 残差链接  ST-gumble
            i += 1
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)

class ERM_Vss2t1(Algorithm): # 卷积版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ConvLayer(h) for h in self.in_c)
        self.conv = ConvLayer(96)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.Adam(self.conv.parameters(),
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])
        # print(f"Parameters: {count_parameters(self.conv)}")
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x = x + x1
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            if i > 1:
                x2 = gumble_linear(x)
                x = x+x2 # 残差链接  ST-gumble
            i += 1
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)

class ERM_V3(Algorithm): # 前两层冻住的 后两层可以训的
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))
        for param in self.Tnetwork.network.patch_embed.parameters():
            param.requires_grad = False
        i = 0 
        for layer in self.Tnetwork.network.layers: #只有最后的层数可以训练
            if i >= 2:
                for param in layer.parameters():
                    param.requires_grad = False



        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_Vss3(Algorithm): # 卷积不离散版本 但是还是有激活函数
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ConvLayer2(h) for h in self.in_c)
        self.conv = ConvLayer2(96)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        if self.count == 100:
            print(x1)
        x = x + x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x+x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




class ERM_Vss4(Algorithm): # 卷积直接乘版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ConvLayer3(h,tau=self.tau) for h in self.in_c)
        self.conv = ConvLayer3(96,tau=self.tau)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)






# 多头注意力的版本
class ViTPatchDepthwiseConvAttention(nn.Module):
    def __init__(self, in_channels, patch_size, num_heads, embed_dim, dropout=0.1):
        super(ViTPatchDepthwiseConvAttention, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # 深度卷积
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, 
                                        stride=patch_size, groups=in_channels, bias=False)
        # Pointwise卷积，用于改变通道数
        self.pointwise_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        # QKV投影
        self.qkv_proj = nn.Conv2d(embed_dim, 3 * embed_dim, kernel_size=1, bias=False)
        # 多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 输出投影
        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
       
        B, C, H, W = x.shape
        # print(f"Input shape: {x.shape}")
        
        # 执行深度卷积和Pointwise卷积
        x = self.depthwise_conv(x)  # [B, C, H', W']
        x = self.pointwise_conv(x)  # [B, embed_dim, H', W']
        # print(f"After convolutions: {x.shape}")
        
        # 展平空间维度并进行 permute 操作
        patches = x.flatten(2).permute(0, 2, 1)  # [B, num_patches, embed_dim]
        # print(f"Patches shape: {patches.shape}")
        
        # 对 patches 进行 QKV 投影并 chunk
        qkv = self.qkv_proj(patches.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).chunk(3, dim=1)
        q, k, v = qkv
        # print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        
        # **关键点**: 先 permute，使得维度顺序正确
        q = q.permute(0, 2, 1)  # [B, num_patches, embed_dim]
        k = k.permute(0, 2, 1)  # [B, num_patches, embed_dim]
        v = v.permute(0, 2, 1)  # [B, num_patches, embed_dim]

        # 执行多头注意力机制
        attn_output, _ = self.attn(q, k, v)
        # print(f"Attention output shape: {attn_output.shape}")
        
        # 映射回原始的 patches 维度
        attn_output = attn_output.permute(0, 2, 1).view(B, -1, int(H/self.patch_size), int(W/self.patch_size))
        attn_output = self.out_proj(attn_output)
        attn_output = F.interpolate(attn_output, size=(H, W), mode='bilinear', align_corners=False)
        attn_output = F.relu(attn_output)
        
        return attn_output


class DepthwiseSeparableConv1(nn.Module):
    def __init__(self, in_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class PatchAttention1(nn.Module):
    def __init__(self, in_channels, reduced_channels, embed_dim, num_heads):
        super(PatchAttention, self).__init__()
        
        # 1x1 convolution to reduce the number of channels
        self.reduce_channels = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        
        # Depthwise separable convolution
        self.depthwise_separable_conv = DepthwiseSeparableConv(reduced_channels)
        
        # Linear projection for embedding
        self.fc_embed = nn.Linear(reduced_channels, embed_dim)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Linear layer for output
        self.fc_out = nn.Linear(embed_dim, reduced_channels)
        
        # 1x1 convolution to restore the original number of channels
        self.restore_channels = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reduce the number of channels
        x = self.reduce_channels(x)
        if not x.requires_grad:
            print(1,"reduce_channels does not propagate gradients")
        
        # Apply depthwise separable conv
        x = self.depthwise_separable_conv(x)
        if not x.requires_grad:
            print(2,"reduce_channels does not propagate gradients")
        
        # Flatten spatial dimensions
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        if not x.requires_grad:
            print(3,"reduce_channels does not propagate gradients")
        
        # Linear projection to embedding space
        x = self.fc_embed(x)
        if not x.requires_grad:
            print(4,"reduce_channels does not propagate gradients")
        
        # Multi-head attention
        attn_output, _ = self.attn(x, x, x)
        if not x.requires_grad:
            print(5,"reduce_channels does not propagate gradients")
        
        # Linear projection back to reduced channels
        x = self.fc_out(attn_output)
        if not x.requires_grad:
            print(6,"reduce_channels does not propagate gradients")
        
        # Reshape back to spatial dimensions
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if not x.requires_grad:
            print(7,"reduce_channels does not propagate gradients")
        
        # Restore the original number of channels
        x = self.restore_channels(x)
        
        # 检查所有操作后的梯度
        # assert x.requires_grad, "Gradient not propagating!"
        
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class PatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size, num_heads, embed_dim, dropout=0.1):
        super(PatchAttention, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # 深度卷积
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, 
                                        stride=patch_size, groups=in_channels, bias=False)
        # Pointwise卷积，用于改变通道数
        self.pointwise_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        # QKV投影
        self.qkv_proj = nn.Conv2d(embed_dim, 3 * embed_dim, kernel_size=1, bias=False)
        # 多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 输出投影
        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 执行深度卷积和Pointwise卷积
        x = self.depthwise_conv(x)  # [B, C, H', W']
        x = self.pointwise_conv(x)  # [B, embed_dim, H', W']
        
        # 计算patches数量
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # 展平空间维度并进行 permute 操作
        x = x.flatten(2).permute(0, 2, 1)  # [B, num_patches, embed_dim]
        
        # 对 patches 进行 QKV 投影并 chunk
        qkv = self.qkv_proj(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).chunk(3, dim=1)
        q, k, v = qkv
        
        # 执行多头注意力机制
        attn_output, _ = self.attn(q, k, v)
        
        # 将注意力输出恢复为四维形状
        H_ = int(H / self.patch_size)
        W_ = int(W / self.patch_size)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, self.embed_dim, H_, W_)
        
        # 应用输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = F.interpolate(attn_output, size=(H, W), mode='bilinear', align_corners=False)
        attn_output = F.relu(attn_output)
        
        return attn_output

def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters())

class ERM_Vss4(Algorithm): # 是不是softmax的方式不对？
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [768,768]
        self.atten1  = ViTPatchDepthwiseConvAttention(in_channels=96, patch_size=7, num_heads=8, embed_dim=96) # 换成注意力层试一试
        self.atten2 = ViTPatchDepthwiseConvAttention(in_channels=192, patch_size=7, num_heads=8, embed_dim=96)
        self.atten3 = ViTPatchDepthwiseConvAttention(in_channels=384, patch_size=7, num_heads=8, embed_dim=96)
        self.gumble_linear_list = nn.ModuleList([self.atten2,self.atten3]) #v ConvLayer3(h,tau=self.tau) for h in self.in_c
        self.gumble_linear_list.append(ConvLayer3(768,tau=self.tau))
        self.gumble_linear_list.append(ConvLayer3(768,tau=self.tau))
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.Adam(self.atten1.parameters(),
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])

      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        x1 = self.atten1(x)
        # print(x.shape,x1.shape)
        # print(f"Parameters: {count_parameters(self.atten1)}")
       
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)



class ERM_Vss5(Algorithm): # 目前来看影响显著的是最后高通道的特征 想办法降低维度试试 # 可能注意力模块确实更好一些？
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [768,768]
        self.atten1  = PatchAttention(in_channels=96, patch_size=7, embed_dim=64, num_heads=2) # ConvLayer3(96,tau=self.tau) #  # 换成注意力层试一试
        self.atten2 = ConvLayer3(192,tau=self.tau) # PatchAttention(in_channels=192,reduced_channels=48, num_heads=2, embed_dim=96)
        self.atten3 = ConvLayer3(384,tau=self.tau) # PatchAttention(in_channels=384, reduced_channels=48, num_heads=2, embed_dim=96)
        self.gumble_linear_list = nn.ModuleList([self.atten2,self.atten3]) #v ConvLayer3(h,tau=self.tau) for h in self.in_c
        self.gumble_linear_list.append( ViTPatchDepthwiseConvAttention(in_channels=768, patch_size=7, embed_dim=96, num_heads=8))
        self.gumble_linear_list.append( ViTPatchDepthwiseConvAttention(in_channels=768, patch_size=7, embed_dim=96, num_heads=8))
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.atten1.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])

      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.atten1(x)
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


class TwoLayerConv(nn.Module): # 额外的卷积类型
    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(TwoLayerConv, self).__init__()
        # 第一层: 深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU()

        # 第二层: 普通卷积
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # 第一层卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x





class ERM_Vss6(Algorithm): 
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [768,768]
        self.atten1  = PatchAttention(in_channels=96, patch_size=7, embed_dim=64, num_heads=2) # ConvLayer3(96,tau=self.tau) #  # 换成注意力层试一试
        self.atten2 = ConvLayer3(192,tau=self.tau) # PatchAttention(in_channels=192,reduced_channels=48, num_heads=2, embed_dim=96)
        self.atten3 = ConvLayer3(384,tau=self.tau) # PatchAttention(in_channels=384, reduced_channels=48, num_heads=2, embed_dim=96)
        self.gumble_linear_list = nn.ModuleList([self.atten2,self.atten3]) #v ConvLayer3(h,tau=self.tau) for h in self.in_c
        self.gumble_linear_list.append( TwoLayerConv(in_channels=768, hidden_channels=256))
        self.gumble_linear_list.append( TwoLayerConv(in_channels=768, hidden_channels=256))
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.atten1.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])

      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        x1 = self.atten1(x)
        # print(x1.shape)
        # print(x.shape,x1.shape)
        # print(f"Parameters: {count_parameters(self.atten1)}")
       
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


# 前面参数影响少，在前面做试试



class ERM_Vss2e(Algorithm): # 卷积版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(ConvLayer(h) for h in self.in_c)
        self.conv = ConvLayer(96)
        self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        self.gumble_linear_list.append(self.conv2)
        self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x2 = self.conv2(x)
        x = x * x1 + x2
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            if i < 2:
                gumble_linear2 = self.gumble_linear_list2[i]
                x2 = gumble_linear(x)
                x3  = gumble_linear2(x)
                x = x*x2 # 残差链接  ST-gumble
                i += 1
            else:
                x2 = gumble_linear(x)
                x = x*x2
                i += 1
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


# 尚未完成的：1 滑动注意力 top-K  2 不同域形式的 预测 3 这两个看起来也可能可以组合
# SE模块是不是就是在干这件事情？ 来试一试吧


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化：将空间维度(HxW)压缩为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 两层全连接网络：先降维，再升维
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels, in_channels, bias=False)
        
        # Sigmoid激活函数：将输出缩放到0到1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入的批次大小和通道数
        batch_size, num_channels, _, _ = x.size()
        
        # Squeeze：全局平均池化
        y = self.global_avg_pool(x).view(batch_size, num_channels)
        
        # Excitation：两层全连接层
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1)
        
        # 重标定：将计算得到的权重与输入特征图逐通道相乘
        return x * y.expand_as(x)

# 示例用法
'''model = SEBlock(in_channels=256)  # 例如输入有256个通道
input_tensor = torch.randn(1, 256, 32, 32)  # 示例输入
output_tensor = model(input_tensor)
print(output_tensor.shape)  # 输出应为 [1, 256, 32, 32]'''



class TopKMasking(nn.Module):
    def __init__(self, k):
        """
        Args:
            k (float): 百分比，例如 k=0.1 表示只保留前 10% 的最大值。
        """
        super(TopKMasking, self).__init__()
        self.k = k

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        num_elements = channels * height * width
        topk_elements = int(num_elements * self.k)
        
        # 将输入展开为 (batch_size, num_elements) 的二维张量
        x_flat = x.reshape(batch_size, -1)
        
        # 获取前 k% 最大值的位置
        topk_values, topk_indices = torch.topk(x_flat, topk_elements, dim=1, sorted=False)
        
        # 创建一个全0的掩码
        mask = torch.zeros_like(x_flat)
        
        # 将前 k% 的位置设为1
        mask.scatter_(1, topk_indices, 1)
        
        # 恢复掩码形状为原始输入形状
        mask = mask.reshape(batch_size, channels, height, width)
        
        # 将掩码应用于输入特征图
        x_masked = x * mask
        
        return x_masked



class ERM_VtopK(Algorithm): # 卷积版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        # self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        # self.in_c = [192,384,768,768]
        # self.in_c2 = [192,384]
        # self.gumble_linear_list = nn.ModuleList(ConvLayer(h) for h in self.in_c)
        self.conv = TopKMasking(0.75)
       #  self.conv2 = TopKMasking(0.5)
        # self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        '''self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        # self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x) # 选取百分之50
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer in self.Tnetwork.network.layers:
            # print(x.shape)
            x = layer(x)
            x = self.conv(x) # 每一层都加上试一下
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


# 真的就在最开头想办法搞一个预训练过的模块来用？
# top - K提取是不是也可以用上在这里？
# 也想人家那样 把高维的层给改成两个维度 一个是通道平均 一个是最大值，然后用一个卷积或者升维度的方法变回n通道 然后激活函数 掩码到原来的特征上面去（这涉及到一个问题 能不能把layer里面的东西拿出来？）
# 看样子确实可以拿出来用 看看这样行不行 至少比加一个维度看起来复杂
# 掩码在特征上一个思路 这些东西提取出来，掩码在输入上是不是也可以？
# 一个用于提取的例子
'''for layer in self.layers:
    # 每个 layer 是 _make_layer 返回的 nn.Sequential 模块
    for block in layer.blocks:
        # block 是 VSSBlock 实例
        # 可以在这里对 block 进行操作，例如 forward 传递
        x = block(x)
    # 经过所有 blocks 之后，进行下采样操作
    x = layer.downsample(x)'''

# 在开头设置一个类似ROI的稀疏性质的图 提取图片的部分
# 然后在后续给一个简单的神经网络 用于粗分类
# 但是在实际使用的时候 把这些部分乘进预训练网络中进行实际测试？

class ERM_VssSE(Algorithm): # SE模块试一试
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(SEBlock(h) for h in self.in_c)
        self.conv = SEBlock(96)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        '''self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        # self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       #  self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1 = self.conv(x)
        x2 = self.conv2(x)
        x = x * x1 
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2
            i += 1
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




class ChannelAvgMaxPool(nn.Module):
    def __init__(self):
        super(ChannelAvgMaxPool, self).__init__()

    def forward(self, x):
        # 输入 x 的形状: (batch_size, channels, height, width)
        
        # 计算通道平均值
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 在通道维度 (dim=1) 上计算平均值
        
        # 计算通道最大值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # 在通道维度 (dim=1) 上计算最大值
        
        # 将两者拼接起来
        output = torch.cat((avg_pool, max_pool), dim=1)  # 在通道维度 (dim=1) 上拼接
        
        return output  # 输出形状: (batch_size, 2, height, width)

class ERM_Vssfor(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.out_c = [96,192,384,768]
        self.in_c = [4,4,30,4]
        self.avgconv =  ChannelAvgMaxPool()
        self.gumble_linear_list = nn.ModuleList(ConvLayer2s(ins,out) for ins,out in zip(self.in_c,self.out_c))
        self.conv = ConvLayer2s(96,96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        '''self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        print(x.shape)
        x1 = self.conv(x)
        x = x*x1 
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer,se_block in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            xg = []
            for block in layer.blocks:
                # block 是 VSSBlock 实例
                # 可以在这里对 block 进行操作，例如 forward 传递
                x = block(x)
                x2 = self.avgconv(x)
                xg.append(x2)
            x2 = torch.cat(xg,dim=1)
            x2 = se_block(x2)
            x = x2*x
            x = layer.downsample(x)

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)

# 想法描述： 只用预训练网络的embed ing层，然后构造一个浅层的神经网络，输入和输出分别对应采样前和采样后的 然后用和预训练模型一样的分类头进行预测
# 训练好的模型拆分开来向上面这种加上去或者剪掉 前面全部固定不动，只再训练一个分类头 或者额外训练一个加和用的alpha
# 或者单层的输入和输出模拟这种模型，只不过权重共享，这样加的位置就不一样了
# 更额外的 利用选择框在前面提前进行变量选择
# 变成三个这样的额外模型


#  额外的稀奇古怪的想法：让这个模型生成的token尽可能的和被随机屏蔽的状态的尽可能相接近（话说这是不是就是MAP来着）
# 或者用上面的模型，变成一个对抗损失?是不是还是太简单了



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 第0层: 输入 [96, 96, 56, 56], 输出 [96, 192, 28, 28]
        self.conv0 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第1层: 输入 [96, 192, 28, 28], 输出 [96, 384, 14, 14]
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第2层: 输入 [96, 384, 14, 14], 输出 [96, 768, 7, 7]
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第3层: 输入 [96, 768, 7, 7], 输出 [96, 768, 7, 7]
        self.conv3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 第0层操作
        x = F.relu(self.conv0(x))  # Conv2d + ReLU
        x = self.pool0(x)          # MaxPool2d
        
        # 第一层操作
        x = F.relu(self.conv1(x))  # Conv2d + ReLU
        x = self.pool1(x)          # MaxPool2d
        
        # 第二层操作
        x = F.relu(self.conv2(x))  # Conv2d + ReLU
        x = self.pool2(x)          # MaxPool2d
        
        # 第三层操作
        x = F.relu(self.conv3(x))  # Conv2d + ReLU

        return x





class ERM_VssC(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.cnet = CustomCNN()
        self.optimizer2 = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        '''self.layer_shapes = []
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.out_c = [96,192,384,768]
        self.in_c = [4,4,30,4]
        self.avgconv =  ChannelAvgMaxPool()
        self.gumble_linear_list = nn.ModuleList(ConvLayer2s(ins,out) for ins,out in zip(self.in_c,self.out_c))
        self.conv = ConvLayer2s(96,96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        '''self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        # x1 = self.conv(x)
        x = self.cnet(x)
        x = self.Tnetwork.network.classifier(x)
        '''if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer,se_block in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            xg = []
            for block in layer.blocks:
                # block 是 VSSBlock 实例
                # 可以在这里对 block 进行操作，例如 forward 传递
                x = block(x)
                x2 = self.avgconv(x)
                xg.append(x2)
            x2 = torch.cat(xg,dim=1)
            if self.count == 0:
                print(x2.shape)
            x2 = se_block(x2)
            # 经过所有 blocks 之后，进行下采样操作
            """  if self.count == 0:
                print(x2.shape)
                print(x.shape) """
            x = x2*x
            x = layer.downsample(x)

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1'''
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)



# 不太行

class ERM_VssC1(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.cnet = CustomCNN()
        self.optimizer2 = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        '''self.layer_shapes = []
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.out_c = [96,192,384,768]
        self.in_c = [4,4,30,4]
        self.avgconv =  ChannelAvgMaxPool()
        self.gumble_linear_list = nn.ModuleList(ConvLayer2s(ins,out) for ins,out in zip(self.in_c,self.out_c))
        self.conv = ConvLayer2s(96,96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        '''self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])'''
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        j = 0
        while j < 10:
            self.loop(all_x,all_y)
            j += 1
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}

    def loop(self,all_x,all_y):
        self.optimizer2.zero_grad()
        loss = F.cross_entropy(self.forward1(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()

    def forward1(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        # x1 = self.conv(x)
        x = self.cnet(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        # print(x1.shape)
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        i  = 0
        list1 = [self.cnet.conv0,self.cnet.conv1,self.cnet.conv2,self.cnet.conv3]
        list2 = [self.cnet.pool0,self.cnet.pool1,self.cnet.pool2,self.cnet.pool2]
        for layer in self.Tnetwork.network.layers :
            layer2 = list1[i]
            layer3 = list2[i]
            x1 = layer(x)
            
            if i < 3:

                x2 = layer2(x) # 每一层都加上试一下
                x2 = layer3(x2)
            else:
                x2 = layer2(x)
            # print(x1.shape)
            # print(x2.shape)
            # x1 = layer3(layer2(x))
            x = x1-x2
            i += 1
            # print(i)
        x = self.Tnetwork.network.classifier(x)
        # self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





# 接下来的思路：部分参数选择，目测在所有block最后一层，下采样之前那一层的影响应该是最显著的，其他冻结住，只训练它试试，然后用类似稀疏参数选择的思路进行比较（目前想法是掩码前和掩码后的特征要尽可能类似，这个扰动算可以吗？）




class ERM_Vss4d(Algorithm): # 卷积直接乘版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [16,16,16,16]
        self.gumble_linear_list = nn.ModuleList(ConvLayer4(h,tau=self.tau) for h in self.in_c)
        self.conv = ConvLayer4(16,tau=self.tau)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        x1 = self.conv(x)
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





class ERM_Vss4d1(Algorithm): # 卷积直接乘版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [96,96,96,96]
        self.gumble_linear_list = nn.ModuleList(ConvLayer5(h,tau=self.tau) for h in self.in_c)
        self.conv = ConvLayer5(96,tau=self.tau)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        x1 = self.conv(x)
        x = x * x1
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x2 = gumble_linear(x)
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name} contains NaN or Inf values.")

class SpecialModelWithAttention(nn.Module):
    def __init__(self):
        super(SpecialModelWithAttention, self).__init__()
        # 第一层卷积，输出 [96, 192, 28, 28]
        self.conv1 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        
        # 第二层卷积，输出 [96, 384, 14, 14]
        self.conv2 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        
        # 第三层卷积，输出 [96, 768, 7, 7]
        self.conv3 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1)
        
        # 第四层卷积，输出 [96, 768, 7, 7]
        self.conv4 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)
        
        # 轻量级注意力机制
        self.attention1 = LightweightAttention(192, 192)
        self.attention2 = LightweightAttention(384, 384)
        self.attention3 = LightweightAttention(768, 768)
        self.attention4 = LightweightAttention(768, 768)
        
        # SE Block
        self.se_block1 = SEBlock(192)
        self.se_block2 = SEBlock(384)
        self.se_block3 = SEBlock(768)
        self.se_block4 = SEBlock(768)

    def forward(self, x):
        # 清空特征列表
        self.features = []

        # 处理每一层
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.se_block1(x)
        self.features.append(F.relu(x))  # 存储卷积层特征
        check_tensor(F.relu(x), "conv1 output")
        
        x = self.conv2(x)
        x = self.attention2(x)
        x = self.se_block2(x)
        self.features.append(F.relu(x))  # 存储卷积层特征
        check_tensor(x, "conv2 output")
        
        x = self.conv3(x)
        x = self.attention3(x)
        x = self.se_block3(x)
        self.features.append(F.relu(x))  # 存储卷积层特征
        check_tensor(x, "conv3 output")
        
        x = self.conv4(x)
        x = self.attention4(x)
        x = self.se_block4(x)
        self.features.append(F.relu(x))  # 存储卷积层特征
        check_tensor(x, "conv4 output")
        
        return x

    def get(self, layer_idx):
        """
        获取指定层的特征。
        """
        if layer_idx < 0 or layer_idx >= len(self.features):
            raise IndexError("Layer index out of range.")
        return self.features[layer_idx]

class LightweightAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        Q = self.query(x).view(batch_size, -1, height * width)
        K = self.key(x).view(batch_size, -1, height * width)
        V = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(Q.permute(0, 2, 1), K)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(V, attention)
        out = out.view(batch_size, channels, height, width)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        excitation = F.relu(self.fc1(squeeze))
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        return x * excitation



class ERM_Vss4d2(Algorithm): # 小网络模型版本
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.tau = hparams['tau']
        self.in_model = SpecialModelWithAttention()
        # self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        # self.in_c = [96,96,96,96]
        # self.gumble_linear_list = nn.ModuleList(ConvLayer5(h,tau=self.tau) for h in self.in_c)
        #  self.conv = ConvLayer5(96,tau=self.tau)
        self.optimizer2 = torch.optim.Adam(self.in_model.parameters(),
        lr = self.hparams["lr"],weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.conv.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
       # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        #self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        # x1 = self.conv(x)
        # x = x * x1
        i = 0
        self.in_model(x)
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed

        for layer in self.Tnetwork.network.layers:
            x = layer(x)
            x2 = self.in_model.get(i)
           #  print(x2.shape)
            i += 1
            x = x*x2 # 残差链接  ST-gumble

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




class ChannelAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionLayer, self).__init__()
        
        # 乘法路径
        self.mul_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
        
        # 加法路径
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
    
    def forward(self, x):
        # 获取乘法和加法的权重
        mul_weight = self.mul_scale(x)  # 大小为 (B, C, 1, 1)
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)
        
        # 乘法操作
        x_mul = x * mul_weight  # 原特征与权重相乘
        
        # 加法操作
        x_add = x + add_weight  # 原特征与权重相加
        
        # 返回乘法和加法之后的结果
        return x_mul + x_add


class ChannelAttentionLayer2(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionLayer2, self).__init__()
        
        # 乘法路径
        self.mul_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
        
        # 加法路径
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
        self.cho = BinaryConcrete(tau=0.1)
    
    def forward(self, x):
        # 获取乘法和加法的权重
        mul_weight = self.mul_scale(x)  # 大小为 (B, C, 1, 1)
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)
        
        # 乘法操作
        x_mul = x * self.cho(mul_weight)  # 原特征与权重相乘
        
        # 加法操作
        x_add = x + self.cho(add_weight)  # 原特征与权重相加
        
        # 返回乘法和加法之后的结果
        return x_mul + x_add



class ERM_Vss4d3(Algorithm): # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ChannelAttentionLayer(h) for h in self.in_c)
        self.conv = ChannelAttentionLayer(96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-4,weight_decay=self.hparams['weight_decay'])


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




class ERM_Vss4d4(Algorithm): # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]

        self.gumble_linear_list = nn.ModuleList(ChannelAttentionLayer2(h) for h in self.in_c)
        self.conv = ChannelAttentionLayer2(96)

        self.gumble_linear_list.append(self.conv)
       
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
       
        

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


class ChannelAttentionLayer3(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionLayer3, self).__init__()
        
        # 乘法路径
        self.mul_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
        
        # 加法路径（可以在 eval 模式下使用）
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )
    
    def forward(self, x):
        # 获取乘法和加法的权重
        mul_weight = self.mul_scale(x)  # 大小为 (B, C, 1, 1)
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)
        
        return mul_weight, add_weight

class ECAAttentionLayer3(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECAAttentionLayer3, self).__init__()
        
        # 乘法路径 (ECA)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D卷积替代全连接层
        self.sigmoid = nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        init.constant_(self.conv.weight, 1)

        # 加法路径 (保持和之前模型一致)
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积  # 激活函数，确保输出在(0, 1)之间
        )
        init.constant_(self.add_scale[0].weight, 0) # 初始化改变一下

    def forward(self, x):
        b, c, h, w = x.size()

        # ECA 乘法路径
        y = self.avg_pool(x).view(b, c)  # 全局平均池化，并将尺寸变为 (B, C)
        y = y.unsqueeze(1)  # 将其扩展为 (B, 1, C) 以适应 1D 卷积
        y = self.conv(y).squeeze(1)  # 1D卷积操作，然后去掉多余的维度，得到 (B, C)
        mul_weight = self.sigmoid(y).view(b, c, 1, 1)  # 大小为 (B, C, 1, 1)

        # 加法路径
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)

        return mul_weight, add_weight



class ECAAttentionLayer4(nn.Module): # 修正结构大小
    def __init__(self, in_channels, k_size=3):
        super(ECAAttentionLayer4, self).__init__()

        # 乘法路径 (ECA)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D卷积替代全连接层
        init.constant_(self.conv.weight, 0) # 0 初始化 不要出问题了

        # 加法路径 (保持和之前模型一致)
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.softmax())
        init.constant_(self.add_scale[0].weight, 0)


    def forward(self, x):
        b, c, h, w = x.size()

        # ECA 乘法路径
        y = self.avg_pool(x).view(b, c)  # 全局平均池化，并将尺寸变为 (B, C)
        y = y.unsqueeze(1)  # 将其扩展为 (B, 1, C) 以适应 1D 卷积
        y = self.conv(y).squeeze(1)  # 1D卷积操作，然后去掉多余的维度，得到 (B, C)

        # 将乘法路径的结果转化为对数空间，随后用 exp 转换回来，确保其大于 0
        log_weight = y  # 假设 log_weight 可以是任意值
        # print(torch.max(log_weight))
        mul_weight = torch.exp(0.5*log_weight).view(b, c, 1, 1)  # exp 确保输出为正，大于 0
        # print(torch.max(mul_weight))

        # 加法路径
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)

        return mul_weight, add_weight

class CustomModel3(nn.Module):
    def __init__(self, in_channels,num_features, k):
        super(CustomModel3, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ChannelAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        split_size = B // self.k  # 确定每一份的大小
        
        if self.training:
            # 训练模式，划分 batch，分别经过独立的处理
            split_x = torch.split(x, split_size, dim=0)  # 划分成 k 份
            output_splits = []
            
            for i in range(self.k):
                mul_weight, _ = self.attention_layers[i](split_x[i])  # 只用乘法路径
                output_splits.append(split_x[i] * mul_weight)  # 每份进行乘法处理
            
            output = torch.cat(output_splits, dim=0)  # 将处理后的份拼接回原来的大小
            
        else:
            # 测试模式，所有份都经过 attention layer 的处理
            total_mul_output = torch.zeros_like(x)  # 用于存放累加结果
            total_add_weight = torch.zeros_like(x[:, :, 0:1, 0:1])  # 存放 add_scale 均值
            total_ln_mean = torch.zeros_like(total_add_weight)  # 存放 LayerNorm 均值

            for i in range(self.k):
                # 使用乘法和加法路径
                mul_weight, add_weight = self.attention_layers[i](x)
                total_mul_output = total_mul_output * x * mul_weight  # 所有层的乘法结果累加 (这里改成相乘了)
                if i == 0:
                    total_add_weight = add_weight  # 加法权重累加
                else:
                    total_add_weight += add_weight

                # 计算 LN 层的均值
                # ln_mean = self.layer_norms[i](x).mean(dim=[2, 3], keepdim=True)  # 对每个 LN 层计算均值
                # total_ln_mean += ln_mean  # 累加每个 LayerNorm 的均值

            # 最后计算加法权重和 LayerNorm 均值的均值
            total_add_weight_mean = total_add_weight / self.k
            # total_ln_mean_mean = total_ln_mean / self.k  # 计算多个 LayerNorm 的均值

            # 将 LayerNorm 的均值添加到输出中
            output = total_mul_output + total_add_weight_mean  # total_ln_mean_mean  
            output = self.layer_norm(x)# 额外添加归一化层
        return output



class ERM_Vss4d5(Algorithm): # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel3(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel3(96,[96,56,56],3)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)




class CustomModel4(nn.Module): # 特征相乘类型
    def __init__(self, in_channels,num_features, k):
        super(CustomModel4, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        
    def split_tensor(self,x, k):
    # 获取输入 tensor 的 shape
        batch_size, *other_dims = x.shape
        
        # 确保 batch_size 可以被 k 整除
        assert batch_size % k == 0, "Batch size must be divisible by k"
        
        # 计算每个分块的大小
        chunk_size = batch_size // k
        
        # 切分 tensor
        split_tensors = torch.chunk(x, k, dim=0)
        
        return split_tensors
    
    def forward(self, x):
        # B, C, H, W = x.shape
        
        if self.training:
            # 训练模式，使用每层的乘法和加法路径
            output_splits = []
            split_x = self.split_tensor(x,self.k)

            for i,xi in zip(range(self.k),split_x):
                mul_weight, add_weight = self.attention_layers[i](xi)  # 使用乘法和加法路径
                output_splits.append(xi * mul_weight + add_weight)  # 每层乘法和加法处理
            
            output = torch.cat(output_splits, dim=0)  # 训练时的输出为每层结果的平均值
            # print(output.shape)
            
        else:
            # 测试模式，将每个层的 `(x * mul_weight + add_weight)` 结果相乘
            total_output = torch.ones_like(x)  # 初始化为全1张量
            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)  # 使用乘法和加法路径
                total_output *= (x * mul_weight + add_weight)  # 相乘结果为每个层的 (x * mul_weight + add_weight)
            
            output = total_output  # 最终输出为相乘结果
            output = self.layer_norm(x)# 额外添加归一化层 这个层不能删掉
            
        return output





class ERM_Vss4d6(Algorithm): # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]

        self.gumble_linear_list = nn.ModuleList(CustomModel4(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel4(96,[96,56,56],3)

        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.count += 1
        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)

        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)

from .new_z import CustomModels,SPU_min,SPUr,SPU5,SPU6
class ERM_Vss4d61(Algorithm): # 这看起来是有用的
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]

        self.gumble_linear_list = nn.ModuleList(CustomModels(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModels(96,[96,56,56],3)

        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-5,weight_decay=self.hparams['weight_decay'])


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.count += 1
        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)

        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


def kl_divergence_normal(mul_weight, add_weight):
    # 确保标准差 (mul_weight) 大于 0，避免log(0)
    # 使用 torch.clamp() 防止数值过小导致 log(0) 错误
    mul_weight = torch.clamp(mul_weight, min=1e-8)

    # 计算KL散度
    kl_loss = torch.log(mul_weight) + 0.5 * (mul_weight.pow(2) + add_weight.pow(2) - 1)
    
    # 对batch维度求平均
    return kl_loss.mean()


class CustomModel5(nn.Module): # 正态分布相乘类型

    def __init__(self, in_channels,num_features, k):
        super(CustomModel5, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer4(in_channels) for _ in range(k)]) # 这里改成4了
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False) 

    def split_tensor(self,x, k):
    # 获取输入 tensor 的 shape
        batch_size, *other_dims = x.shape
        
        # 确保 batch_size 可以被 k 整除
        assert batch_size % k == 0, "Batch size must be divisible by k"
        
        # 计算每个分块的大小
        chunk_size = batch_size // k
        
        # 切分 tensor
        split_tensors = torch.chunk(x, k, dim=0)
        
        return split_tensors
    
    def forward(self, x):
        epsilon = 1e-6  # 加入一个小值，防止除零
        # B, C, H, W = x.shape
        kl_loss = 0
        
        if self.training:
            # 训练模式，使用每层的乘法和加法路径
            output_splits = []
            split_x = self.split_tensor(x,self.k)
            

            for i,xi in zip(range(self.k),split_x):
                # print(xi.requires_grad)
                mul_weight, add_weight = self.attention_layers[i](xi)  # 使用乘法和加法路径
                # print(torch.max(mul_weight))
                output_splits.append(xi * mul_weight + add_weight)  # 每层乘法和加法处理
                kl_loss += kl_divergence_normal(mul_weight, add_weight)
                
            
            output = torch.cat(output_splits, dim=0) 
            
        else:
             
            # 测试模式，将每个层的 `(x * mul_weight + add_weight)` 结果相乘
            
            # 测试模式，根据多个正态分布的乘积计算新的均值和方差
            total_weight = torch.zeros_like(x)  # 用于累加均值部分
            total_precision = torch.zeros_like(x)  # 用于累加精度 (precision) ，即方差的倒数

            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)
                
                # 计算均值 (mul_weight) 和方差 (mul_weight^2 表示方差)
                variance = mul_weight ** 2  # 方差为 mul_weight 的平方
                precision = 1 / variance  # 精度是方差的倒数
                total_precision += precision  # 累加每个分布的精度
                total_weight += precision * mul_weight  # 加权累加每个分布的均值

            # 根据精度计算新的均值和方差
            combined_mean = total_weight / (total_precision + epsilon)
            combined_variance = 1 / (total_precision + epsilon)  # 加入小的值避免出现0的情况
            # check_for_nan_inf(combined_variance, "combined_variance")
            combined_weight = torch.sqrt(combined_variance)  # 标准差（平方根）
            combined_mean = torch.where(torch.isnan(combined_mean), torch.tensor(0.0), combined_mean) # 修正输出看一下
            # 输出形式为 `x * combined_weight + combined_mean`
            output = x * combined_weight + combined_mean  # 理论上来讲这个应该更合适才对，但是好像结果不怎么行，很奇怪。

            check_for_nan_inf(combined_weight, "combined_weight")
            check_for_nan_inf(combined_mean, "combined_mean")
            # if torch.isnan(output).any():
            #     print('1')
            # if torch.isinf(output).any():
            #     print('2')
            # print(torch.isnan(output).any())  # 检查是否有NaN
            # print(torch.isinf(output).any())
            
            
            output = self.layer_norm(output)  # 额外添加归一化层
            
        return output,kl_loss

def check_for_nan_inf(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"Warning: {tensor_name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"Warning: {tensor_name} contains Inf values.")


class ERM_Vss4d7(Algorithm): # adain 新分布 额外加入一个新的kl散度
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5(96,[96,56,56],3)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        y, kl = self.Tnetwork.network(all_x)
        loss = F.cross_entropy(y, all_y)
        loss += 5e-5*kl
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        kl  = 0
        x = self.Tnetwork.network.patch_embed(x)
        x, kl1 = self.conv(x)
        kl += kl1
        # if torch.isnan(x).any():
        #     print(3)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
       #  i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x,kl1 = gumble_linear(x)
            kl += kl1
            # if torch.isnan(x).any():
            #     print(4)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x, kl

    def predict(self, x):
        y, kl = self.Tnetwork.network(x)
        return y



class ERM_Vss4d71(Algorithm): # adain 新分布
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5(96,[96,56,56],3)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x1,kl = self.conv(x)
        x =x + x1
        # if torch.isnan(x).any():
        #     print(3)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
       #  i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x1,kl = gumble_linear(x)
            x =x + x1
            # if torch.isnan(x).any():
            #     print(4)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


class ERM_Vss4d8(Algorithm): # adain 模型初版 加入随机特征
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5(96,[96,56,56],3)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.01,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_rotate_blocks_corners(all_x,self.k)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
        # i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)


# top-K选择（三个都大的）
# 更具体的加法（加到每一层里面去）



class ERM_Vssfor1(Algorithm): # 详细的往里面添加的类型
    """ 
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        # self.k = self.hparams['k']
        # self.Tnetwork.network.forward = self.modified_forward
        # self.count = 0
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [56,28,14,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [96,192,384,768]
        self.gumble_linear_list = nn.ModuleList(ChannelAttentionLayer(h) for h in self.in_c)
        self.conv = ChannelAttentionLayer(96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])

       

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # print(x.shape)
        x = self.conv(x)
        # x = x*x1 
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer,se_block in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            xg = []
            for block in layer.blocks:
                # block 是 VSSBlock 实例
                # 可以在这里对 block 进行操作，例如 forward 传递
                x = block(x)
                x = se_block(x)
                # x2 = self.avgconv(x)
                # xg.append(x2)
            # x2 = torch.cat(xg,dim=1)
            # x2 = se_block(x2)
            # x = x2*x
            x = layer.downsample(x)

        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





class CustomModel6(nn.Module): # 让特征互相可分辨的 还没写完
    def __init__(self, in_channels, num_features, k):
        super(CustomModel6, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        
        # 线性层用于识别不同i的特征
        self.classifier = nn.Linear(num_features, k)

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.training:
            output_splits = []
            total_loss = 0  # 用于累积损失
            
            for i in range(self.k):
                # 获取每层的乘法和加法路径结果
                mul_weight, add_weight = self.attention_layers[i](x)
                output = x * mul_weight + add_weight
                
                # 展平特征并通过线性分类器
                features = output.view(B, -1)
                class_preds = self.classifier(features)  # 分类预测
                
                # 生成对应的标签，每个i对应的标签也是i
                labels = torch.full((B,), i, dtype=torch.long, device=x.device)  # 生成与i匹配的标签
                
                # 计算分类损失（cross-entropy）
                class_loss = nn.CrossEntropyLoss()(class_preds, labels)
                total_loss += class_loss  # 累加损失
                
                # 将处理过的结果存储
                output_splits.append(output)
            
            # 平均化输出特征
            output = torch.stack(output_splits, dim=0).mean(dim=0)
            
            return output, total_loss  # 返回原始输出和累积损失
        
        else:
            # 测试模式：特征乘法累积，不计算损失
            total_output = torch.ones_like(x)
            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)
                total_output *= (x * mul_weight + add_weight)
            
            output = total_output
            output = self.layer_norm(output)  # 额外添加归一化层
            return output,0






class ECAAttentionLayer1(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECAAttentionLayer1, self).__init__()
        
        # 乘法路径 (ECA)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D卷积替代全连接层
        self.sigmoid = nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间

        # 加法路径 (保持和之前模型一致)
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),  # 1x1卷积
            nn.Sigmoid()  # 激活函数，确保输出在(0, 1)之间
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # ECA 乘法路径
        y = self.avg_pool(x).view(b, c)  # 全局平均池化，并将尺寸变为 (B, C)
        y = y.unsqueeze(1)  # 将其扩展为 (B, 1, C) 以适应 1D 卷积
        y = self.conv(y).squeeze(1)  # 1D卷积操作，然后去掉多余的维度，得到 (B, C)
        mul_weight = self.sigmoid(y).view(b, c, 1, 1)  # 大小为 (B, C, 1, 1)

        # 加法路径
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)

        return mul_weight *x + add_weight

class ERM_Vss4deca3(Algorithm): # adaineca 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.gumble_linear_list = nn.ModuleList(ECAAttentionLayer1(h) for h in self.in_c)
        self.conv = ECAAttentionLayer1(96)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-4,weight_decay=self.hparams['weight_decay'])


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
        # x = x * x1 + x2
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)





class ECAAttentionLayer5(nn.Module): # softplus + 无限制
    def __init__(self, in_channels, k_size=3):
        super(ECAAttentionLayer5, self).__init__()

        # 乘法路径 (ECA)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.mul_scale =nn.Sequential( nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
         nn.sigmoid() ) # 1D卷积替代全连接层
        init.constant_(self.mul_scale[0].weight, 1) # 0 初始化 不要出问题了

        # 加法路径 (保持和之前模型一致)
        self.add_scale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False))
        init.constant_(self.add_scale[0].weight, 0)
            # nn.Tanh()  # 1x1卷积 加上额外的激活函数测试
            # relu = nn.ReLU()
            # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
            # elu = nn.ELU(alpha=1.0)
            # softplus = nn.Softplus()
            # output = softmax(input_tensor)
            # output = F.softmax(input_tensor, dim=1)
    #         def swish(x):
    # return x * torch.sigmoid(x)
    # gelu = nn.GELU()
    # 还有一个maxout这里没写


    def forward(self, x):
        b, c, h, w = x.size()

        # ECA 乘法路径
        y = self.avg_pool(x).view(b, c)  # 全局平均池化，并将尺寸变为 (B, C)
        y = y.unsqueeze(1)  # 将其扩展为 (B, 1, C) 以适应 1D 卷积
        mul_weight = self.mul_scale(y).squeeze(1)  # 1D卷积操作，然后去掉多余的维度，得到 (B, C)

        # 将乘法路径的结果转化为对数空间，随后用 exp 转换回来，确保其大于 0
        # log_weight = y  # 假设 log_weight 可以是任意值
        # print(torch.max(log_weight))
        # mul_weight = torch.exp(0.5*log_weight).view(b, c, 1, 1)  # exp 确保输出为正，大于 0
        # print(torch.max(mul_weight))

        # 加法路径
        add_weight = self.add_scale(x)  # 大小为 (B, C, 1, 1)

        return mul_weight, add_weight



class CustomModel5(nn.Module): # 特征相乘类型
    def __init__(self, in_channels,num_features, k):
        super(CustomModel5, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        
    def split_tensor(self,x, k):
    # 获取输入 tensor 的 shape
        batch_size, *other_dims = x.shape
        
        # 确保 batch_size 可以被 k 整除
        assert batch_size % k == 0, "Batch size must be divisible by k"
        
        # 计算每个分块的大小
        chunk_size = batch_size // k
        
        # 切分 tensor
        split_tensors = torch.chunk(x, k, dim=0)
        
        return split_tensors
    def scale_to_01(self,tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def forward(self, x):
        # B, C, H, W = x.shape
        
        if self.training:
            # 训练模式，使用每层的乘法和加法路径
            output_splits = []
            split_x = self.split_tensor(x,self.k)

            for i,xi in zip(range(self.k),split_x):
                mul_weight, add_weight = self.attention_layers[i](xi)  # 使用乘法和加法路径
                output_splits.append(xi * mul_weight + add_weight)  # 每层乘法和加法处理
            
            output = torch.cat(output_splits, dim=0)  # 训练时的输出为每层结果的平均值
            # print(output.shape)
            
        else:
            # 测试模式，将每个层的 `(x * mul_weight + add_weight)` 结果相乘
            total_output = torch.ones_like(x)  # 初始化为全1张量
            tmul_weight = 1
            tadd_weight = 0
            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)  # 使用乘法和加法路径
                tmul_weight = mul_weight*tmul_weight
                tadd_weight = add_weight + tadd_weight
            tadd_weight = tadd_weight/self.k
            # tmul_weight = self.scale_to_01(tmul_weight)  # 改为开三次方？
            tmul_weight = torch.pow(tmul_weight,1/3)
            
            output = x*tmul_weight + tadd_weight # 最终输出为相乘结果
            # output = self.layer_norm(x)# 额外添加归一化层 这个层不能删掉
            
        return output


    # def forward(self, x):
    #     # B, C, H, W = x.shape
        
        
    #         # 训练模式，使用每层的乘法和加法路
            
    #         mul_weight, add_weight = self.attention_layers[0](x)  
    #         output = x * mul_weight + add_weight 

            
    #         return output





class ERM_Vss4d65(Algorithm): # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5(96,[96,56,56],3)
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.Adam(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 5e-3,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # all_x = random_swap_blocks_corners(all_x,self.k)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        x = self.conv(x)
        # x2 = self.conv2(x)
         
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
            # x = x+x1 # 变成残差链接
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)







class CustomModels(nn.Module): # 特征相乘类型
    def __init__(self, in_channels,num_features, k):
        super(CustomModels, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        # self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        


    def forward(self, x):
        # B, C, H, W = x.shape
        
        
            # 训练模式，使用每层的乘法和加法路
            
            mul_weight, add_weight = self.attention_layers[0](x)  
            output = x * mul_weight + add_weight 

            
            return output





class ERM_Vss_straight(Algorithm): # 删掉了其中一个结果
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.optimizer3 =   self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.alpha = hparams['alpha']
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        self.gumble_linear_list = nn.ModuleList(CustomModels(h,c,3) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModels(96,[96,56,56],3)
        self.gumble_linear_list.append(self.conv)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        pre_y = self.predict(all_x)
        loss = F.cross_entropy(pre_y, all_y)
        self.optimizer2.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        x = self.Tnetwork.network.patch_embed(x)
        # x = self.conv(x)
        # i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x= gumble_linear(x)
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)
    





class ERM_Vss_jt(Algorithm):  # adain 模型初版
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        # 定义规范化层
        self.count = 0
        self.alpha = hparams['alpha']
        self.layer_shapes = []
        self.h_list = [28, 14, 7, 7]  # 根据基础模型调整
        self.in_c = [192, 384, 768, 768]
        self.hs = [[192, 28, 28], [384, 14, 14], [768, 7, 7], [768, 7, 7]]
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,  # 假设您已经定义了 LayerNorm2d
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        if norm_layer is None:
            raise ValueError(f"Unsupported normalization layer: {norm_layer}")

        # 初始化 Featurizer
        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False  # 冻结 Featurizer 的所有参数

        # 重新定义分类器，确保其可训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict([
            ('norm', norm_layer(self.Tnetwork.network.num_features)),  # B,H,W,C
            ('permute', Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(1)),
            ('head', nn.Linear(self.Tnetwork.network.num_features, num_classes)),
        ]))

        # 定义分类器的优化器
        self.optimizer_classifier = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # 定义 gumble_linear_list 的优化器
        self.gumble_linear_list = nn.ModuleList(
            CustomModels(h, c, 3) for h, c in zip(self.in_c, self.hs)
        )
        self.conv = CustomModels(96, [96, 56, 56], 3)
        self.gumble_linear_list.append(self.conv)
        self.optimizer_gumble = torch.optim.Adam(
            self.gumble_linear_list.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # 定义其他网络部分的优化器（不包括分类器和 gumble_linear_list）
        other_params = [
            param for name, param in self.Tnetwork.network.named_parameters()
            if 'classifier' not in name ]
        self.optimizer_other = torch.optim.Adam(
            other_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # 替换网络的 forward 方法
        self.Tnetwork.network.forward = self.modified_forward

        # 初始化计数器和其他参数

        self.gumble_linear_list = nn.ModuleList(
            CustomModels(h, c, 3) for h, c in zip(self.in_c, self.hs)
        )
        self.conv = CustomModels(96, [96, 56, 56], 3)
        self.gumble_linear_list.append(self.conv)
        self.optimizer_gumble = torch.optim.Adam(
            self.gumble_linear_list.parameters(),
            lr=3e-5,
            weight_decay=self.hparams['weight_decay']
        )

    def set_trainable(self, train_gumble=True):
        """
        设置参数的 requires_grad 属性。
        
        Args:
            train_gumble (bool): 如果为 True，则训练 gumble_linear_list；否则，训练其他部分。
        """
        # 始终训练分类器
        for param in self.Tnetwork.network.classifier.parameters():
            param.requires_grad = True

        if train_gumble:
            # 训练 gumble_linear_list
            for param in self.gumble_linear_list.parameters():
                param.requires_grad = True
            # 冻结其他网络部分
            for name, param in self.Tnetwork.network.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        else:
            # 冻结 gumble_linear_list
            for param in self.gumble_linear_list.parameters():
                param.requires_grad = False
            # 训练其他网络部分
            for name, param in self.Tnetwork.network.named_parameters():
                if 'classifier' not in name :
                    param.requires_grad = True

    def update(self, x, y, **kwargs):
        # 将输入和标签拼接
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        
        # 预测
        pre_y = self.predict(all_x)
        
        # 计算损失
        loss = F.cross_entropy(pre_y, all_y)
        
        # 决定当前要训练的部分
        train_gumble = (self.count % 10 == 0)


        
        '''if train_gumble:
            print("Training classifier and gumble_linear_list")
        else:
            print("Training classifier and other network parameters")'''
        
        # 设置参数的 requires_grad
        self.set_trainable(train_gumble=train_gumble)
        
        # 清零所有优化器的梯度
        self.optimizer_classifier.zero_grad()
        self.optimizer_gumble.zero_grad()
        self.optimizer_other.zero_grad()
        
        # 反向传播
        loss.backward()
        
        # 更新分类器的参数
        self.optimizer_classifier.step()
        
        # 根据当前训练的部分，更新相应的参数
        if train_gumble:
            self.optimizer_gumble.step()
        else:
            self.optimizer_other.step()
        
        # 增加计数器
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        # 通过 patch_embed 层
        x = self.Tnetwork.network.patch_embed(x)
        
        # 添加位置嵌入（如果存在）
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        
        # 通过每一层和对应的 gumble_linear 层
        for layer, gumble_linear in zip(self.Tnetwork.network.layers, self.gumble_linear_list):
            x = layer(x)
            x = gumble_linear(x)
        
        # 分类器
        x = self.Tnetwork.network.classifier(x)
        
        # 增加计数器
        
        return x

    def predict(self, x):
        return self.Tnetwork.network(x)
# 具体做法：1 训练的时候，自己过自己的，但是维护一个矩阵，让一个域的过三个特征，用来维护这个“不变特征”，约束为不要离的太远
# 2 测试的时候还是一致的。


from.new_z import CustomModelslect
class ERM_Vss4d_select(nn.Module): # 里面东西没有填完 被gpt删掉了一部分
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_Vss4d_select, self).__init__()
        self.Tnetwork = networks.Featurizer(input_shape, hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(self.Tnetwork.network.num_features)),
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(1)),
            ('head', nn.Linear(self.Tnetwork.network.num_features, num_classes)),
        ]))

        self.gumble_linear_list = nn.ModuleList(CustomModelslect(h, c, 3) for h, c in zip(self.in_c, self.hs))
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=hparams["lr"], weight_decay=hparams['weight_decay'])
        self.count = 0

    def update(self, x, y):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()

        for gumble in self.gumble_linear_list:
            gumble.apply_masks_and_average()  # 应用掩码和平均化

        self.optimizer.step()
        self.count += 1
        return {"loss": loss.item()}

    def predict(self, x):
        outputs = []
        for gumble in self.gumble_linear_list:
            output = gumble(x)
            outputs.append(output)

        # 返回所有掩码为0的部分
        return torch.cat(outputs, dim=0)


def euclidean_distance_loss1(feature_list, epsilon=1e-6): # 欧几里得距离损失
    """
    feature_list: 列表，每个元素是一个四维张量 [b, c, h, w]
    计算列表中每对特征之间的欧氏距离，鼓励它们相互远离
    """
    num_features = len(feature_list)
    distance_loss = 0.0
    
    # 遍历特征对 (i, j)，计算每对特征之间的欧氏距离
    for i in range(num_features - 1):
        for j in range(i + 1, num_features):
            # 计算两个四维张量的欧氏距离
            dist = torch.norm(feature_list[i] - feature_list[j], p=2)
            distance_loss += 1 / (dist + epsilon)  # 距离越近，损失越大

    return distance_loss


def euclidean_distance_loss2(feature_list, epsilon=1e-6): # 欧几里得距离损失,拉近距离
    """
    feature_list: 列表，每个元素是一个四维张量 [b, c, h, w]
    计算列表中每对特征之间的欧氏距离，鼓励它们相互远离
    """
    num_features = len(feature_list)
    distance_loss = 0.0
    
    # 遍历特征对 (i, j)，计算每对特征之间的欧氏距离
    for i in range(num_features - 1):
        for j in range(i + 1, num_features):
            # 计算两个四维张量的欧氏距离
            dist = torch.norm(feature_list[i] - feature_list[j], p=2)
            distance_loss += dist   # 距离越近，损失越大

    return distance_loss

class EuclideanDistanceLossWithCenter: # 滑动更新方法
    def __init__(self, alpha=0.1):
        """
        初始化，alpha 是滑动更新的系数。
        """
        self.alpha = alpha
        self.center = None  # 初始状态下 center 为空
    
    def update_center(self, feature_list):
        with torch.no_grad():
            if self.center is None:
                # 如果 center 还没有初始化，直接用第一个特征初始化
                self.center = torch.zeros_like(feature_list[0])
            
            # 遍历特征列表，更新中心值
            for feature in feature_list:
                    self.center = self.alpha * feature + (1 - self.alpha) * self.center  # 滑动更新公式
    
    def compute_loss(self, feature_list):
        """
        计算所有特征与中心值之间的欧氏距离损失。
        """
        # 更新中心值
        self.update_center(feature_list)
        
        # 计算特征与中心值的欧氏距离
        distance_loss = 0.0
        for feature in feature_list:
            dist = torch.norm(feature - self.center, p=2)  # L2距离
            distance_loss += dist

        return distance_loss

def cosine_distance_loss(feature_list, epsilon=1e-6): # 余弦距离
    """
    feature_list: 列表，每个元素是一个四维张量 [b, c, h, w]
    计算列表中每对特征之间的余弦相似度，鼓励它们相互远离
    """
    num_features = len(feature_list)
    distance_loss = 0.0
    
    # 遍历特征对 (i, j)，计算每对特征之间的余弦距离
    for i in range(num_features - 1):
        for j in range(i + 1, num_features):
            # 展平四维张量为一维向量
            feature_i = feature_list[i].reshape(-1)
            feature_j = feature_list[j].reshape(-1)
            
            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(feature_i, feature_j, dim=0)
            # 余弦距离 = 1 - 余弦相似度
            distance_loss += ((1 - cosine_sim) + epsilon) # 现在是正的

    return distance_loss


dist_dict = {'euclidean_distance_loss1':euclidean_distance_loss1,'euclidean_distance_loss2':euclidean_distance_loss2,}

class CustomModel5loss(nn.Module): # 特征相乘类型,损失版本
    def __init__(self, in_channels,num_features, k,loss_name):
        super(CustomModel5loss, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        if loss_name == 'cosine_distance_loss':
            self.dist = cosine_distance_loss
        if loss_name == 'euclidean_distance_loss1':
            self.dist = euclidean_distance_loss1
        if loss_name == 'euclidean_distance_loss2':
            self.dist = euclidean_distance_loss2
        if loss_name == ' combined_distance_loss':
            self.dist =  combined_distance_loss
        if loss_name == 'EuclideanDistanceLossWithCenter':
            self.dist = EuclideanDistanceLossWithCenter()
        
    def split_tensor(self,x, k):
    # 获取输入 tensor 的 shape
        batch_size, *other_dims = x.shape
        
        # 确保 batch_size 可以被 k 整除
        assert batch_size % k == 0, "Batch size must be divisible by k"
        
        # 计算每个分块的大小
        chunk_size = batch_size // k
        
        # 切分 tensor
        split_tensors = torch.chunk(x, k, dim=0)
        
        return split_tensors
    def scale_to_01(self,tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def forward(self, x):
        # B, C, H, W = x.shape
        loss = 0
        
        if self.training:
            # 训练模式，使用每层的乘法和加法路径
            output_splits = []
            split_x = self.split_tensor(x,self.k)

            for i,xi in zip(range(self.k),split_x):
                mul_weight, add_weight = self.attention_layers[i](xi)  # 使用乘法和加法路径
                output_splits.append(xi * mul_weight + add_weight)  # 每层乘法和加法处理

            
            output = torch.cat(output_splits, dim=0)  # 训练时的输出为每层结果的平均值
            loss = self.dist(output_splits)
            # print(output.shape)
            
        else:
            # 测试模式，将每个层的 `(x * mul_weight + add_weight)` 结果相乘
            total_output = torch.ones_like(x)  # 初始化为全1张量
            tmul_weight = 1
            tadd_weight = 0
            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)  # 使用乘法和加法路径
                tmul_weight = mul_weight*tmul_weight
                tadd_weight = add_weight + tadd_weight
            tadd_weight = tadd_weight/self.k
            # tmul_weight = self.scale_to_01(tmul_weight)  # 改为开三次方？
            tmul_weight = torch.pow(tmul_weight,1/3)
            
            output = x*tmul_weight + tadd_weight # 最终输出为相乘结果
            # output = self.layer_norm(x)# 额外添加归一化层 这个层不能删掉
            
        return output,loss

def correlation_loss(features):
    k = len(features)
    pooled_features = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in features]
    concatenated = torch.stack(pooled_features, dim=0).permute(1, 0, 2)

    batch_size_k = concatenated.size(0)
    C = concatenated.size(2)

    concatenated = (concatenated - concatenated.mean(dim=0)) / (concatenated.std(dim=0) + 1e-5)
    cov = torch.bmm(concatenated.transpose(1, 2), concatenated) / (C - 1)

    # 使用与 cov 形状一致的 eye
    eye = torch.eye(cov.size(1), device=features[0].device).unsqueeze(0)  # [1, 192, 192] 或者其他相应的形状

    # 计算非对角元素
    cov_no_diag = cov * (1 - eye)
    loss = torch.mean(torch.sum(torch.abs(cov_no_diag), dim=(1, 2)))

    return loss

class CustomModel5lossc(nn.Module):  # 特征相乘类型, 损失版本
    def __init__(self, in_channels, num_features, k, loss_name, lambda_corr=0.1):
        """
        初始化自定义模型，集成特征去相关机制。
        
        Args:
            in_channels (int): 输入通道数
            num_features (int): 特征维度
            k (int): 划分的份数
            loss_name (str): 损失函数名称
            lambda_corr (float): 去相关损失的权重
        """
        super(CustomModel5lossc, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.lambda_corr = lambda_corr  # 去相关损失的权重
        
        # if loss_name == 'EuclideanDistanceLossWithCenter':
        #     self.dist = EuclideanDistanceLossWithCenter()
        # else:
        #     raise ValueError(f"Unsupported loss name: {loss_name}")
        
    def split_tensor(self, x, k):
        """
        将输入张量按 batch 维度切分为 k 份。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, C, H, W]
            k (int): 切分的份数
        
        Returns:
            List[Tensor]: 切分后的张量列表
        """
        batch_size = x.size(0)
        assert batch_size % k == 0, "Batch size must be divisible by k"
        split_tensors = torch.chunk(x, k, dim=0)
        return split_tensors
    
    def scale_to_01(self, tensor):
        """
        将张量缩放到 [0, 1] 范围。
        
        Args:
            tensor (Tensor): 输入张量
        
        Returns:
            Tensor: 缩放后的张量
        """
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def forward(self, x):
        """
        前向传播方法。
        
        Args:
            x (Tensor): 输入张量，形状为 [B, C, H, W]
        
        Returns:
            Tuple[Tensor, Tensor]: 输出张量和损失（训练模式下）
        """
        loss = 0
        
        if self.training:
            # 训练模式，使用每层的乘法和加法路径
            output_splits = []
            split_x = self.split_tensor(x, self.k)
            
            for i, xi in enumerate(split_x):
                mul_weight, add_weight = self.attention_layers[i](xi)  # 使用乘法和加法路径
                output_splits.append(xi * mul_weight + add_weight)  # 每层乘法和加法处理
            
            output = torch.cat(output_splits, dim=0)  # 训练时的输出为所有分割后的输出拼接
            
            # 计算欧几里得距离损失
            # loss_dist = self.dist.compute_loss(output_splits)
            
            # 计算特征去相关损失
            loss_corr = correlation_loss(output_splits)
            
            # 总损失 = 欧几里得距离损失 + lambda_corr * 去相关损失
            loss = self.lambda_corr * loss_corr
            
        else:
            tmul_weight = torch.ones_like(x)  # 初始化为全1张量
            tadd_weight = torch.zeros_like(x)
            for i in range(self.k):
                mul_weight, add_weight = self.attention_layers[i](x)  # 使用乘法和加法路径
                tmul_weight = mul_weight * tmul_weight
                tadd_weight = add_weight + tadd_weight
            # 对乘法权重进行缩放
            tmul_weight = torch.pow(tmul_weight, 1/self.k)
            
            output = x * tmul_weight + tadd_weight  # 最终输出为相乘结果
            
        return output, loss


def combined_distance_loss(feature_list, lambda_euclidean=0.5, lambda_cosine=0.5, epsilon=1e-6): # 混合距离损失
    """
    feature_list: 列表，每个元素是一个四维张量 [b, c, h, w]
    结合欧氏距离和余弦距离的正则化损失
    """
    euclidean_loss = euclidean_distance_loss(feature_list, epsilon)
    cosine_loss = cosine_distance_loss(feature_list, epsilon)
    
    return lambda_euclidean * euclidean_loss + lambda_cosine * cosine_loss

class ERM_Vss4d65loss(Algorithm): # 加入损失类型
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.alpha = hparams['alpha']
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5loss(h,c,3,hparams['loss_name']) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5loss(96,[96,56,56],3,hparams['loss_name'])
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        pre_y,loss1 = self.predict1(all_x)
        loss = F.cross_entropy(pre_y, all_y)
        loss = loss + self.alpha *loss1
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        loss = 0
        x = self.Tnetwork.network.patch_embed(x)
        # x,loss1 = self.conv(x)
        # loss += loss1
        # x2 = self.conv2(x)
         
        # i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x,loss1 = gumble_linear(x)
            loss += loss1
            # x = x+x1 # 变成残差链接
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x , loss

    def predict1(self, x):
        return self.Tnetwork.network(x)
    

    def predict(self, x):
        y,loss = self.Tnetwork.network(x)
        return y



class ERM_Vss4d65lossc(Algorithm): # 加入损失类型2
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        for param in self.Tnetwork.network.parameters():
            param.requires_grad = False
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))

        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
        self.k = self.hparams['k']
        self.Tnetwork.network.forward = self.modified_forward
        self.count = 0
        self.alpha = hparams['alpha']
        self.layer_shapes = []
        self.count = 0
        self.h_list = [28,14,7,7] # 这里只是测试！如果改了基本模型这里也得跟着改
        self.in_c = [192,384,768,768]
        self.hs = [[192,28,28],[384,14,14],[768,7,7],[768,7,7]]
        # self.in_c2 = [192,384]
        self.gumble_linear_list = nn.ModuleList(CustomModel5lossc(h,c,3,hparams['loss_name']) for h,c in zip(self.in_c,self.hs))
        self.conv = CustomModel5lossc(96,[96,56,56],3,hparams['loss_name'])
        # self.conv2 = ConvLayer(96)
        self.gumble_linear_list.append(self.conv)
        # self.gumble_linear_list.append(self.conv2)
        # self.gumble_linear_list2 = nn.ModuleList(ConvLayer(h) for h in self.in_c2)
        self.optimizer2 = torch.optim.SGD(self.gumble_linear_list.parameters(),  # 这切换成了Adam试试
        lr = 0.1,weight_decay=self.hparams['weight_decay'])
        # self.optimizer3 = torch.optim.SGD(self.gumble_linear_list2.parameters(),
        # lr = 0.1,weight_decay=self.hparams['weight_decay'])
        

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        pre_y,loss1 = self.predict1(all_x)
        loss = F.cross_entropy(pre_y, all_y)
        loss = loss + self.alpha *loss1
        self.optimizer2.zero_grad()
        # self.optimizer3.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # self.optimizer3.step()
        self.count += 1

        return {"loss": loss.item()}
    
    def modified_forward(self, x: torch.Tensor):
        loss = 0
        x = self.Tnetwork.network.patch_embed(x)
       #  x,loss1 = self.conv(x)
        # loss += loss1
        # x2 = self.conv2(x)
         
        i = 0
        if self.Tnetwork.network.pos_embed is not None:
            pos_embed = self.Tnetwork.network.pos_embed.permute(0, 2, 3, 1) if not self.Tnetwork.network.channel_first else self.Tnetwork.network.pos_embed
            x = x + pos_embed
        for layer ,gumble_linear in zip(self.Tnetwork.network.layers,self.gumble_linear_list):
            x = layer(x)
            x,loss1 = gumble_linear(x)
            loss += self.alpha * loss1
            # x = x+x1 # 变成残差链接
        x = self.Tnetwork.network.classifier(x)
        self.count+= 1
        return x , loss

    def predict1(self, x):
        return self.Tnetwork.network(x)
    

    def predict(self, x):
        y,loss = self.Tnetwork.network(x)
        return y




class ERM_R_a(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_R_a, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.network.forward = self.mod_forward
        self.count = 0
      

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def mod_forward(self,x):
        for name, layer in self.featurizer.network.named_children():  # 访问 ResNet 中的每个层
            x = layer(x)  # 将输入通过当前层
            
            # print(f'After {name}, shape: {x.shape}')  # 打印当前输出形状
        x = x.squeeze()
        # print(f'shape: {x.shape}')
        x = self.classifier(x)
        return x

    def predict(self, x):
        return self.network(x)









# 冻结参数进行微调的做法





arga = {
    'device': 'cuda',
    'score_batch_percentage': 1.0,
    'save_ckpt': False,
    'log_path': None,
    'selection_rate': 0.001,
    'k':3,
    'swa_start':4000,
    'swa_freq':100,
    'c': 0.5,
    'selected_layers':'conv3',
    'way':'mean',
    'quantile_threshold':0.75,
    'batch_size':32,
    'mask_merge_method':'intersection'
}

class ERM_V_select(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        
        norm_layer = 'ln2d'
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.Tnetwork = networks.Featurizer(input_shape, self.hparams)
        # 分类头必须要重新训练
        self.Tnetwork.network.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.Tnetwork.network.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.Tnetwork.network.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.Tnetwork.network.num_features, num_classes),
        ))
        self.small_set_x = []
        self.small_set_y = []


        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.Tnetwork.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])
      #self.optimizer = get_optimizer(
            #hparams["optimizer"],
            #self.Tnetwork.network.parameters(),
            #lr=self.hparams["lr"],
            #weight_decay=self.hparams["weight_decay"],
        # ) 
        self.k = self.hparams['k']
        self.select = SPU(arga,self.Tnetwork.network)
        self.select_name = mlp.c_fc  # 这个要查找一下
        for name, param in self.Tnetwork.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        if self.count <= self.set_count:
                self.small_set_x.append(all_x)
                self.small_set_y.append(all_y)
        elif self.count == self.set_count:
            # 这里的loss可能要单独写
            # 比如一个类似map形式的loss
            self.select.compute_importance(self.small_set_x,slef.small_set_y,'classfication',self.select_name)
        else: # 锁定之后再开始优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.Tnetwork.network(x)


from .new_z import SPU_x2,SPU_list,SPU_step,SPU_step2,SPU_stepg,SPU_step_random,SPU_step_du,SPU_step_direct,SPU_step_du2

class SPU:
    def __init__(self, args, Tnetwork):
        self.args = args  # args 现在是字典
        self.Tnetwork = Tnetwork  # 模型
        self.mask = {}
        self.trainable_params = []

    def compute_score(self, data_list, labels_list, selected_layers):
        """
        计算模型中特定层的梯度显著性，数据为简单列表
        """
        # Initialize importance matrix for the selected layers
        importance = {name: torch.zeros_like(param, device='cpu') for name, param in self.Tnetwork.network.named_parameters() if name in selected_layers}

        # 模型切换到训练模式
        self.Tnetwork.network.train()
        self.Tnetwork.network.zero_grad()

        total_batches = len(data_list)
        num_batches_for_score = int(total_batches * self.args['score_batch_percentage'])
        print(f'Total batches for importance: {total_batches}, use {num_batches_for_score} batches')

        stop_flag = 1

        for num_batch in range(min(num_batches_for_score, total_batches)):
            # 获取 batch 数据
            images = data_list[num_batch].to(self.args['device'])
            labels = labels_list[num_batch].to(self.args['device'])

            # 前向传播
            logits_per_image = self.Tnetwork.network(images)

            # 计算交叉熵损失
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss.backward()

            # 仅累积特定层的梯度重要性
            for name, param in self.Tnetwork.network.named_parameters():
                if selected_layers in name and param.requires_grad and param.grad is not None:
                    # 累积梯度到CPU，避免显存溢出
                    importance[name] += param.grad.detach().cpu().clone()

                    if importance[name].abs().min() < 1e-12:
                        stop_flag = 0

            # 当达到指定批次并且停止标志生效时停止
            if num_batch >= num_batches_for_score and stop_flag:
                break

        return importance

    def compute_importance(self, data_list, labels_list, task, selected_layers):
        """
        计算特定任务下模型中特定层的梯度显著性，并锁住不需要更新的参数
        """
        print('Compute importance for the current task...')

        # 计算当前任务的显著性
        cur_importance = self.compute_score(data_list, labels_list, selected_layers)

        # 保存显著性矩阵到CPU
        if self.args['save_ckpt']:
            with open(os.path.join(self.args['log_path'], f'task{task}_importance.torchSave'), 'wb') as file:
                torch.save(cur_importance, file)

        # 使用显著性矩阵构建掩码并锁住不重要的参数
        with torch.no_grad():
            for name, param in self.Tnetwork.network.named_parameters():
                if name in self.trainable_params:
                    if name not in cur_importance.keys():
                        print(f' Importance of `{name}` is none')
                        continue

                    importance = cur_importance[name]
                    magnitudes = importance.abs()
                    k = int(magnitudes.numel() * self.args['selection_rate'])

                    # 计算 top-k 显著性并生成 mask
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    self.mask[name] = torch.zeros_like(magnitudes).to(self.args['device'])
                    self.mask[name].view(-1)[topk_indices] = 1

                    # 根据 mask 锁住不需要更新的参数
                    if self.mask[name].sum() == 0:
                        param.requires_grad = False  # 锁住不需要更新的参数
                    else:
                        param.requires_grad = True   # 保留需要更新的参数



class SPU1:
    def __init__(self, args, Tnetwork, lr, weight_decay):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.trainable_params = []
        self.lr = lr
        self.weight_decay = weight_decay

    def compute_score(self, data_list, labels_list, selected_layers):
        importance = {name: torch.zeros_like(param, device='cpu') for name, param in self.network.named_parameters() if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        total_batches = len(data_list)
        num_batches_for_score = int(total_batches * self.args['score_batch_percentage'])
        print(f'Total batches for importance: {total_batches}, use {num_batches_for_score} batches')

        for num_batch in range(min(num_batches_for_score, total_batches)):
            images = data_list[num_batch].to(self.args['device'])
            labels = labels_list[num_batch].to(self.args['device'])

            logits_per_image = self.network(images)
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss.backward()

            for name, param in self.network.named_parameters():
                if selected_layers in name and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()

        return importance

    def compute_importance(self, data_list, labels_list, task, selected_layers):
        print('Compute importance for the current task...')
        cur_importance = self.compute_score(data_list, labels_list, selected_layers)
        for name, param in self.network.named_parameters():
                if selected_layers not in name:
                    param.requires_grad = False
                    # print(f"{name} is locked (not in selected layers)")
                else:
                    # 如果参数在选择的层中，处理显著性
                    if name not in cur_importance.keys():
                        print(f'Importance of `{name}` is none')
                        continue

        params_to_update = []
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if selected_layers in name:
                    if name not in cur_importance.keys():
                        print(f' Importance of `{name}` is none')
                        continue

                    importance = cur_importance[name]
                    magnitudes = importance.abs()
                    k = int(magnitudes.numel() * self.args['selection_rate'])
                    # print(magnitudes.numel())
                    # print(k,'has been selected')

                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    selected_param_indices = topk_indices.tolist()  # 获取索引列表

                    # 创建掩码并选择显著性参数
                    self.mask[name] = torch.zeros_like(magnitudes).to(self.args['device'])
                    self.mask[name].view(-1)[selected_param_indices] = 1

                    # 针对选定的显著性参数
                    for idx in selected_param_indices:
                        param.requires_grad = True  # 确保显著参数可以更新
                        params_to_update.append(param.view(-1)[idx])  # 添加具体参数
                        # print(idx)

        self.optimizer = torch.optim.Adam(
            params_to_update,
            lr=self.lr,
            weight_decay=self.weight_decay)

class SPU2:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.trainable_params = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.selected_layers = args['selected_layers'].split()
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
            
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data_list, labels_list, selected_layers):
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}
        self.network.train()
        self.network.zero_grad()

        # total_batches = len(data_list)
        # num_batches_for_score = int(total_batches * self.args['score_batch_percentage'])
        # print(f'Total batches for importance: {total_batches}, use {num_batches_for_score} batches')
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx] 
            for data, labels in zip( current_data , current_labels):
                # 前向传播
                logits_per_image = self.network(data)  # 获取当前批次的输出
                loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
                loss.backward()  # 反向传播计算梯度

                # 累加梯度
                for name, param in self.network.named_parameters():
                    if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                        importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

                # 清除梯度，避免累加
                self.network.zero_grad()

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        print('Compute importance for the current task...')
        cur_importance = self.compute_score(data_list, labels_list, selected_layers)
        
        # 冻结所有参数
        for name, param in self.network.named_parameters():
            param.requires_grad = False
        
        # 选择并解冻重要参数
        selected_params = []
        self.mask = {}
        for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in cur_importance:
                   #  print(f'Importance of `{name}` is none')
                    # continue

                    importance = cur_importance[name]
                    magnitudes = importance.abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    # print(f'Parameter: {name}, Total params: {magnitudes.numel()}, Selected params: {k}')

                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    selected_param_indices = topk_indices.tolist()  # 获取索引列表

                    # 创建掩码并选择显著性参数
                    mask = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask.view(-1)[topk_indices] = 1
                    self.mask[name] = mask

                    # 解冻整个参数张量
                    param.requires_grad = True
                    selected_params.append(param)
        
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        print(f'Number of trainable parameters: {len(selected_params)}')
    
    def train_step(self, data, labels):
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        x = self.network(data)
        y = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码，保留选定参数的梯度，其余设为零
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        self.optimizer.step()
        self.optimizer2.step()

        return [loss.item(),0]
    
    def predict(self, data):
        x = self.network(data)
        y = self.classifier(x)
        return y

class SPU3:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                # print(name)
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
    
                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}
    
        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1
    
                final_mask[name] = layer_mask
    
        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        # s = 0
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                    # s += 1
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练
    
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # print(f'Number of trainable parameters: {s}')
    
        self.mask = final_mask  # 保存最终掩码
        # print(f'Final mask computed for each layer.')
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count
           #  print(f'Mask for {key} has {count} ones.')

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y


class SPU_sgd:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
    
                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}
    
        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1
    
                final_mask[name] = layer_mask
    
        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练
    
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.SGD(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        print(f'Number of trainable parameters: {len(selected_params)}')
    
        self.mask = final_mask  # 保存最终掩码
        print(f'Final mask computed for each layer.')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y



class SPU4: # 加上滑动平均
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类，并集成滑动平均。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
            swa_start (int): 从哪个训练步骤开始进行滑动平均。
            swa_freq (int): 每隔多少步执行一次滑动平均。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']

        # 滑动平均相关
        self.swa_start = args['swa_start']
        self.swa_freq = args['swa_freq']
        self.swa_model = None  # 用于保存滑动平均的模型权重
        self.swa_n = 0  # 用于计算滑动平均的次数
        self.step = 0

    def apply_swa(self):
        """
        仅对未被冻结的参数执行滑动平均。
        """
        if self.swa_model is None:
            # 如果是第一次更新，直接复制当前模型的状态
            self.swa_model = {name: param.clone().detach().cpu() for name, param in self.network.named_parameters()
                              if name in self.mask and self.mask[name].sum() > 0}
        else:
            # 更新滑动平均
            for name, param in self.network.named_parameters():
                if name in self.mask and self.mask[name].sum() > 0:
                    self.swa_model[name] += param.clone().detach().cpu()

        self.swa_n += 1

    def update_swa_model(self):
        """
        应用滑动平均模型参数，替换网络中的权重。
        """
       
        if self.swa_model is not None:
            # print('now begin swa models'*10)
            for name, param in self.network.named_parameters():
                if name in self.swa_model:
                    avg_param = self.swa_model[name] / self.swa_n
                    param.data.copy_(avg_param.to(param.device))

    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']): 
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        
        final_mask = {}
    
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])

                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
    
                final_mask[name] = layer_mask
    
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
    
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
        self.mask = final_mask
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()
            total_count += count

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤，并根据训练步数执行滑动平均。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        self.optimizer.step()
        self.optimizer2.step()
        self.step+= 1
        
        # 每隔一定步数执行滑动平均
        if self.step >= self.swa_start and self.step % self.swa_freq == 0:
            self.apply_swa()

        return loss.item()
    
    def predict(self, data):
        """
        进行预测。
        """
        self.update_swa_model()
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y

class ERM_select(Algorithm): # resnet 版本 试一试
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        '''self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])'''
       
        # self.small_set_x = []
        # self.small_set_y = []
        self.selection_rate = hparams['selection_rate']
        self.warm_optimizer = torch.optim.Adam(self.classifier.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams['weight_decay'])
        arga["selection_rate"] = self.selection_rate
        arga['swa_start'] = hparams['swa_start']
        arga['swa_freq'] = hparams['swa_freq']
        arga['selected_layers'] = hparams['select_name']
        arga['way'] = hparams['way']
        arga['quantile_threshold'] = hparams['quantile_threshold']
        arga['batch_size'] = hparams['batch_size']
        self.hparams = hparams

        # self.k = self.hparams['k'] 
        if hparams['select'] == 'SPU3':
            self.select = SPU3(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU4':
            self.select = SPU4(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_sgd':
            self.select = SPU_sgd(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_min':
            self.select = SPU_min(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPUr':
            self.select = SPUr(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU5':
            self.select = SPU5(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU6':
            self.select = SPU6(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_kmeans':
            self.select = SPU_kmeans(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_x2':
            self.select = SPU_x2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_list':
            self.select = SPU_list(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step':
            self.select = SPU_step(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step2':
            self.select = SPU_step2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_stepg':
            self.select = SPU_stepg(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU2':
            self.select = SPU2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_random':
            self.select = SPU_step_random(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du':
            self.select = SPU_step_du(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_direct':
            self.select = SPU_step_direct(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du2':
            self.select = SPU_step_du2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)

        else:
            print('wrong method.')
            


        self.select_name =  hparams['select_name'] # 这个要查找一下
        self.count = 0
        self.set_count = hparams['set_count']
        self.recompute_count = hparams['recompute_count']
        self.per = hparams['per']
        self.k = hparams['k']
        self.name = hparams['select']
        self.warm = hparams['warm']
        self.small_set_x = [[] for _ in range(self.k)]
        # print(len(self.small_set_x)) 
        self.small_set_y = [[] for _ in range(self.k)] 

        # self.optimizer =  torch.optim.Adam(self.classifier.parameters(),lr = self.hparams['lr'],weight_decay = self.hparams['weight_decay'])
        '''for name, param in self.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")'''


    def update(self, x, y, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = 0
        memory1 = 0
        if self.count <= self.warm:
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.warm_optimizer.zero_grad()
            loss.backward()
            self.warm_optimizer.step()
            loss = loss.item()
        if self.count <= self.set_count: # 这里需要修正一下
            for i,(x_k,y_k) in enumerate(zip(x,y)):
                self.small_set_x[i].append(x_k)
                self.small_set_y[i].append(y_k)
        if self.count == self.set_count:
            # 这里的loss可能要单独写
            # 比如一个类似map形式的loss
            self.select.compute_importance(self.small_set_x,self.small_set_y,self.select_name)
            if self.name == 'SPU_step':
                self.select.analyze_mask() # 看一下秩和值的关系
            if self.name == 'SPUr':
                self.select.refine_final_mask(self.small_set_x,self.small_set_y,self.select_name)
        if self.count >  self.set_count: # 锁定之后再开始优化
            if self.recompute_count:
                if self.count % self.recompute_count== 0:
                    self.select.compute_importance(self.small_set_x, self.small_set_y, self.select_name) # 重新计算
            
        # 缩小更新的参数量
            if 'du' in self.name:
                orig_mean, orig_max, delta_mean, delta_max = self.select.train_step(all_x, all_y)
                self.count += 1
                return {'orig_mean':orig_mean, 'orig_max':orig_max, 'delta_mean':delta_mean, 'delta_max':delta_max}
            else:
                loss,memory1 = self.select.train_step(all_x, all_y)
                self.count += 1
                return {"loss": loss,"memory_allocated_GB": torch.cuda.max_memory_allocated() / (1024 ** 3),"memory1":memory1}
        self.count += 1
        return {'loss':0}    
        # loss = self.select.train_step(all_x, all_y)
       

    def predict(self, x):
        return self.select.predict(x)




class ERM_select23(Algorithm): # resnet 版本 试一试
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        '''self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])'''
       
        # self.small_set_x = []
        # self.small_set_y = []
        self.selection_rate = hparams['selection_rate']
        self.warm_optimizer = torch.optim.Adam(self.classifier.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams['weight_decay'])
        arga["selection_rate"] = self.selection_rate
        arga['swa_start'] = hparams['swa_start']
        arga['swa_freq'] = hparams['swa_freq']
        arga['selected_layers'] = hparams['select_name']
        arga['way'] = hparams['way']
        arga['quantile_threshold'] = hparams['quantile_threshold']
        arga['batch_size'] = hparams['batch_size']
        self.hparams = hparams

        # self.k = self.hparams['k'] 
        if hparams['select'] == 'SPU3':
            self.select = SPU3(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU4':
            self.select = SPU4(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_sgd':
            self.select = SPU_sgd(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_min':
            self.select = SPU_min(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPUr':
            self.select = SPUr(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU5':
            self.select = SPU5(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU6':
            self.select = SPU6(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_kmeans':
            self.select = SPU_kmeans(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_x2':
            self.select = SPU_x2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_list':
            self.select = SPU_list(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step':
            self.select = SPU_step(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step2':
            self.select = SPU_step2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_stepg':
            self.select = SPU_stepg(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU2':
            self.select = SPU2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_random':
            self.select = SPU_step_random(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du':
            self.select = SPU_step_du(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_direct':
            self.select = SPU_step_direct(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du2':
            self.select = SPU_step_du2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)

        else:
            print('wrong method.')
            


        self.select_name =  hparams['select_name'] # 这个要查找一下
        self.count = 0
        self.set_count = hparams['set_count']
        self.recompute_count = hparams['recompute_count']
        self.per = hparams['per']
        self.k = hparams['k']
        self.name = hparams['select']
        self.warm = hparams['warm']
        self.small_set_x = [[] for _ in range(self.k)]
        # print(len(self.small_set_x)) 
        self.small_set_y = [[] for _ in range(self.k)] 

        # self.optimizer =  torch.optim.Adam(self.classifier.parameters(),lr = self.hparams['lr'],weight_decay = self.hparams['weight_decay'])
        '''for name, param in self.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")'''


    def update(self, x, y, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = 0
        memory1 = 0
        # if self.count <= self.warm:
        #     loss = F.cross_entropy(self.network(all_x), all_y)
        #     self.warm_optimizer.zero_grad()
        #     loss.backward()
        #     self.warm_optimizer.step()
        #     loss = loss.item()
        if self.count <= self.set_count: # 这里需要修正一下
            for i,(x_k,y_k) in enumerate(zip(x,y)):
                self.small_set_x[i].append(x_k)
                self.small_set_y[i].append(y_k)
        if self.count == self.set_count:
            # 这里的loss可能要单独写
            # 比如一个类似map形式的loss
            self.select.compute_importance(self.small_set_x,self.small_set_y,self.select_name)
            if self.name == 'SPU_step':
                self.select.analyze_mask() # 看一下秩和值的关系
            if self.name == 'SPUr':
                self.select.refine_final_mask(self.small_set_x,self.small_set_y,self.select_name)
        if self.count >  self.set_count: # 锁定之后再开始优化
            if self.recompute_count:
                if self.count % self.recompute_count== 0:
                    self.select.compute_importance(self.small_set_x, self.small_set_y, self.select_name) # 重新计算
            
        # 缩小更新的参数量
            if 'du' in self.name:
                orig_mean, orig_max, delta_mean, delta_max = self.select.train_step(all_x, all_y)
                self.count += 1
                return {'orig_mean':orig_mean, 'orig_max':orig_max, 'delta_mean':delta_mean, 'delta_max':delta_max}
            else:
                loss,memory1 = self.select.train_step(all_x, all_y)
                self.count += 1
                return {"loss": loss,"memory_allocated_GB": torch.cuda.max_memory_allocated() / (1024 ** 3),"memory1":memory1}
        self.count += 1
        return {'loss':0}    
        # loss = self.select.train_step(all_x, all_y)
       

    def predict(self, x):
        return self.select.predict(x)




class ERM_select_noise(Algorithm): # resnet 版本 试一试
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        '''self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])'''
       
        # self.small_set_x = []
        # self.small_set_y = []
        self.selection_rate = hparams['selection_rate']
        self.warm_optimizer = torch.optim.Adam(self.classifier.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams['weight_decay'])
        arga["selection_rate"] = self.selection_rate
        arga['swa_start'] = hparams['swa_start']
        arga['swa_freq'] = hparams['swa_freq']
        arga['selected_layers'] = hparams['select_name']
        arga['way'] = hparams['way']
        arga['quantile_threshold'] = hparams['quantile_threshold']
        arga['batch_size'] = hparams['batch_size']
        self.noise_level = hparams['noise_level']
        self.hparams = hparams

        # self.k = self.hparams['k'] 
        if hparams['select'] == 'SPU3':
            self.select = SPU3(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU4':
            self.select = SPU4(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_sgd':
            self.select = SPU_sgd(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_min':
            self.select = SPU_min(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPUr':
            self.select = SPUr(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU5':
            self.select = SPU5(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU6':
            self.select = SPU6(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_kmeans':
            self.select = SPU_kmeans(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_x2':
            self.select = SPU_x2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_list':
            self.select = SPU_list(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step':
            self.select = SPU_step(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step2':
            self.select = SPU_step2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_stepg':
            self.select = SPU_stepg(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU2':
            self.select = SPU2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_random':
            self.select = SPU_step_random(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du':
            self.select = SPU_step_du(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_direct':
            self.select = SPU_step_direct(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)

        else:
            print('wrong method.')
            


        self.select_name =  hparams['select_name'] # 这个要查找一下
        self.count = 0
        self.set_count = hparams['set_count']
        self.recompute_count = hparams['recompute_count']
        self.per = hparams['per']
        self.k = hparams['k']
        self.name = hparams['select']
        self.warm = hparams['warm']
        self.small_set_x = [[] for _ in range(self.k)]
        # print(len(self.small_set_x)) 
        self.small_set_y = [[] for _ in range(self.k)] 

        # self.optimizer =  torch.optim.Adam(self.classifier.parameters(),lr = self.hparams['lr'],weight_decay = self.hparams['weight_decay'])
        '''for name, param in self.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")'''


    def update(self, x, y, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        # 加噪声试验
        all_x = self.add_gaussian_noise(all_x)
        all_y = torch.cat(y)
        loss = 0
        memory1 = 0
        if self.count <= self.warm:
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.warm_optimizer.zero_grad()
            loss.backward()
            self.warm_optimizer.step()
            loss = loss.item()
        if self.count <= self.set_count: # 这里需要修正一下
            for i,(x_k,y_k) in enumerate(zip(x,y)):
                self.small_set_x[i].append(x_k)
                self.small_set_y[i].append(y_k)
        if self.count == self.set_count:
            # 这里的loss可能要单独写
            # 比如一个类似map形式的loss
            self.select.compute_importance(self.small_set_x,self.small_set_y,self.select_name)
            if self.name == 'SPU_step':
                self.select.analyze_mask() # 看一下秩和值的关系
            if self.name == 'SPUr':
                self.select.refine_final_mask(self.small_set_x,self.small_set_y,self.select_name)
        if self.count >  self.set_count: # 锁定之后再开始优化
            if self.recompute_count:
                if self.count % self.recompute_count== 0:
                    self.select.compute_importance(self.small_set_x, self.small_set_y, self.select_name) # 重新计算
            
        # 缩小更新的参数量
            if self.name == 'SPU_step_du':
                orig_mean, orig_max, delta_mean, delta_max = self.select.train_step(all_x, all_y)
                self.count += 1
                return {'orig_mean':orig_mean, 'orig_max':orig_max, 'delta_mean':delta_mean, 'delta_max':delta_max}
            else:
                loss,memory1 = self.select.train_step(all_x, all_y)
                self.count += 1
                return {"loss": loss,"memory_allocated_GB": torch.cuda.max_memory_allocated() / (1024 ** 3),"memory1":memory1}
        self.count += 1
        return {'loss':0}    
        # loss = self.select.train_step(all_x, all_y)
       
    def add_gaussian_noise(self, all_x):
        """
        向输入数据的不同域添加不同强度的高斯噪声
        参数:
            all_x: 输入数据张量 (形状: [batch_size, ...])，batch_size=96
            self.noise_level: 基础噪声强度
        返回:
            加噪后的数据 (形状同 all_x)
        """
        if self.noise_level <= 0:
            return all_x

        # 确保 batch_size 可被 3 整除（3 个域）
        assert all_x.size(0) % 3 == 0, "Batch size must be divisible by 3 (3 domains)"
        samples_per_domain = all_x.size(0) // 3  # 每个域的样本数 = 32

        # 为每个域生成不同强度的噪声 (k=1, 2, 3)
        noisy_x = []
        for k in range(1, 4):
            domain_start = (k-1) * samples_per_domain
            domain_end = k * samples_per_domain
            domain_x = all_x[domain_start:domain_end]
            
            # 生成当前域的噪声 (强度 = noise_level * k)
            noise = torch.randn_like(domain_x) * (self.noise_level * k)
            noisy_domain_x = domain_x + noise
            noisy_domain_x = torch.clamp(noisy_domain_x, 0, 1)  # 限制范围
            
            noisy_x.append(noisy_domain_x)
        
        # 合并所有域
        noisy_x = torch.cat(noisy_x, dim=0)
        return noisy_x

    def predict(self, x):
        return self.select.predict(x)

class ERM_select_warm(Algorithm): # 用来看一眼到底warmup之后的损失
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        '''self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])'''
       
        # self.small_set_x = []
        # self.small_set_y = []
        self.selection_rate = hparams['selection_rate']
        self.warm_optimizer = torch.optim.Adam(self.classifier.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams['weight_decay'])
        arga["selection_rate"] = self.selection_rate
        arga['swa_start'] = hparams['swa_start']
        arga['swa_freq'] = hparams['swa_freq']
        arga['selected_layers'] = hparams['select_name']
        arga['way'] = hparams['way']
        arga['quantile_threshold'] = hparams['quantile_threshold']
        arga['batch_size'] = hparams['batch_size']
        self.hparams = hparams

        # self.k = self.hparams['k'] 
        if hparams['select'] == 'SPU3':
            self.select = SPU3(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU4':
            self.select = SPU4(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_sgd':
            self.select = SPU_sgd(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_min':
            self.select = SPU_min(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPUr':
            self.select = SPUr(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU5':
            self.select = SPU5(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU6':
            self.select = SPU6(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_kmeans':
            self.select = SPU_kmeans(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_x2':
            self.select = SPU_x2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_list':
            self.select = SPU_list(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step':
            self.select = SPU_step(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step2':
            self.select = SPU_step2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_stepg':
            self.select = SPU_stepg(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU2':
            self.select = SPU2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_random':
            self.select = SPU_step_random(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_du':
            self.select = SPU_step_du(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step_direct':
            self.select = SPU_step_direct(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)

        else:
            print('wrong method.')
            


        self.select_name =  hparams['select_name'] # 这个要查找一下
        self.count = 0
        self.set_count = hparams['set_count']
        self.recompute_count = hparams['recompute_count']
        self.per = hparams['per']
        self.k = hparams['k']
        self.name = hparams['select']
        self.warm = hparams['warm']
        self.small_set_x = [[] for _ in range(self.k)]
        # print(len(self.small_set_x)) 
        self.small_set_y = [[] for _ in range(self.k)] 

        # self.optimizer =  torch.optim.Adam(self.classifier.parameters(),lr = self.hparams['lr'],weight_decay = self.hparams['weight_decay'])
        '''for name, param in self.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")'''


    def update(self, x, y, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        all_x = torch.cat(x)
        # if self.k:
        #      all_x = random_swap_blocks_across_images(all_x,k=self.k)
        all_y = torch.cat(y)
        loss = 0
        memory1 = 0
        self.count += 1
        if self.count <= self.warm:
            loss = F.cross_entropy(self.network(all_x), all_y, reduction='none')  # 关键：reduction='none'，返回每个样本的loss

            # 2. 计算统计量
            max_loss = loss.max().item()  # 当前batch中单个样本的最大损失
            mean_loss = loss.mean().item()  # 当前batch的平均损失

            # 3. 原有训练流程保持不变
            self.warm_optimizer.zero_grad()
            loss.mean().backward()  # 注意：这里要用.mean()，因为loss现在是每个样本的loss
            self.warm_optimizer.step()

            # 4. 返回两个变量（max_loss, mean_loss）
            return {'max':max_loss, 'loss':mean_loss}
        else:
            loss = F.cross_entropy(self.network(all_x), all_y, reduction='none')  # 关键：reduction='none'，返回每个样本的loss
            # 2. 计算统计量
            max_loss = loss.max().item()  # 当前batch中单个样本的最大损失
            mean_loss = loss.mean().item()  # 当前batch的平均损失
            return {'max':max_loss, 'loss':mean_loss}



        # if self.count <= self.set_count: # 这里需要修正一下
        #     for i,(x_k,y_k) in enumerate(zip(x,y)):
        #         self.small_set_x[i].append(x_k)
        #         self.small_set_y[i].append(y_k)
        # if self.count == self.set_count:
        #     # 这里的loss可能要单独写
        #     # 比如一个类似map形式的loss
        #     self.select.compute_importance(self.small_set_x,self.small_set_y,self.select_name)
        #     if self.name == 'SPU_step':
        #         self.select.analyze_mask() # 看一下秩和值的关系
        #     if self.name == 'SPUr':
        #         self.select.refine_final_mask(self.small_set_x,self.small_set_y,self.select_name)
        # if self.count >  self.set_count: # 锁定之后再开始优化
        #     if self.recompute_count:
        #         if self.count % self.recompute_count== 0:
        #             # current_selection_rate = self.select.selection_rate
        #             # current_selection_rate = self.select.selection_rate * (1 + self.per * (self.count // self.recompute_count)) # 改成加法看一眼
        #             # self.select.selection_rate = max(current_selection_rate, 0.98)  # 确保不小于一个阈值
        #             self.select.compute_importance(self.small_set_x, self.small_set_y, self.select_name) # 重新计算
            
        # # 缩小更新的参数量
        #     if self.name == 'SPU_step_du':
        #         orig_mean, orig_max, delta_mean, delta_max = self.select.train_step(all_x, all_y)
        #         self.count += 1
        #         return {'orig_mean':orig_mean, 'orig_max':orig_max, 'delta_mean':delta_mean, 'delta_max':delta_max}
        #     else:
        #         loss,memory1 = self.select.train_step(all_x, all_y)
        #         self.count += 1
        #         return {"loss": loss,"memory_allocated_GB": torch.cuda.max_memory_allocated() / (1024 ** 3),"memory1":memory1}
        # return {'loss':0}    
        # loss = self.select.train_step(all_x, all_y)
       

    def predict(self, x):
        return self.select.predict(x)


class ERM_select2(Algorithm): # 线性层稍微修正一下,初始全为常数
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # 得重新构造一下
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        nn.init.constant_(self.classifier.weight, 1.0)  
        nn.init.constant_(self.classifier.bias, 1.0)

        self.selection_rate = hparams['selection_rate']
        arga["selection_rate"] = self.selection_rate
        arga['swa_start'] = hparams['swa_start']
        arga['swa_freq'] = hparams['swa_freq']
        arga['selected_layers'] = hparams['select_name']
        arga['way'] = hparams['way']
        arga['quantile_threshold'] = hparams['quantile_threshold']
        arga['batch_size'] = hparams['batch_size']

        # self.k = self.hparams['k'] 
        if hparams['select'] == 'SPU3':
            self.select = SPU3(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU4':
            self.select = SPU4(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_sgd':
            self.select = SPU_sgd(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_min':
            self.select = SPU_min(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPUr':
            self.select = SPUr(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU5':
            self.select = SPU5(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU6':
            self.select = SPU6(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_kmeans':
            self.select = SPU_kmeans(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_x2':
            self.select = SPU_x2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_list':
            self.select = SPU_list(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step':
            self.select = SPU_step(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_step2':
            self.select = SPU_step2(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)
        elif hparams['select'] == 'SPU_stepg':
            self.select = SPU_stepg(arga,self.featurizer,self.hparams['lr'],self.hparams['weight_decay'],self.classifier)

        else:
            print('wrong method.')
            


        self.select_name =  hparams['select_name'] # 这个要查找一下
        self.count = 0
        self.set_count = hparams['set_count']
        self.recompute_count = hparams['recompute_count']
        self.per = hparams['per']
        self.k = hparams['k']
        self.name = hparams['select']
        self.warm = hparams['warm']
        self.small_set_x = [[] for _ in range(self.k)]
        # print(len(self.small_set_x)) 
        self.small_set_y = [[] for _ in range(self.k)] 

        # self.optimizer =  torch.optim.Adam(self.classifier.parameters(),lr = self.hparams['lr'],weight_decay = self.hparams['weight_decay'])
        '''for name, param in self.network.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")'''


    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        # if self.k:
        #      all_x = random_swap_blocks_across_images(all_x,k=self.k)
        all_y = torch.cat(y)
        loss = 0
        # if self.count <= self.warm:

        #     loss = F.cross_entropy(self.network(all_x), all_y)
        #     self.warm_optimizer.zero_grad()
        #     loss.backward()
        #     self.warm_optimizer.step()
        #     loss = loss.item()

        if self.count <= self.set_count: # 这里需要修正一下
            for i,(x_k,y_k) in enumerate(zip(x,y)):
                self.small_set_x[i].append(x_k)
                self.small_set_y[i].append(y_k)
        if self.count == self.set_count:
            # 这里的loss可能要单独写
            # 比如一个类似map形式的loss
            self.select.compute_importance(self.small_set_x,self.small_set_y,self.select_name)
            if self.name == 'SPUr':
                self.select.refine_final_mask(self.small_set_x,self.small_set_y,self.select_name)
        if self.count >  self.set_count: # 锁定之后再开始优化
            if self.recompute_count:
                if self.count % self.recompute_count== 0:
                    current_selection_rate = self.select.selection_rate * (1 + self.per * (self.count // self.recompute_count)) # 改成加法看一眼
                    self.select.selection_rate = max(current_selection_rate, 0.98)  # 确保不小于一个阈值
                    self.select.compute_importance(self.small_set_x, self.small_set_y, self.select_name)
            
        # 缩小更新的参数量
            
            loss = self.select.train_step(all_x, all_y)

        # loss = self.select.train_step(all_x, all_y)


        self.count += 1

        return {"loss": loss}

    def predict(self, x):
        return self.select.predict(x)



def track_param_updates(model, param_name):
    # 获取指定参数
    param_before = None
    
    for name, param in model.named_parameters():
        # print(type(param_name))
        # print(type(name))
        if param_name in name :
            param_before = param.clone().detach()  # 克隆并分离出更新前的参数值
            break
    return param_before

# 计算参数更新
def compute_param_update(param_before, model, param_name):
    for name, param in model.named_parameters():
        if param_name in name :
            param_after = param.detach()  # 获取更新后的参数
            update = torch.norm(param_after - param_before).item()  # 计算 L2 范数来衡量更新量
            return update
    return None


from .new_z import custom_forward,custom_forward2
from functools import partial

from torchvision.models.resnet import Bottleneck
class ERM_repalce(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_repalce, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.learnable_params = []
        for name, module in self.featurizer.named_modules():
            # print(name)
            if isinstance(module, Bottleneck) and not isinstance(module, nn.Dropout):  # 假设你已经导入了Bottleneck类
                # print(name)
               
                if module.downsample is not None:
                    # 使用下采样的最后一层来获取输出通道数
                    out_channels = module.downsample[-2].out_channels
                else:
                    # 没有下采样，直接使用 conv1 的输出通道数
                    out_channels = module.conv1.in_channels
        

                learnable_param = nn.Parameter(torch.ones(out_channels),requires_grad=True)

                # 用自定义的 forward 方法重写
                module.forward = lambda x, mod=module, param=learnable_param: custom_forward(mod, x, param) 
                # module.forward = partial(custom_forward2, mod=module, param=learnable_param)

                self.learnable_params.append(learnable_param)
                    
            # 做一个简单替换 
        self.learnable_params_tensor = nn.ParameterList(self.learnable_params)
        self.learnable_params_tensor = self.learnable_params_tensor.to('cuda')
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer2 = torch.optim.SGD(
            self.learnable_params_tensor.parameters(),
            lr=50,
            weight_decay=self.hparams["weight_decay"],
        )
        
      

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer2.step()
        # i = 0
        # for param in self.learnable_params_tensor:
        #     if i == 0 :
        #         print(param)
        #         i += 1


        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)
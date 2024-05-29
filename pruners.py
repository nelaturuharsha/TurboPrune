import torch
import torch.nn as nn
from utils.conv_type import ConvMask, LinearMask

def prune_mag(model, density):
    score_list = {}
    for n, m in model.named_modules():
        # torch.cat([torch.flatten(v) for v in self.scores.values()])
        if isinstance(m, (ConvMask, LinearMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after magnitude pruning at current iteration = ', total_num / total_den)
    return model

def prune_random_erk(model, density):

    sparsity_list = []
    num_params_list = []
    total_params = 0
    score_list = {}

    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()

    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * density
    C = num_params_to_keep / num_params_kept
    print('Factor: ', C)
    sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]

    total_num = 0
    total_den = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            global_scores = torch.flatten(score_list[n])
            k = int((1 - sparsity_list[cnt]) * global_scores.numel())
            if k == 0:
                threshold = 0
            else: 
                threshold, _ = torch.kthvalue(global_scores, k)
            print('Layer', n, ' params ', k, global_scores.numel())

            score = score_list[n].to(m.weight.device)
            zero = torch.tensor([0.]).to(m.weight.device)
            one = torch.tensor([1.]).to(m.weight.device)
            m.mask = torch.where(score <= threshold, zero, one)
            total_num += (m.mask == 1).sum()
            total_den += m.mask.numel()
            cnt += 1

    print('Overall model density after random global (ERK) pruning at current iteration = ', total_num / total_den)
    return model

def prune_snip(model, trainloader, density):
    criterion = nn.CrossEntropyLoss()
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device('cuda'))
        target = target.to(torch.device('cuda')).long()
        model.zero_grad()
        output = model(images)
        criterion(output, target).backward()
        break

    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after snip pruning at current iteration = ', total_num / total_den)
    return model


def prune_synflow(model, trainloader, density):

    @torch.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for n, param in model.state_dict().items():
            param.mul_(signs[n])

    signs = linearize(model)

    # (data, _) = next(iter(trainloader))
    for i, (images, target) in enumerate(trainloader):
        images = images.to(torch.device('cuda'))
        target = target.to(torch.device('cuda')).long()
        input_dim = list(images[0,:].shape)
        input = torch.ones([1] + input_dim).to('cuda')#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        break

    score_list = {}
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            score_list[n] = (m.mask.to(m.weight.device) * m.weight.grad * m.weight).detach().abs_()

    model.zero_grad()

    nonlinearize(model, signs)

    global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
    k = int((1 - density) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    if not k < 1:
        total_num = 0
        total_den = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()

    print('Overall model density after synflow pruning at current iteration = ', total_num / total_den)
    return model

def prune_random_balanced(model, density):

        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                total_params += m.weight.numel()
                l += 1
        L = l
        X = density * total_params / l
        score_list = {}
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)
                    # correction for taking care of exact sparsity
                    diff = X - m.mask.numel()
                    X = X + diff / (L - l)
                l += 1

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask, LinearMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (balanced) pruning at current iteration = ', total_num / total_den)
        return model

def prune_er_erk(model, er_sparse_init):
    sparsity_list = []
    num_params_list = []
    total_params = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
            num_params_list.append(m.weight.numel())
            total_params += m.weight.numel()

    num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
    num_params_to_keep = total_params * er_sparse_init
    C = num_params_to_keep / num_params_kept
    sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)

def prune_er_balanced(model, er_sparse_init):
    total_params = 0
    l = 0
    sparsity_list = []
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            total_params += m.weight.numel()
            l += 1
    L = l
    X = er_sparse_init * total_params / l
    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            if X / m.weight.numel() < 1.0:
                sparsity_list.append(X / m.weight.numel())
            else: 
                sparsity_list.append(1)
                # correction for taking care of exact sparsity
                diff = X - m.mask.numel()
                X = X + diff / (L - l)
            l += 1

    l = 0
    for n, m in model.named_modules():
        if isinstance(m, (ConvMask, LinearMask)):
            m.set_er_mask(sparsity_list[l])
            l += 1
    print(sparsity_list)
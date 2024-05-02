import torch.nn as nn
import torch

LearnedBatchNorm = nn.BatchNorm2d

LearnedBatchNorm1d = nn.BatchNorm1d

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class NoRunningStatTrackBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NoRunningStatTrackBN, self).__init__(dim, track_running_stats = False)
        #momentum value of unity means x_new = (1 - momentum) * x_est + momentum * x_t
        #x_t is the observed value and x_est is the estimated statistic


class NoRunningStatBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NoRunningStatBN, self).__init__(dim, momentum=1)
        #momentum value of unity means x_new = (1 - momentum) * x_est + momentum * x_t
        #x_t is the observed value over the batch and x_est is the estimated statistic

class NoMomentumBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NoMomentumBN, self).__init__(dim, momentum=0)
        #momentum value of unity means x_new = (1 - momentum) * x_est + momentum * x_t
        # Hence here the running mean = previous running mean, so should be 1 always

class NonAffineNoStatBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatBatchNorm, self).__init__(dim, affine=False, momentum=1)


class CustomBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(CustomBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.curr_mean = None
        self.curr_var = None

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Update the current mean and variance
        self.curr_mean = mean
        self.curr_var = var

        return input

class CustomBNNonAffine(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CustomBNNonAffine, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.curr_mean = None
        self.curr_var = None

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Update the current mean and variance
        self.curr_mean = mean
        self.curr_var = var

        return input

class CustomBNNoRunningStatBN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=1,
                 affine=True, track_running_stats=True):
        super(CustomBNNoRunningStatBN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.curr_mean = None
        self.curr_var = None

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            self.curr_mean = mean
            self.curr_var = var 
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Update the current mean and variance
        

        return input
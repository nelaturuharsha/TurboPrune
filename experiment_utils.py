from utils.conv_type import ConvMask

def get_sparsity(model):
        # finds the current density of the model and returns the density scalar value
        nz = 0
        total = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                nz += m.mask.sum()
                total += m.mask.numel()
        
        return nz / total
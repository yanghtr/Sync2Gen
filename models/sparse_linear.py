import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import sys
import os

import itertools
import numpy as np
import random
import copy


def generate_pairs(numLayer, subset, nextLayer):
    # make sure len(allpairs) >= len(allpairs2)
    # otherwise del order will lead to order empty
    # if it is the case, when order is empty, set len(allpairs2) == 0 and continue 
    assert(numLayer > (subset + 2))

    result = []
    allpairs = list(itertools.combinations(range(numLayer), subset))
    allpairs2 = list(itertools.combinations(range(numLayer), 2))
    order = list(np.random.permutation(len(allpairs)))

    for i in range(nextLayer):
        if len(allpairs2) == 0:
            for j in order[0 : nextLayer - i]:
                result.append(list(allpairs[j]))
            break

        idx = np.random.randint(0, len(order)-1)
        select = allpairs[order[idx]]
        while count_hit(select, allpairs2) < 1:
            idx = np.random.randint(0, len(order)-1)
            select = allpairs[order[idx]]
        
        result.append(list(select))

        pairs4 = list(itertools.combinations(select, 2))
        for pair in pairs4:
            if pair in allpairs2:
                allpairs2.remove(pair)

        order.remove(order[idx])

    return torch.tensor(result)


def count_hit(select, pairs2) -> int:
    pairs4 = list(itertools.combinations(select, 2))
    for pair in pairs4:
        if pair in pairs2:
            return 1
    return 0


def generate_pairs_reverse(numCat, numRep, pre_graph):
    '''
    @Args:
        numCat: previously select sparse_num=4 from Hin, len(pre_graph) = C_{Hin}^{4}. To generate reversed 
                index, numCat = Hin
        numRep: number of repeat
        pre_graph: pairs generated from generate_pairs 
    @Returns:
        result: list of index tensor. Each index may have different len. len(result) = numCat * numRep
    '''
    result = []
    for i in range(numCat):
        cat_result = []
        for j in range(len(pre_graph)):
            if i in pre_graph[j]:
                cat_result.append(j)

        for _ in range(numRep):
            result.append(torch.tensor(cat_result))
    return result


class SparseLinear(Module):
    '''
    @Args:
        inputs: (B, Hin, Win), Hin can not be too large (e.g.200) since len(allpairs) = C_{Hin}^{4}. Good choice: 10/20/30.
    @Returns:
        outputs: (B, Hout, Wout), Hout = len(self.pairs) need to be computed manually
    '''
    def __init__(self, Hin: int, Win: int, Hout: int, Wout: int, sparse_num: int, pairs: list, bias: bool = True, 
                 is_decoder: bool = False, num_repeat: int = None) -> None:
        super(SparseLinear, self).__init__()
        self.Hin = Hin
        self.Win = Win
        self.Hout = Hout
        self.Wout = Wout
        self.sparse_num = sparse_num
        self.bias = bias
        self.is_decoder = is_decoder

        self.pairs = pairs
        
        self.weight_list = []
        self.bias_list = []
        for i in range(len(self.pairs)):
            if is_decoder:
                self.weight_list.append(Parameter(data=torch.Tensor(Wout, Win * len(self.pairs[i])), requires_grad=True))
            else:
                self.weight_list.append(Parameter(data=torch.Tensor(Wout, Win * sparse_num), requires_grad=True))
            self.register_parameter("weight" + str(i), self.weight_list[-1])

            if self.bias:
                self.bias_list.append(Parameter(data=torch.Tensor(Wout), requires_grad=True))
            else:
                self.bias_list.append(None)
            self.register_parameter("bias" + str(i), self.bias_list[-1])

        try:
            self.reset_parameters()
        except:
            from IPython import embed; embed()

    def reset_parameters(self) -> None:
        for i in range(len(self.weight_list)):
            init.kaiming_uniform_(self.weight_list[i], a=math.sqrt(5))

        if self.bias:
            for i in range(len(self.weight_list)):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_list[i])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_list[i], -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        S1: Select sparse_num tuple from Hin and get (B, sparse_num, Win),
        S2: Reshape to (B, sparse_num * Win)
        S3: Multiply weight(sparse_num * Win, Wout) and get (B, Wout)
        S4: Stack all together and get (B, Hout, Wout), Hout = num_pairs
        @Args:
            inputs: (B, Hin, Win)
        @Returns:
            outputs: (B, Hout, Wout), Hout = len(self.pairs) need to be computed manually
        '''
        B = inputs.shape[0] # batch size
        assert(len(self.pairs) == self.Hout)

        class_stack = []
        for i in range(len(self.pairs)):
            ndex = self.pairs[i].long().to(inputs.device)
            # Weight and bias don't need .to(device) since we have registered parameter
            layerdata = F.linear(torch.index_select(inputs, 1, ndex).reshape(B, -1), self.weight_list[i], self.bias_list[i]) # (B, Wout)

            class_stack.append(layerdata)
        
        return torch.stack(class_stack, 1)


    def extra_repr(self) -> str:
        return 'Hin={}, Win={}, Hout={}, Wout={}, bias={}'.format(
            self.Hin, self.Win, self.Hout, self.Wout, self.bias
        )




if __name__ == '__main__':

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.sc1 = SparseLinear(30, 40, 20, 30, 4)
            print('sc1 finish')
            self.sc2 = SparseLinear(20, 30, 80, 20, 4, bias=False)
            print('sc2 finish')

        def forward(self, x):
            x = self.sc1(x)
            x = self.sc2(x)
            return x

    model = Model()

    for name, param in model.named_parameters():
        print(name)
    print('-' * 30)

    for name, param in model.named_buffers():
        print(name)
    print('-' * 30)

    device = 'cuda:0'
    input_tensor = torch.randn(8, 30, 40).to(device)
    model.to(device)
    output_tensor = model(input_tensor)
    

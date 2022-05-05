from torch import nn
import torch as th

from ..invertible.actnorm import ActNorm
from ..invertible.affine import AffineCoefs, AffineModifier, AdditiveCoefs
from ..invertible.branching import ChunkChans, ChunkByIndices
from ..invertible.coupling import CouplingLayer
from ..invertible.distribution import Unlabeled, NClassIndependentDist
from ..invertible.graph import CatChansNode
from ..invertible.graph import Node, SelectNode, CatAsListNode
from ..invertible.graph import get_nodes_by_names
from ..invertible.identity import Identity
from ..invertible.inv_permute import InvPermute, Shuffle
from ..invertible.sequential import InvertibleSequential
from ..invertible.split_merge import ChunkChansIn2, EverySecondChan
from ..invertible.splitter import SubsampleSplitter
from ..invertible.view_as import Flatten2d, ViewAs


def convert_glow_to_pre_dist_model(model):
    model_log_act_nodes = get_nodes_by_names(
        model, 'm0-act-0', 'm0-act-1', 'm0-act-2')
    for a in model_log_act_nodes:
        a.next = []
    model_log_det_node = CatChansNode(
        model_log_act_nodes,
        notify_prev_nodes=True)
    return model_log_det_node


def split_glow_into_pre_dist_and_dist(model):
    # remove references to previous dist node
    model_log_act_nodes = get_nodes_by_names(
        model, 'm0-act-0', 'm0-act-1', 'm0-act-2')
    for a in model_log_act_nodes:
        a.next = []
    model_log_det_node = CatChansNode(
        model_log_act_nodes,
        notify_prev_nodes=True)

    model_dist_nodes = get_nodes_by_names(model, 'm0-dist-0',
                                          'm0-dist-1',
                                          'm0-dist-2')
    rechunker = ChunkByIndices((6 * 16 * 16, 6 * 16 * 16 + 48 * 4 * 4))
    nd_in_split_for_dist = Node(None, rechunker)
    dist_node = CatChansNode(
        [Node(SelectNode(nd_in_split_for_dist, i),
              model_dist_nodes[i].module)
         for i in range(len(model_dist_nodes))])
    return model_log_det_node, dist_node


def create_glow_model(
        hidden_channels,
        K,
        L,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        n_chans,
        block_type='conv',
        use_act_norm=True,
        image_size=32,
):
    image_shape = (image_size, image_size, n_chans)

    H, W, C = image_shape
    flows_per_scale = []
    act_norms_per_scale = []
    dists_per_scale = []
    for i in range(L):
        C, H, W = C * 4, H // 2, W // 2

        splitter = SubsampleSplitter(
            2, via_reshape=True, chunk_chans_first=True, checkerboard=False,
            cat_at_end=True)

        if block_type == 'dense':
            pre_flow_layers = [Flatten2d()]
            in_channels = C * H * W
        else:
            assert block_type == 'conv'
            pre_flow_layers = []
            in_channels = C

        flow_layers = [flow_block(in_channels=in_channels,
                                  hidden_channels=hidden_channels,
                                  flow_permutation=flow_permutation,
                                  flow_coupling=flow_coupling,
                                  LU_decomposed=LU_decomposed,
                                  cond_channels=0,
                                  cond_merger=None,
                                  block_type=block_type,
                                  use_act_norm=use_act_norm) for _ in range(K)]

        if block_type == 'dense':
            post_flow_layers = [ViewAs((-1, C * H * W), (-1, C, H, W))]
        else:
            assert block_type == 'conv'
            post_flow_layers = []
        flow_layers = pre_flow_layers + flow_layers + post_flow_layers
        flow_this_scale = InvertibleSequential(splitter, *flow_layers)
        flows_per_scale.append(flow_this_scale)

        if i < L - 1:
            # there will be a chunking here
            C = C // 2
        # act norms for distribution (mean/std as actnorm isntead of integrated
        # into dist)
        act_norms_per_scale.append(InvertibleSequential(Flatten2d(),
                                                        ActNorm((C * H * W),
                                                                scale_fn='exp')))
        dists_per_scale.append(Unlabeled(
            NClassIndependentDist(1, C * H * W, optimize_mean_std=False)))


    dist_nodes = []
    nd_cur = None
    for i in range(L):
        if i > 0:
            nd_cur = SelectNode(nd_cur, 1, name=f'm0-in-flow-{i}')
        nd_cur = Node(nd_cur, flows_per_scale[i], name=f'm0-flow-{i}')
        if i < (L - 1):
            nd_cur = Node(nd_cur, ChunkChans(2), name=f'm0-flow-{i}')
            nd_cur_out = SelectNode(nd_cur, 0)
        else:
            # at last scale, there is no further splitting off of dimensions
            nd_cur_out = nd_cur
        nd_cur_act = Node(nd_cur_out, act_norms_per_scale[i], name=f'm0-act-{i}')
        nd_cur_dist = Node(nd_cur_act, dists_per_scale[i], name=f'm0-dist-{i}')
        dist_nodes.append(nd_cur_dist)

    model = CatAsListNode(dist_nodes, name='m0-full')
    return model


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(th.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * th.exp(self.logs * self.logscale_factor)



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", do_actnorm=True, weight_std=0.05):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=(not do_actnorm))

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm(out_channels, scale_fn='exp', eps=0)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


def get_conv_block(in_channels, out_channels, hidden_channels, nonlin_name):
    assert nonlin_name in ['elu', 'relu']
    nonlin = {'elu': nn.ELU(inplace=False),
         'relu': nn.ReLU(inplace=False)}[nonlin_name]
    block = nn.Sequential(Conv2d(in_channels, hidden_channels),
                          nonlin,
                          Conv2d(hidden_channels, hidden_channels,
                                 kernel_size=(1, 1)),
                          nonlin,
                          Conv2dZeros(hidden_channels, out_channels))
    return block


def get_dense_block(in_channels, out_channels, hidden_channels, nonlin_name):
    nonlin = {'elu': nn.ELU(inplace=False),
         'relu': nn.ReLU(inplace=False)}[nonlin_name]
    block = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                          nonlin,
                          nn.Linear(hidden_channels, hidden_channels),
                          nonlin,
                          nn.Linear(hidden_channels, out_channels))
    return block


def flow_block(in_channels, hidden_channels,
                 flow_permutation, flow_coupling, LU_decomposed,
               cond_channels, cond_merger, block_type, use_act_norm,
               nonlin_name='relu'):
    if use_act_norm:
        actnorm = ActNorm(in_channels, scale_fn='exp', eps=0)
    # 2. permute
    if flow_permutation == "invconv":
        flow_permutation = InvPermute(
            in_channels, fixed=False, use_lu=LU_decomposed)
    elif flow_permutation == 'invconvfixed':
        flow_permutation = InvPermute(in_channels,
                                      fixed=True,
                                             use_lu=LU_decomposed)
    elif flow_permutation == "identity":
        flow_permutation = Identity()
    else:
        assert flow_permutation == 'shuffle'
        flow_permutation = Shuffle(in_channels)

    if flow_coupling == "additive":
        out_channels = in_channels // 2
    else:
        out_channels = in_channels

    if type(block_type) is str:
        if block_type == 'conv':
            block_fn = get_conv_block
        else:
            assert block_type == 'dense'
            block_fn = get_dense_block
    else:
        block_fn = block_type

    block = block_fn(in_channels // 2 + cond_channels,
                          out_channels,
                          hidden_channels,
                     nonlin_name=nonlin_name)

    if flow_coupling == "additive":
        coupling = CouplingLayer(
            ChunkChansIn2(swap_dims=True),
            AdditiveCoefs(block,),
            AffineModifier(sigmoid_or_exp_scale='sigmoid',
                            eps=0,  add_first=True, ),
            condition_merger=cond_merger
        )
    elif flow_coupling == "affine":
        coupling = CouplingLayer(
            ChunkChansIn2(swap_dims=True),
            AffineCoefs(block, EverySecondChan()),
            AffineModifier(sigmoid_or_exp_scale='sigmoid', eps=0,  add_first=True),
            condition_merger=cond_merger,
        )
    else:
        assert False, f"unknown flow_coupling {flow_coupling}"
    if use_act_norm:
        sequential = InvertibleSequential(actnorm, flow_permutation, coupling)
    else:
        sequential = InvertibleSequential(flow_permutation, coupling)
    return sequential



def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]
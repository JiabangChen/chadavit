"""
ChAda-ViT (i.e Channel Adaptive ViT) is a variant of ViT that can handle multi-channel images.
"""
import math
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from src.utils.misc import trunc_normal_

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(Module):
    r"""
    Mostly copied from torch.nn.TransformerEncoderLayer, but with the following changes:
    - Added the possibility to retrieve the attention weights
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True, # jiabang's change
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # MultiheadAttention是pytorch的内置类，用来实现整个attn模块，即从输入（N+1,D）到输出（N+1,D），里面的project到qkv，然后reshape
        # 成每个head的qkv，然后各自求A，求新v，再相互级联然后project到输出维度都在里面自己完成了
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        # 上面的两个线性层是MLP
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # 这里的**factory_kwargs是一种更加精细化的操作，即是否把每一个子模块放到GPU上，并且控制模块的数据类型（精度）
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, return_attention = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required). 是（B，1960+1，192）即输入的z
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional). 是（B，1960+1）即表示输入的batch中每个sample哪些
            patch是padding的。如果是padding那么就是True

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first: # jiabang's change，我在init初始化函数中，我把这个改成了true
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x #输出与输入的x形状一致（B,1960+1,D）

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], return_attention: bool = False) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=return_attention,
                            average_attn_weights=False)
        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TokenLearner(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) # 每一个channel可以割出多少个patch
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 这里是对每个channel中的每个patch做patch embedding，这里的patch的channel只有1，与普通vit不一样
    def forward(self, x): # 输入的x的shape是（X*num_channels,1,H,W）
        x = self.proj(x) # patch embdedding, 对batch中每个sample的每个channel中的每个patch做embedding，结果形状是（X*num_channels，192，14，14））
        x = x.flatten(2) # 把后面两个展开成一个长向量，变成（X*num_channels，192，196）
        x = x.transpose(1, 2)# shape转为（X*num_channels，196，192），即(B,N,D)
        return x

class ChAdaViT(nn.Module):
    """ Channel Adaptive Vision Transformer"""
    def __init__(self, img_size=[224], in_chans=1, embed_dim=192, patch_size=16, num_classes=0, depth=12,
                 num_heads=12, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, return_all_tokens=True, max_number_channels=10, **kwargs):
        super().__init__()

        # Embeddings dimension
        self.num_features = self.embed_dim = embed_dim # token的维数

        # Num of maximum channels in the batch
        self.max_channels = max_number_channels # 一个batch中可能不同的图片其channel数量不一样，这是一个
        # batch中允许出现的最大channel

        # Tokenization module
        self.token_learner = TokenLearner(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim)
        # token_leaner是一个TokenLearner类，这个类中有一个叫做proj的卷积，是nn.Module的子类，因此在print(model)中会被打印出来，且会标明是在
        # TokenLearner这个类中，而且这个Conv的名字应该是token_learner.proj而不是proj，因为它不是在Chadavit的最外层。
        num_patches = self.token_learner.num_patches # 应该是一个channel可以分几个patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # (B, max_channels * num_tokens, embed_dim)
        # 这里之所以前面要加上1，1两个维度是因为第一个维度是Batch size,第二个维度是token number（N），这是为了按batch输入时更好地与batch中每个sample，
        # 与单个sample中整个（N,D）token矩阵拼接，如果只有（，embed_dim）无法与输入拼接（无法广播），可以理解为第一个1是batch数量，第二个1是这个cls_token的数量
        # 拼接时，第一个1，即batch size expand为B，变成（B,1,D），然后与输入batch中每一个（N,D）级联，因此必须要在前面有两个维度，否则无法expand到Batch
        # 中每一个sample，也无法在N这个维度上与输入级联
        # 但因为它最后是与一个（B,N（N = max_channels * num_tokens）,D）矩阵拼接，因此维持三维就好，不需要考虑channel数量
        self.channel_token = nn.Parameter(torch.zeros(1, self.max_channels, 1, self.embed_dim)) # (B, max_channels, 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches + 1, self.embed_dim)) # (B, max_channels, num_tokens + 1, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate) # 一般不dropout 因此这里的p=0
        # cls_token, channel_token, pos_embed，都是nn.Parameter，因此模型print中不会出现，但是他们也是可训练的参数，因此会出现在预训练的参数字典中，
        # 如果想要正确加载，那么这几个变量名字得与参数字典中的名字一样
        # TransformerEncoder block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # 这个意思是设置一个dropout的概率列表，一共有12个元素，从0开始，等间距分布到第十二个元素（即drop_path_rate为目标），这些概率代表了dropout整个
        # block的概率，当然这里都是0.
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=dpr[i], batch_first=True)
            for i in range(depth) # 建立十个transformer block
        ])
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity() # DINO训练时直接是identity

        # Return only the [CLS] token or all tokens
        self.return_all_tokens = return_all_tokens

        trunc_normal_(self.pos_embed, std=.02) # 用标准差为 0.02 均值为0的截断正态分布初始化位置编码参数 pos_embed，确保其值在合理范围内
        # 这个截断指的是所有值分布在-2std,2std之间，是在截断区间内重新采样，但它仍然是一个正态分布
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.channel_token, std=.02)
        self.apply(self._init_weights) # 对当前模块 self 及其所有子模块递归调用self._init_weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): # 如果该模块属于nn.Linear类
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): # 如果该模块属于nn.LayerNorm类
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_pos_encoding_per_channel(self, x, w, h, class_pos_embed:bool=False):
        """
        Adds num_patches positional embeddings to EACH of the channels.
        """
        npatch = x.shape[2] # 每个channel中有几个patch
        N = (self.pos_embed.shape[2] - 1) # 这个N也是每个channel中有几个patch

        # --------------------- [CLS] positional encoding --------------------- #
        if class_pos_embed: # class_pos_embed=False，暂时先不赋予CLS token的position embedding
            return self.pos_embed[:, :, 0]

        # --------------------- Patches positional encoding --------------------- #
        # If the input size is the same as the training size, return the positional embeddings for the desired type
        if npatch == N and w == h:
            # 输入图和预训练图大小应该都一致，因此输出position embedding，并且是去除了第一行，即cls的pos_embedding
            # 其对batch中每一个sample，一个sample中每一个channel都是一样的
            return self.pos_embed[:, :, 1:]

        # Otherwise, interpolate the positional encoding for the input tokens
        class_pos_embed = self.pos_embed[:, :, 0]
        patch_pos_embed = self.pos_embed[:, :, 1:]
        dim = x.shape[-1]
        w0 = w // self.token_learner.patch_size
        h0 = h // self.token_learner.patch_size
        # a small number is added by DINO team to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.unsqueeze(0)

    def channel_aware_tokenization(self, x, index, list_num_channels, max_channels=3):
        # jiabang's change, set the max_channel = 3
        B, nc, w, h = x.shape    # (B*num_channels, 1, w, h)

        # Tokenize through linear embedding
        tokens_per_channel = self.token_learner(x) # 输出shape为（X*num_channels，196，192），即(B,N,D)

        # Concatenate tokens per channel in each image
        chunks = torch.split(tokens_per_channel, list_num_channels[index], dim=0) # List of (img_channels, embed_dim)
        # 由于X*num_channels中对应了batch中的每一个sample，不同的sample可能拥有的channel不一样，list_num_channels记录了不同的sample含有的通道数
        # index应该是这个batch的索引，即第几个list_num_channels是此batch的内容，一般为0。这里的意思就是把tokens_per_channel，按照第0维，按照
        # list_num_channels[index]的划分方案（即每个image有几个channel），划分成若干份，每一份代表原始的每一个sample，其实就是把原来batch中
        # 的每一个sample沿着channel摊开，然后做patch embedding，然后再归属给每一个sample，把通道对回每张图像，相当于每张图像所能分到的tokens_per_channel
        # 就是chunks中的每一个元素，其形状为（C，196，192），就是（C,N,D），C为每一个sample原有的channel数量

        # Pad the tokens tensor with zeros for each image separately in the chunks list
        padded_tokens = [torch.cat([chunk, torch.zeros((max_channels - chunk.size(0), chunk.size(1), chunk.size(2)), device=chunk.device)], dim=0) if chunk.size(0) < max_channels else chunk for chunk in chunks]
        # 对chunks中的每一个元素，就是每张图的（C,N,D）token矩阵（一般vit应该是（N,D），这里其实是用C的增长换了D的减小），如果C小于10，
        # 即这个图的channel数小于10，那么就给他级联（c,196,192）,c=10-C，使得每一个输入都有10个channel的patch embedding，级联部分的
        # patch embedding值是全0。注意！这里的device=chunk.device是必须的，因为torch.zeros定义的是一个普通tensor，
        # 如果外面model.to(device)，或者用factory_kwargs,会让chunk加载到gpu上，但是这个全零的普通tensor不会这样在forward中做这个级联
        # 操作会因为不在同一个设备上而报错

        # Stack along the batch dimension
        padded_tokens = torch.stack(padded_tokens, dim=0) # (B, img_channels, num_tokens, embed_dim)
        # 这里是把padden_tokens这个列表中的各个元素，即每一个sample的各个channel的patch embedding级联后再与全0级联，补成（10，196,192），沿着第零维
        # 堆叠起来，形成一个新的tensor，形状为(B, 10, 196, 192)，stack不是级联，因此它会在前面多出一个维度B，就是真的
        # batch_size,上文提到的X*num_channels是假的batch_size，只是这个batch中所有channel的总数。

        num_tokens = padded_tokens.size(2) # 196，不算cls_token

        # Reshape the patches embeddings on the channel dimension
        padded_tokens = padded_tokens.reshape(padded_tokens.size(0), -1, padded_tokens.size(3)) # (B, max_channels*num_tokens, embed_dim)
        # padded_tokens reshape成(B, 1960, 192),因为chadavit做attn是用每一个channel的patch embedding所flatten成的一个大的向量作为输入的，
        # 这样才能形成inter和intra attention，因此这里channel数（已经全部padding成10）x每一个channel中的patch数就是在做这个flatten

        # Compute the masking for avoiding self-attention on empty padded channels => CRUCIAL to perform this operation here, before having added the [POS] and [CHANNEL] tokens
        channel_mask = torch.all(padded_tokens == 0., dim=-1)  # Check if all elements in the last dimension are zeros (indicating a padded channel)
        # 形成一个（B，1960）的mask，告诉model，在batch中的某一个sample的某一个patch是否为全0，全0就是True，说明这个patch是padding的，否则为false

        # Destack to obtain the original number of channels
        padded_tokens = padded_tokens.reshape(-1, max_channels, num_tokens, padded_tokens.size(-1)) # (B, img_channels, num_tokens, embed_dim)
        # 恢复成原来的（B，10，196，192）形状

        # Add the [POS] token to the embed patch tokens
        padded_tokens = padded_tokens + self.add_pos_encoding_per_channel(padded_tokens, w, h, class_pos_embed=False)
        # 把输入的（B,10,196,192）patch embedding 加上position embedding（1，1，196，192），用广播机制

        # Add the [CHANNEL] token to the embed patch tokens
        if max_channels == self.max_channels:
            channel_tokens = self.channel_token.expand(padded_tokens.shape[0],-1,padded_tokens.shape[2],-1) # (1, max_channels, 1, embed_dim) -> (B, max_channels, num_tokens, embed_dim)
            # channel_token的形状是（1，10，1，192），这里也是广播成patch embedding的形状（B，10，196，192）
            padded_tokens = padded_tokens + channel_tokens  # Add a different channel_token to the ALL the patches of a same channel
            # 某个channel embedding对batch中的每一个sample，每个sample在那一个channel上的每一个patch，都是一样的

        ########################### Sanity Check ###########################
        # self.channel_token_sanity_check(channel_tokens)

        # Restack the patches embeddings on the channel dimension
        embeddings = padded_tokens.reshape(padded_tokens.size(0), -1, padded_tokens.size(3)) # (B, max_channels*num_tokens, embed_dim)
        # padded_tokens reshape成(B, 1960, 192),因为chadavit做attn是用每一个channel的patch embedding所flatten成的一个大的向量作为输入的，
        # 这样才能形成inter和intra attention，因此这里channel数（已经全部padding成10）x每一个channel中的patch数就是在做这个flatten

        # Expand the [CLS] token to the batch dimension
        cls_tokens = self.cls_token.expand(embeddings.shape[0],-1,-1) # (1, 1, embed_dim) -> (B, 1, embed_dim)

        # Add [POS] positional encoding to the [CLS] token
        cls_tokens = cls_tokens + self.add_pos_encoding_per_channel(embeddings, w, h, class_pos_embed=True)
        # 这里是把pos_embed中的第一行赋给cls_tokens做positional embedding，但它没有channel embedding

        # Concatenate the [CLS] token to the embed patch tokens
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # Append cls_token to the beginning of each image
        # 按照第一个维度把cls_tokens和embeddings级联，形成（B,1960+1,D）

        # Adding a False value to the beginning of each channel_mask to account for the [CLS] token
        channel_mask = torch.cat([torch.tensor([False], device=channel_mask.device).expand(channel_mask.size(0),1), channel_mask], dim=1)
        # 初始channel_mask是（B，1960），由于cls_token不需要mask，因此其值为False，然后从原本形状为（1，），expand为（B，1），再与原channel_mask
        # 级联，形成的形状为（B,1960+1）;级联时，若按照第1维级联，那么其他维的维度大小要一致

        return self.pos_drop(embeddings), channel_mask

    def forward(self, x, index, list_num_channels):
        # Apply the TokenLearner module to obtain learnable tokens
        x, channel_mask = self.channel_aware_tokenization(x, index, list_num_channels) # (B*num_channels, embed_dim)

        # Apply the self-attention layers with masked self-attention
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=channel_mask)  # Use src_key_padding_mask to mask out padded tokens
        # 这个的输出也应该是（B,1960+1,D）
        # Normalize
        x = self.norm(x) # 最后的输出会再做一次layernorm，这个的输出也应该是（B,1960+1,D）

        if self.return_all_tokens:
            # Create a mask to select non-masked tokens (excluding CLS token)
            non_masked_tokens_mask = ~channel_mask[:, 1:]
            non_masked_tokens = x[:, 1:][non_masked_tokens_mask]
            return non_masked_tokens  # return non-masked tokens (excluding CLS token)
        else:
            return x[:, 0]  # return only the [CLS] token，这个的形状是（B,D）

    def channel_token_sanity_check(self, x):
        """
        Helper function to check consistency of channel tokens.
        """
        # 1. Compare Patches Across Different Channels
        print("Values for the first patch across different channels:")
        for ch in range(10):  # Assuming 10 channels
            print(f"Channel {ch + 1}:", x[0, ch, 0, :5])  # Print first 5 values of the embedding for brevity

        print("\n")

        # 2. Compare Patches Within the Same Channel
        for ch in range(10):
            is_same = torch.all(x[0, ch, 0] == x[0, ch, 1])
            print(f"First and second patch embeddings are the same for Channel {ch + 1}: {is_same.item()}")

        # 3. Check Consistency Across Batch
        print("Checking consistency of channel tokens across the batch:")
        for ch in range(10):
            is_consistent = torch.all(x[0, ch, 0] == x[1, ch, 0])
            print(f"Channel token for first patch is consistent between first and second image for Channel {ch + 1}: {is_consistent.item()}")

    def get_last_selfattention(self, x):
        x, channel_mask = self.channel_aware_tokenization(x, index=0, list_num_channels=[1], max_channels=1)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, src_key_padding_mask=channel_mask)
            else:
                # return attention of the last block
                return blk(x, src_key_padding_mask=channel_mask, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x, channel_mask = self.channel_aware_tokenization(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, src_key_padding_mask=channel_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def chada_vit(**kwargs):
    patch_size = kwargs['patch_size']
    embed_dim = kwargs['embed_dim']
    return_all_tokens = kwargs['return_all_tokens']
    max_number_channels = kwargs['max_number_channels']
    model = ChAdaViT(patch_size=patch_size, embed_dim=embed_dim, depth=12, num_heads=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=return_all_tokens, max_number_channels=max_number_channels)
    return model

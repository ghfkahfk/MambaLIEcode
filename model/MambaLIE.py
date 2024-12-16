
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from mamba_ssm.modules.mamba_new import Mamba
import einops
from einops import rearrange
import numpy as np

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Light(nn.Module):

    def __init__(self, fea_middle, fea_in=4, fea_out=3):
      
        super(Light, self).__init__()

        self.depth_conv = nn.Conv2d(
            fea_in, fea_middle, kernel_size=5, padding=2, bias=True, groups=fea_in)
        self.conv1 = nn.Conv2d(fea_middle, fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        
        x1 = self.depth_conv(img)
        fea = x1
        map = self.conv1(x1)
    
        return fea, map



###### Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x

###### ChannelAttention
class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
    
######  Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim//2, LayerNorm_type)
        self.attn1 = Mamba(dim//2)
        self.att2=ChannelAttention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.convbranch= nn.Sequential(
            nn.BatchNorm2d(dim // 2),
            nn.Conv2d(in_channels=dim//2,out_channels=dim//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
    def forward(self, x):
        input_left, input_right = x.chunk(2,dim=1)
        
        
        input_right= self.norm1(input_right)
        bs, cn, h, w =  input_right.shape
        input_right =  input_right.reshape((bs, h*w, cn))
        input_right2 = (self.attn1(input_right)).permute(0,2,1).reshape((bs, cn, h, w))

        input_left=self.norm1(input_left)
        input_left = self.convbranch(input_left)
        input_left = input_left.permute(0,2,3,1).contiguous()
        # print(input_left.shape)
        # print(input_right2.shape)
        input_left=input_left.permute(0, 3, 1, 2)
        output = torch.cat((input_left,input_right2),dim=1)
        x=output+x
        x=self.norm2(x)
        x=x+self.att2(x)
        x = x + self.ffn(self.norm2(x))
        return x




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out

##########################################################################
##---------- MambaLIE-----------------------
class MambaLIE(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 16,
        num_blocks = [1,2,4,8],
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        attention=True,
        skip = False,
        feat = 40
    ):

        super(MambaLIE, self).__init__()

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((4, 2,int(int(dim * 2 * 4))))),
                                        requires_grad=attention)
        
        self.light =Light(feat)
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.encoder_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.encoder_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.layer_fussion = LAM_Module_v2(in_dim=int(dim*3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_1 = Downsample(int(dim)) ## From Level 0 to Level 1

        self.decoder_level1_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down_2 = Downsample(int(dim *2)) ## From Level 1 to Level 2
        self.decoder_level2_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down_3 = Downsample(int(dim * 2*2)) ## From Level 2 to Level 3
        self.decoder_level3_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down_4 = Downsample(int(dim * 2 * 4)) ## From Level 3 to Level 4
        self.decoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 8)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 *8))  ## From Level 4 to Level 3
        self.decoder_level3_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 *4)) ## From Level 3 to Level 2
        self.decoder_level2_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 *2)) ## From Level 2 to Level 1
        self.decoder_level1_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.up2_0 = Upsample(int(dim * 2))  ## From Level 1 to Level 0
        ### skip connection wit weights
        self.coefficient_4_3 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 4))))), requires_grad=attention)
        self.coefficient_3_2 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 2))))), requires_grad=attention)
        self.coefficient_2_1 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2))))), requires_grad=attention)
        self.coefficient_1_0 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim))))), requires_grad=attention)

        ### skip then conv 1x1
        self.skip_4_3 = nn.Conv2d(int(int(dim * 2 * 4)), int(int(dim * 2 * 4)), kernel_size=1, bias=bias)
        self.skip_3_2 = nn.Conv2d(int(int(dim * 2 * 2)), int(int(dim * 2 * 2)), kernel_size=1, bias=bias)
        self.skip_2_1 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)
        self.skip_1_0 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement_1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.layer_fussion_2 = LAM_Module_v2(in_dim=int(dim*3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip


    def forward(self, inp_img):

        fea, map = self.light(inp_img)
        input_img = inp_img * map + inp_img
        
        inp_enc_encoder1 = self.patch_embed(input_img)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)


        inp_fusion_123 = torch.cat([out_enc_encoder1.unsqueeze(1),out_enc_encoder2.unsqueeze(1),out_enc_encoder3.unsqueeze(1)],dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)


        inp_enc_level1_0 = self.down_1(out_fusion_123)
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)



        inp_enc_level2_0 = self.down_2(out_enc_level1_0)
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)



        inp_enc_level3_0 = self.down_3(out_enc_level2_0)
        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)



        inp_enc_level4_0 =   self.down_4(out_enc_level3_0)
        out_enc_level4_0 = self.decoder_level4(inp_enc_level4_0)


        out_enc_level4_0 = self.up4_3(out_enc_level4_0)

        inp_enc_level3_1 = self.coefficient_4_3[0, :][None, :, None, None] * out_enc_level3_0 + self.coefficient_4_3[1, :][None, :, None, None] * out_enc_level4_0
        inp_enc_level3_1 = self.skip_4_3(inp_enc_level3_1)  ### conv 1x1


        out_enc_level3_1 = self.decoder_level3_1(inp_enc_level3_1)


        out_enc_level3_1 = self.up3_2(out_enc_level3_1)
        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0 + self.coefficient_3_2[1, :][None, :, None, None] * out_enc_level3_1
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)  ### conv 1x1


        out_enc_level2_1 = self.decoder_level2_1(inp_enc_level2_1)

        out_enc_level2_1 = self.up2_1(out_enc_level2_1)

        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_enc_level1_0 + self.coefficient_2_1[1, :][None, :, None, None] *  out_enc_level2_1

        inp_enc_level1_1 = self.skip_1_0(inp_enc_level1_1)  ### conv 1x1

        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)

        out_enc_level1_1 = self.up2_0(out_enc_level1_1)


        out_fusion_123 = self.latent(out_fusion_123)


        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123  + self.coefficient_1_0[1, :][None, :, None, None] *  out_enc_level1_1

        out_1 = self.refinement_1(out)

        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat([out_1.unsqueeze(1),out_2.unsqueeze(1),out_3.unsqueeze(1)],dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)

        if self.skip:
            out = self.output(out)+ inp_img
        else:
            out = self.output(out)

        return out


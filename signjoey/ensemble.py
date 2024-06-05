import numpy as np
import torch
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, TransformerEncoder
from signjoey.decoders import Decoder, TransformerDecoder
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict




#construct new ensemble ckpt 
def BuildEnsebleCKPT(ckpts): 
    
    o1,n1='encoder','encoder.Encoders'
    o2,n2='decoder','decoder.Decoders'
    o3,n3='sgn_embed','sgn_embed.SpatialEmbeddings'
    o4,n4='txt_embed','txt_embed.Embeddings'
    o5,n5='gloss', 'gloss.Glosses'
    
    cs = [ torch.load(ckpt,map_location=torch.device('cpu')) for ckpt in  ckpts]
    checkpoint = torch.load(ckpts[0],map_location=torch.device('cpu'))
    layers=list(cs[0]['model_state']) 
    
    for i in range(len(ckpts)):
        for layer in layers:
            Is='.'+str(i)
            
            if  o1 in  layer :
                checkpoint['model_state'][layer.replace(o1,n1+Is)]=cs[i]['model_state'][layer]
            elif  o2 in  layer :
                checkpoint['model_state'][layer.replace(o2,n2+Is)]=cs[i]['model_state'][layer]
                              
            elif  o3 in  layer :
                checkpoint['model_state'][layer.replace(o3,n3+Is)]=cs[i]['model_state'][layer]
                                
            elif  o4 in  layer :
                checkpoint['model_state'][layer.replace(o4,n4+Is)]=cs[i]['model_state'][layer]

            elif o5 in layer:
                checkpoint['model_state'][layer.replace(o5,n5+Is)]=cs[i]['model_state'][layer]
                    
            if layer not in ["txt_embed.lut.posterior_mean","txt_embed.lut.posterior_un_scale","txt_embed.lut.bias_mean",
                            "txt_embed.lut.bias_un_scale","txt_embed.lut.weight"] and layer in checkpoint['model_state'] :
                    del checkpoint['model_state'][layer]
                
         
    return checkpoint

#############################################
# Wrappers on Endoder Decoder and Embedings #
#############################################

class EnsembleTransformerDecoder(Decoder):
    def __init__(self,*args, **kwargs):
        super(EnsembleTransformerDecoder, self).__init__()
        
        self.N=kwargs['N']
        self.Decoders=[TransformerDecoder(*args, **kwargs) for i in range(self.N) ]
        self.Decoders=nn.ModuleList(self.Decoders)
        for param in self.Decoders[0].parameters():
            param.requires_grad = False
        self._output_size = self.Decoders[0].output_size
    
    def forward(
        self,
        trg_embed  = None,
        encoder_output = None,
        encoder_hidden: Tensor = None,
        src_mask: Tensor = None,
        unroll_steps: int = None,
        hidden: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs
    ):
        if self.training:
            return self.Decoders[0](trg_embed[0],encoder_output,
                                                 encoder_hidden,src_mask,unroll_steps,hidden,trg_mask,**kwargs)
       
        out1,out2=0,0
        for i in range(self.N):
           
            
            out1_, out2_, _, _=self.Decoders[i](trg_embed[i],encoder_output[...,i],
                                               encoder_hidden,src_mask,unroll_steps,hidden,trg_mask,**kwargs)
            
            out2 = out2+out2_
            out1 = out1+out1_
            
        out1, out2 = out1/self.N,out2/self.N
        return out1, out2, None, None
    
    
class EnsembleTransformerEncoder(Encoder):
    def __init__(self,*args, **kwargs):
        super(EnsembleTransformerEncoder, self).__init__()
       
        self.N=kwargs['N']
        self.Encoders=[TransformerEncoder(*args, **kwargs) for i in range(self.N) ]
        self.Encoders=nn.ModuleList(self.Encoders)
        self._output_size = kwargs['hidden_size']
        for param in self.Encoders[0].parameters():
            param.requires_grad = False
        
    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor, num_token: list = None, breakpoint: list = None):
         if self.training :
            return self.Encoders[0](embed_src[0],src_length,mask, num_token, breakpoint), None
        
            
         out1=[]
         for i in range(self.N):
            x_, _=  self.Encoders[i](embed_src[i],src_length,mask, num_token, breakpoint)
            out1.append(torch.unsqueeze(x_,-1))
            if i == 0:
                out2_ = x_
            else:
                out2_ += x_
               
            
         out1 = torch.cat(out1,-1)
         out2 = out2_/self.N

         return out1, None, out2

class EnsembleEmbeddings(nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
        
        self.N=kwargs['N']
        self.Embeddings=[Embeddings(*args, **kwargs) for i in range(self.N) ]
        self.Embeddings=nn.ModuleList(self.Embeddings)
        self.embedding_dim=self.Embeddings[0].embedding_dim
        self.lut=self.Embeddings[0].lut
        for param in self.Embeddings[0].parameters():
            param.requires_grad = False
    def      forward(self,*args, **kwargs):
        return   [self.Embeddings[i](*args, **kwargs) for i in range(self.N)]
    
    
    
class EnsembleSpatialEmbeddings(nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
        
        self.N=kwargs['N']
        self.SpatialEmbeddings=[SpatialEmbeddings(*args, **kwargs) for i in range(self.N) ]
        self.SpatialEmbeddings=nn.ModuleList(self.SpatialEmbeddings)
        self.embedding_dim =self.SpatialEmbeddings[0].embedding_dim
        for param in self.SpatialEmbeddings[0].parameters():
            param.requires_grad = False
    def      forward(self,*args, **kwargs):
        return [self.SpatialEmbeddings[i](*args, **kwargs) for i in range(self.N)]
        # embed_output = [self.SpatialEmbeddings[i](*args, **kwargs) for i in range(self.N)]
        # embed_src = [out[0] for out in embed_output]
        # mask = embed_output[0][1]
        # return embed_src, mask

class EnsembleGlosses(nn.Module):
    def __init__(self, N, enc_output_size, gls_output_size):
        super().__init__()
        
        self.N = N
        self.Glosses = [nn.Linear(enc_output_size, gls_output_size) for i in range(self.N)]
        # self.Embeddings=[Embeddings(*args, **kwargs) for i in range(self.N) ]
        self.Glosses = nn.ModuleList(self.Glosses)
        # self.Embeddings=nn.ModuleList(self.Embeddings)
        for param in self.Glosses[0].parameters():
            param.requires_grad = False
    def forward(self, encoder_output:Tensor):
        
        scores = [self.Glosses[i](encoder_output[:,:,:,i]) for i in range(self.N)]
        scores = torch.stack(scores)
        scores = scores.permute(1,2,3,0)
        gloss_scores = torch.sum(scores, dim=3) / self.N
        return gloss_scores
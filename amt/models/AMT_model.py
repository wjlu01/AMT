import torch.nn as nn
import torch

from amt.models.OIOT import MultimodalBertEncoder
from amt.models.OTGT import BertEncoder

class DoubleMultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(DoubleMultimodalBertClf, self).__init__()
        self.args = args
        self.OIOT_enc = MultimodalBertEncoder(args)
        self.OTGT_enc = BertEncoder(args)
        self.dense = nn.Linear(args.hidden_sz*2, args.hidden_sz)
        self.tanh = nn.Tanh()

        self.clf = nn.Linear(args.hidden_sz*3, args.n_classes)

    def forward(self, txt, mask1, segment1, txtAndCaption, mask2, segment2, txt_indexs, img):

        feature1 = self.OIOT_enc(txt, mask1, segment1, img)
        feature2 = self.OTGT_enc(txtAndCaption, mask2, segment2)
        
        img_feature1 = feature1[:,:self.args.num_image_embeds+2,:]
        txt_feature1 = feature1[:,self.args.num_image_embeds+2:,:]
        img_feature1 = torch.mean(img_feature1,dim=1).squeeze(1)
        txt_feature1 = torch.mean(txt_feature1,dim=1).squeeze(1)
        
        img_feature2 = None
        txt_feature2 = None
        

        for i,ind in enumerate(txt_indexs):
            if txt_feature2 == None:
                txt_feature2=torch.mean(feature2[i,:ind[0],:],dim=0)[None,]
            else:
                txt_feature2=torch.cat([txt_feature2,torch.mean(feature2[i,:ind[0],:],dim=0)[None,]],dim=0)
                
            if img_feature2 == None:
                img_feature2=torch.mean(feature2[i,ind[0]:sum(ind),:],dim=0)[None,]
            else:
                img_feature2=torch.cat([img_feature2,torch.mean(feature2[i,ind[0]:sum(ind),:],dim=0)[None,]],dim=0)
                
        x = self.tanh(self.dense(torch.cat([feature1[:,0],feature2[:,0]],dim=-1)))
        img_feature = self.tanh(self.dense(torch.cat([img_feature1,img_feature2],dim=-1)))
        txt_feature = self.tanh(self.dense(torch.cat([txt_feature1,txt_feature2],dim=-1)))
        
        x = self.clf(torch.cat([x,img_feature,txt_feature],dim=-1))

        return x

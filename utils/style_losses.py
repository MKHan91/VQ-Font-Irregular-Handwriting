import torch
import torch.nn as nn
import torchvision.models as models


def gram_matrix(feature):
    (b, ch, h, w) = feature.size()
    features = feature.view(b, ch, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (ch * h * w)


class VGGFeatureExtractor(nn.Module):
    """VGG16 feature extractor for style loss (returns conv1_1, conv2_1, conv3_1, conv4_1)
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        # slice layers to get outputs at desired conv layers
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(0, 4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(16, 23)]) # relu4_3
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        x = (x - self.mean) / self.std
        h = x
        h1 = self.slice1(h)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


class StyleLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = VGGFeatureExtractor().to(device)

    def forward(self, out, input_imgs):
        """Compute style loss between input and target images using Gram matrices.

        out and input_imgs expected in range [0,1] and shape [B, C, H, W].
        """
        B = out.shape[0]
        K = 3
        # VGG expects 3-channel images; if single-channel, repeat
        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
            input_imgs = input_imgs.repeat(1, 3, 1, 1)
        else:
            out = out
            input_imgs = input_imgs

        feats_out = self.vgg(out)
        feats_in = self.vgg(input_imgs)

        loss = 0.0
        for f_out, f_in in zip(feats_out, feats_in):
            G_out = gram_matrix(f_out)
            G_in = gram_matrix(f_in)
            
            # k=3 단위로 묶어서 평균 내기
            # ---------------------------------------
            C = G_in.shape[1]
            G_in = G_in.view(B, K, C, C)
            target_gram_avg = G_in.mean(dim=1)
            # ---------------------------------------
            
            loss += nn.functional.mse_loss(G_out, target_gram_avg)

        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Prototype SE ----------------
class ProtoSE(nn.Module):
    """
    SE module for prototype enhancement
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

# ---------------- SPM Module ----------------
class SPM(nn.Module):
    """
    Prototype Cross-scale Adaptive Feature Enhancement
    """
    def __init__(self, c1, c2, K=4, tau=0.07, topk=None):
        """
        c1 : 输入通道
        c2 : 输出通道
        K  : prototype 数量
        tau: temperature
        topk: 如果不为None，表示取 topk 个 prototype
        """
        super().__init__()
        self.K = K
        self.tau = nn.Parameter(torch.tensor(tau))
        self.topk = topk

        # embedding
        self.conv1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)

        # prototype generator
        self.proto_fusion = nn.Sequential(
            nn.Conv2d(c2*2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.proto_pool = nn.AdaptiveAvgPool2d((K,1))
        self.proto_transform = nn.Sequential(
            nn.Conv2d(c2, c2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            ProtoSE(c2)
        )

        # enhanced fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c2*2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

    def prototype_similarity(self, feat, proto):
        """
        Cosine similarity with prototype
        feat: (B,C,H,W)
        proto: (B,C,K)
        """
        B,C,H,W = feat.shape
        feat_flat = feat.view(B,C,-1)
        feat_norm = F.normalize(feat_flat, dim=1)
        proto_norm = F.normalize(proto, dim=1)

        sim = torch.einsum('bck,bcn->bkn', proto_norm, feat_norm)
        sim = sim.view(B,self.K,H,W)
        sim = sim / self.tau

        # topk or softmax
        if self.topk is not None and self.topk < self.K:
            sim_topk = torch.topk(sim, self.topk, dim=1)[0]
            sim = torch.mean(sim_topk, dim=1, keepdim=True)
        else:
            sim = F.softmax(sim, dim=1)
            sim = torch.max(sim, dim=1, keepdim=True)[0]

        return sim

    def forward(self, flow, fhigh):
        # 1 embedding
        flow_feat = self.conv1(flow)
        high_feat = self.conv2(fhigh)

        # 2 generate prototype
        proto_feat = self.proto_fusion(torch.cat([flow_feat, high_feat], dim=1))
        proto = self.proto_pool(proto_feat)
        proto = self.proto_transform(proto)
        B,C,K,_ = proto.shape
        proto = proto.view(B,C,K)

        # 3 guide flow
        sim_flow = self.prototype_similarity(flow_feat, proto)
        flow_enhanced = flow_feat * (1 + torch.sigmoid(sim_flow))

        # 4 guide fhigh
        sim_high = self.prototype_similarity(high_feat, proto)
        high_enhanced = high_feat * (1 + torch.sigmoid(sim_high))

        # 5 fusion
        out = self.fusion(torch.cat([flow_enhanced, high_enhanced], dim=1))
        return out

# ---------------- test ----------------
if __name__ == "__main__":
    flow = torch.randn(2,64,40,40)
    fhigh = torch.randn(2,64,40,40)
    model = PCAFE(c1=64, c2=64, K=4, tau=0.07, topk=None)
    out = model(flow,fhigh)
    print(out.shape)  # torch.Size([2, 64, 40, 40])
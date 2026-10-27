"""
models.py — CDA (Clothing-Domain Attention) 모델 정의
=========================================================================
구성:
  1. NSCM (Noise-Suppressing Channel Module) — Soft Gate 버전
     - entropy, border 집중도를 "학습되는 게이트의 입력(prior)"으로 사용
     - 하드 threshold 대신, 작은 MLP(게이트 네트워크)가 두 prior를 받아
       0~1 사이의 연속적인 채널별 억제 강도를 출력 (완전히 미분 가능)
  2. SBSM (Structural Boundary Spatial Module)
     - 고정 Sobel 커널로 경계선 지도 생성 -> 공간 주의집중 가중치로 사용
  3. CDABlock — NSCM + SBSM을 합친 attention 모듈, ResNet50 layer3 뒤에 삽입
  4. Baseline: Plain ResNet50, CBAM-ResNet50 (비교용)

사용 예시:
  from models import build_model
  model = build_model("cda", num_classes=5)         # 제안 모델
  model = build_model("resnet50", num_classes=5)    # baseline
  model = build_model("cbam", num_classes=5)         # CBAM baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================
# 1. NSCM — Soft Gate (학습 가능한 entropy ∩ border 결합)
# ============================================================
class NSCM(nn.Module):
    """
    입력 feature map F: (B, C, H, W)
    각 채널에 대해
      - 2D Shannon entropy (히스토그램 기반, non-parametric prior)
      - border(테두리) 영역 activation 집중도
    를 계산한 뒤, 이 두 값을 작은 게이트 네트워크(MLP)에 통과시켜
    채널별 억제 강도 g_c ∈ (0,1) 를 얻는다. (g_c가 작을수록 억제)

    최종 출력: F' = F * g   (채널별 스케일링, 완전히 미분 가능)

    entropy/border 계산 자체는 통계량 산출이라 파라미터가 없지만,
    "이 두 값을 어떻게 조합해서 얼마나 억제할지"는 게이트 네트워크가 학습한다.
    -> 파일럿 검증에서 쓴 고정 threshold(entropy 상위 40% ∩ border 상위 40%)를
       하드 규칙이 아니라, 데이터로부터 학습되는 연속 함수로 일반화한 버전.
    """

    def __init__(self, num_channels, entropy_bins=32, border_frac=0.15, hidden_dim=16, residual=False):
        super().__init__()
        self.entropy_bins = entropy_bins
        self.border_frac = border_frac
        self.residual = residual
        if residual:
            # Residual Gate: F_out = F * (1 - gamma * (1 - g))
            # gamma=0 초기화 -> 학습 시작 시 F_out == F (순정 ResNet50과 동일하게 출발)
            # gamma가 학습되면서 g가 낮은(배경) 채널만 점진적으로 억제
            self.gamma = nn.Parameter(torch.zeros(1))

        # 게이트 네트워크: [entropy, border_ratio] (채널별 2차원) -> 억제 강도(1차원)
        # 채널 인덱스에 무관하게 동일 MLP를 모든 채널에 공유 적용 (파라미터 효율적)
        self.gate_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # 게이트 초기값이 "거의 다 통과(g≈0.88)" 근처에서 시작하도록 마지막 레이어를 설계.
        # 단, 완전히 0으로 초기화하면 첫 스텝에 gate_net 앞쪽 레이어까지 gradient가
        # 전혀 흐르지 않는 문제가 있어(chain rule에서 0-가중치가 곱해짐),
        # 아주 작은 랜덤값으로 초기화해 "거의 상수에서 시작하되 처음부터 입력에 반응"하게 함.
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate_net[-1].bias, 2.0)  # sigmoid(2.0) ≈ 0.88

    @staticmethod
    def _channel_entropy(feat, bins=32, eps=1e-8):
        """
        feat: (B, C, H, W) -> entropy: (B, C)
        미분 가능하도록 soft-histogram(가우시안 커널 기반)을 사용.
        """
        B, C, H, W = feat.shape
        x = feat.reshape(B, C, -1)  # (B, C, H*W)
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        x_norm = (x - x_min) / (x_max - x_min + eps)  # 0~1 정규화

        # soft histogram: 각 bin 중심과의 거리 기반 가우시안 가중치
        bin_centers = torch.linspace(0, 1, bins, device=feat.device).view(1, 1, 1, bins)
        x_exp = x_norm.unsqueeze(-1)  # (B, C, N, 1)
        sigma = 1.0 / bins
        weights = torch.exp(-((x_exp - bin_centers) ** 2) / (2 * sigma ** 2))  # (B,C,N,bins)
        hist = weights.sum(dim=2)  # (B, C, bins)
        p = hist / (hist.sum(dim=-1, keepdim=True) + eps)
        entropy = -(p * torch.log(p + eps)).sum(dim=-1)  # (B, C)

        # 채널 간 비교 가능하도록 0~1로 정규화 (최대 entropy = log(bins))
        entropy = entropy / (torch.log(torch.tensor(float(bins), device=feat.device)) + eps)
        return entropy  # (B, C), 0~1 범위

    @staticmethod
    def _border_concentration(feat, border_frac=0.15, eps=1e-8):
        """
        feat: (B, C, H, W) -> border_ratio: (B, C)
        """
        B, C, H, W = feat.shape
        by = max(1, int(H * border_frac))
        bx = max(1, int(W * border_frac))

        feat_pos = F.relu(feat)  # 음수 activation 제외
        mask = torch.zeros(H, W, device=feat.device)
        mask[:by, :] = 1
        mask[-by:, :] = 1
        mask[:, :bx] = 1
        mask[:, -bx:] = 1
        mask = mask.view(1, 1, H, W)

        border_sum = (feat_pos * mask).sum(dim=(2, 3))  # (B, C)
        total_sum = feat_pos.sum(dim=(2, 3)) + eps
        return border_sum / total_sum  # (B, C), 0~1 범위

    def forward(self, feat):
        B, C, H, W = feat.shape
        entropy = self._channel_entropy(feat, self.entropy_bins)          # (B, C)
        border_ratio = self._border_concentration(feat, self.border_frac)  # (B, C)

        gate_input = torch.stack([entropy, border_ratio], dim=-1)  # (B, C, 2)
        gate_logit = self.gate_net(gate_input).squeeze(-1)          # (B, C)
        gate = torch.sigmoid(gate_logit)                             # (B, C), 0~1

        gate = gate.view(B, C, 1, 1)
        if self.residual:
            scale = 1.0 - self.gamma * (1.0 - gate)
        else:
            scale = gate
        refined = feat * scale
        return refined, gate.squeeze(-1).squeeze(-1)  # (refined feature, 채널별 게이트값 — 분석/시각화용)


# ============================================================
# 2. SBSM — 구조적 경계선 공간 모듈 (고정 Sobel, 공간 주의집중)
# ============================================================
class SBSM(nn.Module):
    """
    입력 F': (B, C, H, W)
    3x3 Sobel(Gx, Gy)을 채널별로 적용해 edge magnitude를 구하고,
    채널 평균으로 (B,1,H,W) 공간 지도를 만든 뒤 sigmoid로 0~1 attention map 생성.
    F'' = F' * (1 + spatial_attn)   (residual 형태로 곱해 경계 강조, 정보 손실 방지)
    """

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))
        # 채널 평균 edge map -> 1x1 conv로 attention map 보정 (약간의 학습 여지)
        self.attn_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, feat):
        B, C, H, W = feat.shape
        x = feat.reshape(B * C, 1, H, W)
        edge_x = F.conv2d(x, self.gx, padding=1)
        edge_y = F.conv2d(x, self.gy, padding=1)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)  # (B*C,1,H,W)
        magnitude = magnitude.view(B, C, H, W).mean(dim=1, keepdim=True)  # (B,1,H,W) 채널 평균

        attn = torch.sigmoid(self.attn_conv(magnitude))  # (B,1,H,W), 0~1
        refined = feat * (1 + attn)  # residual 강조
        return refined, attn.squeeze(1)  # (refined feature, attention map — 시각화용)


# ============================================================
# 3. CDABlock — NSCM + SBSM 결합
# ============================================================
class CDABlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.nscm = NSCM(num_channels)
        self.sbsm = SBSM()

    def forward(self, feat):
        f_prime, channel_gate = self.nscm(feat)
        f_double_prime, spatial_attn = self.sbsm(f_prime)
        return f_double_prime, {"channel_gate": channel_gate, "spatial_attn": spatial_attn}


# ============================================================
# 4. CDA 전체 모델 (ResNet50 backbone + CDABlock + Layer4 + Classifier)
# ============================================================
class CDAResNet50(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # 여기까지 통과한 특징맵 F 에 CDA 적용
        self.layer4 = backbone.layer4

        c3_channels = 1024  # ResNet50 layer3 출력 채널 수
        self.cda_block = CDABlock(c3_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f = self.layer3(x)                       # (B, 1024, H, W)
        f_refined, attn_info = self.cda_block(f)  # CDA 적용
        x = self.layer4(f_refined)
        x = self.gap(x).flatten(1)
        logits = self.fc(x)
        if return_attn:
            return logits, attn_info
        return logits


# ============================================================
# 4-1. Ablation — NSCM만 적용 (SBSM 제외)
# ============================================================
class NSCMOnlyResNet50(nn.Module):
    """CDA에서 SBSM을 뺀 버전. NSCM(entropy∩border Soft Gate)만의 순수 기여도 측정용.
    residual=True 이면 Residual Gate 버전 (gamma=0에서 출발, 순정 ResNet50과 동일 시작점)."""

    def __init__(self, num_classes=5, pretrained=True, residual=False):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.nscm = NSCM(1024, residual=residual)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f = self.layer3(x)
        f_refined, channel_gate = self.nscm(f)
        x = self.layer4(f_refined)
        x = self.gap(x).flatten(1)
        logits = self.fc(x)
        if return_attn:
            return logits, {"channel_gate": channel_gate}
        return logits


# ============================================================
# 4-2. Ablation — SBSM만 적용 (NSCM 제외)
# ============================================================
class SBSMOnlyResNet50(nn.Module):
    """CDA에서 NSCM을 뺀 버전. SBSM(구조적 경계선 공간 모듈)만의 순수 기여도 측정용."""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.sbsm = SBSM()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f = self.layer3(x)
        f_refined, spatial_attn = self.sbsm(f)
        x = self.layer4(f_refined)
        x = self.gap(x).flatten(1)
        logits = self.fc(x)
        if return_attn:
            return logits, {"spatial_attn": spatial_attn}
        return logits


# ============================================================
# 4-3. Ablation — SBSM을 Layer2(28×28, 512ch)로 이관한 버전
#      (Layer3(14×14)는 해상도가 낮아 세밀한 구조를 못 잡는다는 가설 검증용)
# ============================================================
class SBSMLayer2OnlyResNet50(nn.Module):
    """SBSM을 layer2 출력(28x28, 512ch)에 적용. NSCM은 사용하지 않음."""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # (B, 512, 28, 28)
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.sbsm = SBSM()  # Sobel 커널은 채널 수 무관하게 동작하므로 그대로 재사용 가능
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)                      # (B, 512, 28, 28)
        f2_refined, spatial_attn = self.sbsm(f2)  # 고해상도에서 경계선 강조
        x = self.layer3(f2_refined)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        logits = self.fc(x)
        if return_attn:
            return logits, {"spatial_attn": spatial_attn}
        return logits


# ============================================================
# 4-4. CDA v2 — NSCM(Layer3) + SBSM(Layer2) 결합
#      (각 모듈을 각자에게 유리한 해상도/의미 수준에 배치)
# ============================================================
class CDAv2ResNet50(nn.Module):
    """SBSM은 고해상도 layer2(28x28)에서 경계선을 잡고,
    NSCM은 의미적으로 더 무르익은 layer3(14x14, 1024ch)에서 채널 억제를 수행."""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.sbsm = SBSM()          # layer2(512ch, 28x28)에 적용
        self.nscm = NSCM(1024)      # layer3(1024ch, 14x14)에 적용

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, return_attn=False):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f2_refined, spatial_attn = self.sbsm(f2)
        f3 = self.layer3(f2_refined)
        f3_refined, channel_gate = self.nscm(f3)
        x = self.layer4(f3_refined)
        x = self.gap(x).flatten(1)
        logits = self.fc(x)
        if return_attn:
            return logits, {"spatial_attn": spatial_attn, "channel_gate": channel_gate}
        return logits


# ============================================================
# 5. Baseline — Plain ResNet50
# ============================================================
def build_plain_resnet50(num_classes=5, pretrained=True):
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# 6. Baseline — CBAM-ResNet50 (표준 CBAM: Channel + Spatial Attention)
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CBAMResNet50(nn.Module):
    """CDA와 동일한 위치(layer3 뒤)에 CBAM을 삽입한 공정 비교용 baseline"""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam = CBAM(1024)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ============================================================
# 7. 최신 CNN 백본 (torchvision ImageNet 사전학습) — 분류기 마지막 층만 교체
#    convnext_tiny     : ConvNeXt-T (2022), 약 28M  <- ResNet50(약 25M)과 규모가 비슷한 최신 CNN
#    convnext_small    : ConvNeXt-S (2022), 약 50M
#    efficientnet_v2_s : EfficientNetV2-S (2021), 약 21M
#    regnety_8gf       : RegNetY-8GF (2020), 약 39M
# ============================================================
MODERN_CNNS = ("convnext_tiny", "convnext_small", "efficientnet_v2_s", "regnety_8gf")


def build_modern_cnn(name, num_classes, pretrained=True):
    if name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext_small":
        m = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "regnety_8gf":
        m = tvm.regnet_y_8gf(weights=tvm.RegNet_Y_8GF_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(name)
    return m


# ============================================================
# 7. 모델 빌더
# ============================================================
def build_model(name, num_classes=5, pretrained=True):
    name = name.lower()
    if name in ("cda",):
        return CDAResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("nscm_only", "nscm"):
        return NSCMOnlyResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("nscm_res", "nscm_residual"):
        return NSCMOnlyResNet50(num_classes=num_classes, pretrained=pretrained, residual=True)
    elif name in ("sbsm_only", "sbsm"):
        return SBSMOnlyResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("sbsm_only_l2", "sbsm_l2"):
        return SBSMLayer2OnlyResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("cda_v2", "cda2"):
        return CDAv2ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("resnet50", "plain", "baseline"):
        return build_plain_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif name in ("cbam",):
        return CBAMResNet50(num_classes=num_classes, pretrained=pretrained)
    elif name in MODERN_CNNS:
        return build_modern_cnn(name, num_classes, pretrained)
    else:
        raise ValueError(
            f"알 수 없는 모델 이름: {name} "
            f"(cda / nscm_only / nscm_res / sbsm_only / sbsm_only_l2 / cda_v2 / resnet50 / cbam / " + " / ".join(MODERN_CNNS) + " 중 선택)"
        )


if __name__ == "__main__":
    # 간단한 shape 검증 (실제 가중치 다운로드 없이 구조만 확인하려면 pretrained=False)
    for name in ["resnet50", "cbam", "nscm_only", "nscm_res", "sbsm_only", "sbsm_only_l2", "cda_v2", "cda"]:
        m = build_model(name, num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        if name in ("cda", "nscm_only", "nscm_res", "sbsm_only", "sbsm_only_l2", "cda_v2"):
            out, attn = m(x, return_attn=True)
            extra = ", ".join(f"{k} shape={v.shape}" for k, v in attn.items())
            print(f"[{name}] output shape: {out.shape}, {extra}")
        else:
            out = m(x)
            print(f"[{name}] output shape: {out.shape}")
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  총 파라미터 수: {n_params:,}")

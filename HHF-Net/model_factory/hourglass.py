===========================================================
Algorithm 2: HHF-Net Backbone – PC-Hourglass
===========================================================

Inputs:
    D_in ← input depth map (B × 1 × 128 × 128)

Modules:
    Conv7×7(), BottleneckBlock(), HourglassModule(),
    CrissCrossAttention(), EstimationHead()

-----------------------------------------------------------
Step 1 — Shallow Convolutional Stem
-----------------------------------------------------------
F0 ← Conv7×7(D_in, out=64)
F0 ← BN + ReLU(F0)

-----------------------------------------------------------
Step 2 — Initial Residual Feature Extraction
-----------------------------------------------------------
F1 ← BottleneckBlock(F0, planes=64)
F1 ← MaxPool(F1, kernel=2, stride=2)     # → 64×64

F2 ← BottleneckBlock(F1, planes=64)
F3 ← BottleneckBlock(F2, planes=128)

Base_feature ← F3     # (B × 256 × 64 × 64)

-----------------------------------------------------------
Step 3 — Hourglass Encoder-Decoder (depth = 4)
-----------------------------------------------------------
Function Hourglass(F, depth=4):

    If depth == 1:
        Up_branch   ← Residual(F)
        Down_branch ← MaxPool(F)
        Low_feat    ← Residual(Down_branch)
        Up_feat     ← Upsample(Low_feat)
        Return Up_branch + Up_feat

    Else:
        Up_branch   ← Residual(F)
        Down_branch ← MaxPool(F)
        Low_feat    ← Hourglass(Residual(Down_branch), depth-1)
        Low_feat2   ← Residual(Low_feat)
        Up_feat     ← Upsample(Low_feat2)
        Return Up_branch + Up_feat

End Hourglass

F_hg ← Hourglass(Base_feature)

-----------------------------------------------------------
Step 4 — CrissCross Attention Refinement
-----------------------------------------------------------
F_ca ← CrissCrossAttention(F_hg):

    # Project features to query/key/value
    Q ← Conv1×1(F)
    K ← Conv1×1(F)
    V ← Conv1×1(F)

    # Horizontal attention
    Q_H, K_H → attention along H
    A_H ← Softmax(Q_H @ K_H^T)
    Out_H ← V_H @ A_H

    # Vertical attention
    Q_W, K_W → attention along W
    A_W ← Softmax(Q_W @ K_W^T)
    Out_W ← V_W @ A_W

    return γ * (Out_H + Out_W) + F

-----------------------------------------------------------
Step 5 — Estimation Head
-----------------------------------------------------------
UVD ← EstimationHead(F_ca)

Return:
    UVD (final 3D joints)

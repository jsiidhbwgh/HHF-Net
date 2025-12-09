===========================================================
Algorithm 1: EstimationHead – Multi-Branch 3D Pose Estimator
===========================================================

Inputs:
    F_in  ← feature map from PC-Hourglass backbone (B × 256 × 64 × 64)
Parameters:
    J     ← number of joints (14)
    D     ← depth feature dimension (64)
    β     ← learnable attention blending parameter (1 × J × 1 × 1)
    Xs, Ys ← precomputed coordinate grids (1 × 4096)
Modules:
    ResidualBlock(), FCBlock(), Conv1×1(), AdaptiveSpatialSoftmax()

-----------------------------------------------------------
Step 1 — UV Branch: Heatmap Regression
-----------------------------------------------------------
UV_feat     ← ResidualBlock(F_in, depth=3)
UV_feat     ← FCBlock(UV_feat)
UV_logits   ← Conv1×1(UV_feat, out_channels=J)     # heatmap logits

# Spatial softmax to convert logits into probability distribution
HMP         ← AdaptiveSpatialSoftmax(UV_logits)    # (B × J × 64 × 64)

-----------------------------------------------------------
Step 2 — Soft-Argmax for 2D Joint Regression
-----------------------------------------------------------
For each joint j in 1..J:
    Flatten heatmap channel j → hmp_j (B × 4096)
    U_j ← Σ(hmp_j * Xs)
    V_j ← Σ(hmp_j * Ys)

UV_coords ← concatenate(U_j, V_j)   # (B × J × 2)
UV_coords ← UV_coords * 2           # map 64→128 resolution

-----------------------------------------------------------
Step 3 — Attention Enhancement Branch (CBAM-like)
-----------------------------------------------------------
Att_feat    ← ResidualBlock(F_in, depth=3)
Att_feat    ← FCBlock(Att_feat)
Att_logits  ← Conv1×1(Att_feat, out_channels=J)

F_fused_att ← β * Att_logits + (1 - β) * UV_logits
Att_map     ← AdaptiveSpatialSoftmax(F_fused_att)  # (B × J × 64 × 64)

-----------------------------------------------------------
Step 4 — Depth Feature Extraction Branch
-----------------------------------------------------------
Depth_feat  ← ResidualBlock(F_in)
Depth_feat  ← FCBlock(Depth_feat)
Depth_feat  ← Conv1×1(Depth_feat, out_channels=D)   # (B × D × 64 × 64)

-----------------------------------------------------------
Step 5 — Attention-Weighted Depth Pooling
-----------------------------------------------------------
For each joint j in 1..J:
    value_j ← Σ_{x,y} Att_map[j,x,y] * Depth_feat[:, :, x, y]
    depth_vector[j] = value_j     # (B × D)

Depth_matrix ← stack(depth_vector[j])  # (B × J × D)

-----------------------------------------------------------
Step 6 — Linear Depth Regression
-----------------------------------------------------------
D_j = Linear(depth_vector[j])     # (B × J × 1)

Depth_coords ← stack(D_j)         # (B × J × 1)

-----------------------------------------------------------
Step 7 — Compose 3D Joint Output (UVD)
-----------------------------------------------------------
UVD ← concatenate(UV_coords, Depth_coords)   # (B × J × 3)

Return:
    UVD
    (optional: all intermediate maps for visualization)

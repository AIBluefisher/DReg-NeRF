import torch


class InfoNCELoss(torch.nn.Module):
    def __init__(self, d_embed, r_p, r_n) -> None:
        """
        Args:
            d_embed: Embedding dimension
            r_p: Positive radius (points nearer than r_p are matches)
            r_n: Negative radius (points nearer than r_p are not matches)
        """
        super().__init__()

        self.r_p = r_p
        self.r_n = r_n
        self.n_sample = 256
        self.W = torch.nn.Parameter(torch.zeros(d_embed, d_embed), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.W, std=0.1)

    def compute_infonce(self, anchor_feat, positive_feat, anchor_xyz, positive_xyz):
        """
        Args:
            anchor_feat: shape ([B,] N_anc, D)
            positive_feat: shape ([B,] N_pos, D)
            anchor_xyz: ([B,] N_anc, 3)
            positive_xyz: ([B,] N_pos, 3)
        Returns:
        """
        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum(
            '...ic,cd,...jd->...ij',
            anchor_feat,
            W_symmetrical,
            positive_feat
        ) # (..., N_anc, N_pos)

        with torch.no_grad():
            dist_keypoints = torch.cdist(anchor_xyz, positive_xyz)
            dist1, idx1 = dist_keypoints.topk(k=1, dim=-1, largest=False) # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypoints < self.r_n # Ignore all the points within a certain boundary.
            ignore.scatter_(-1, idx1, 0)    # except the positive (..., N_anc, N_pos)

        match_logits[..., ignore] = -float('inf')

        loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + \
                torch.logsumexp(match_logits, dim=-1)
        loss = torch.sum(loss[mask]) / torch.sum(mask)

        return loss

    def forward(self, src_feat, tgt_feat, src_xyz, tgt_xyz):
        """
        Args:
            src_feat: List(B) of source features (N_src, D)
            tgt_feat: List(B) of target features (N_tgt, D)
            src_xyz: List(B) of source coordinates (N_src, 3)
            tgt_xyz: List(B) of target coordinates (N_tgt, 3)
        Returns:
        """
        B = len(src_feat)
        infonce_loss = [
            self.compute_infonce(
                src_feat[b], tgt_feat[b], src_xyz[b], tgt_xyz[b]
            ) for b in range(B)
        ]

        return torch.mean(torch.stack(infonce_loss))

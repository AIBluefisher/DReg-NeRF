import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from conerf.model.feature_pyramid_net import FeaturePyramidNet3D
from conerf.register.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from conerf.register.se3 import compute_rigid_transform
from conerf.register.grid_downsample import hierarchical_grid_subsample
from conerf.register.transformer import TransformerCrossEncoderLayer, TransformerCrossEncoder
from conerf.loss.confidence_loss import compute_visibility_score


######################################### Helper Functions ########################################

def pad_sequence(
    sequences: list,
    require_padding_mask: bool = False,
    require_lens: bool = False,
    batch_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_sequences(padded, seq_lens: list):
    """Reverse of pad_sequence"""
    sequences = [
        padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))
    ]
    
    return sequences


def split_src_tgt(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    B = len(stack_lengths) // 2
    separate = torch.split(feats, stack_lengths, dim=dim)
    return separate[:B], separate[B:]

###################################################################################################


class NeRFRegTr(torch.nn.Module):
    def __init__(
        self,
        pos_emb_type: str = 'sine',
        pos_emb_dim: int = 256,
        pos_emb_scaling: float = 1.0,
        num_downsample: int = 6,
    ) -> None:
        super().__init__()
        self.num_downsample = num_downsample

        # Grid feature backbone, where inputs are 3D feature grid with rgba property.
        self.fpn3d = FeaturePyramidNet3D(in_channels=4, backbone='resnet50', pretrained=False)

        # Embeddings.
        if pos_emb_type == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, pos_emb_dim, scale=pos_emb_scaling)
        else:
            self.pos_embed = PositionEmbeddingLearned(3, pos_emb_dim)
        
        # Attention propagation.
        encoder_layer = TransformerCrossEncoderLayer(
            pos_emb_dim, num_heads=8, dim_feedforward=1024, dropout=0.0,
            activation='relu',
            normalize_before=True,
            sa_val_has_pos_emb=True,
            ca_val_has_pos_emb=True,
            attention_type='dot_prod',
        )
        encoder_norm = nn.LayerNorm(pos_emb_dim)
        self.transformer_encoder = TransformerCrossEncoder(
            cross_encoder_layer=encoder_layer,
            num_layers=6,
            norm=encoder_norm,
            return_intermediate=True
        )

        # Output layers
        self.correspondence_decoder = CorrespondenceDecoder(pos_emb_dim, True, self.pos_embed)

    def forward(self, data):
        """
        Args:
            data['src_xyz_rgba']: xyz and rgba properties from source grid with 
                                  shape [1, DIM_XYZ + DIM_RGBA, z_res, x_res, y_res]
            data['tgt_xyz_rgba']: xyz and rgba properties from target grid with 
                                  shape [1, DIM_XYZ + DIM_RGBA, z_res, x_res, y_res]
        """

        if len(data['src_xyz_rgba'].shape) == 6:
            data['src_xyz_rgba'] = data['src_xyz_rgba'].squeeze(0)
            data['tgt_xyz_rgba'] = data['tgt_xyz_rgba'].squeeze(0)
            data['src_mask'] = data['src_mask'].squeeze(0)
            data['tgt_mask'] = data['tgt_mask'].squeeze(0)
            data['src_nerf_path'] = data['src_nerf_path'][0]
            data['tgt_nerf_path'] = data['tgt_nerf_path'][0]
            data['pose'] = data['pose'].squeeze(0)

        batch_size = len(data['src_xyz_rgba'])
        assert len(data['src_xyz_rgba'].shape) == 5 # [batch_size, C, z_dim, x_dim, y_dim]

        src_xyz, tgt_xyz = data['src_xyz_rgba'][:, :3], data['tgt_xyz_rgba'][:, :3]
        src_rgba, tgt_rgba = data['src_xyz_rgba'][:, 3:], data['tgt_xyz_rgba'][:, 3:]
        src_feats, tgt_feats = self.fpn3d(src_rgba), self.fpn3d(tgt_rgba)

        # Upsampling the learned features to the same resolution as original grid.
        src_res, tgt_res = src_xyz.shape[-3:], tgt_xyz.shape[-3:]
        src_feats = F.interpolate(src_feats, size=src_res, mode='trilinear', align_corners=True)
        tgt_feats = F.interpolate(tgt_feats, size=tgt_res, mode='trilinear', align_corners=True)

        src_mask, tgt_mask = data['src_mask'], data['tgt_mask']
        cond_feat_dim = src_feats.shape[1]
        src_xyz = src_xyz.permute(0, 3, 4, 2, 1).reshape(batch_size, -1, 3)[0, src_mask] # [N, XYZ_DIM]
        tgt_xyz = tgt_xyz.permute(0, 3, 4, 2, 1).reshape(batch_size, -1, 3)[0, tgt_mask] # [N, XYZ_DIM]
        src_feats = src_feats.permute(0, 3, 4, 2, 1).reshape(batch_size, -1, cond_feat_dim)[0, src_mask] # [N, FEAT_DIM]
        tgt_feats = tgt_feats.permute(0, 3, 4, 2, 1).reshape(batch_size, -1, cond_feat_dim)[0, tgt_mask] # [N, FEAT_DIM]

        # Have to downsample both the source point cloud and 
        # the target point cloud since transformer requires large memory.
        point_clouds = torch.cat([src_xyz, tgt_xyz], dim=0)
        pc_features = torch.cat([src_feats, tgt_feats], dim=0)
        point_lengths = torch.tensor(
            [src_xyz.shape[0], tgt_xyz.shape[0]],
            dtype=torch.int64,
            device=src_xyz.device
        )
        ds_point_clouds, ds_features, ds_points_length = hierarchical_grid_subsample(
            points=point_clouds,
            features=pc_features,
            point_lengths=point_lengths,
            num_hierarchical=self.num_downsample
        )

        src_length, tgt_length = ds_points_length[0], ds_points_length[1]
        # print(f'src_length: {src_length}, tgt_length: {tgt_length}\n', flush=True)
        src_xyz, tgt_xyz = ds_point_clouds[:src_length], ds_point_clouds[src_length:] # [N, 3]
        src_feats, tgt_feats = ds_features[:src_length], ds_features[src_length:]

        src_pe, tgt_pe = self.pos_embed(src_xyz), self.pos_embed(tgt_xyz) # [N, DIM=256]
        src_pe_padded, _, _ = pad_sequence([src_pe]) # [N, batch=1, DIM=256]
        tgt_pe_padded, _, _ = pad_sequence([tgt_pe]) # [N, batch=1, DIM=256]

        # Performs padding, then apply attention to condition on the other 
        # grid features.
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(
            [src_feats], require_padding_mask=True
        )
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(
            [tgt_feats], require_padding_mask=True
        )

        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src=src_feats_padded, tgt=tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded,
            tgt_pos=tgt_pe_padded
        ) # [n_layer=6, N, batch_size=1, dim=256]

        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, [src_xyz], [tgt_xyz])

        # src_corr_list, tgt_corr_list = self.correspondence_decoder(
        #     src_feats_cond, tgt_feats_cond, [src_xyz], [tgt_xyz]
        # ) # list of [nl, N, dim=3]
        # # Replace the overlapping score by the visibility score from NeRF.
        # src_overlap_list = compute_visibility_score(src_corr_list, data['src_nerf_path']) # list of [nl, N, 1]
        # tgt_overlap_list = compute_visibility_score(tgt_corr_list, data['tgt_nerf_path']) # list of [nl, N, 1]

        # number of points for source features and target features
        src_slens, tgt_slens = src_xyz.shape[0], tgt_xyz.shape[0]
        src_feats_list = unpad_sequences(src_feats_cond, [src_slens]) # [nl=6, N, feat_dim=256]
        tgt_feats_list = unpad_sequences(tgt_feats_cond, [tgt_slens]) # [nl=6, N, feat_dim=256]
        num_pred = src_feats_cond.shape[0]

        # Stacks correspondences in both directions and computes the pose.
        src_xyz, tgt_xyz = [src_xyz], [tgt_xyz]
        corr_all, overlap_prob = [], []
        for b in range(batch_size):
            corr_all.append(torch.cat([
                torch.cat([src_xyz[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            # overlap_prob.append(torch.cat([
            #     torch.sigmoid(src_overlap_list[b][:, :, 0]),
            #     torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            # ], dim=1))
            overlap_prob.append(torch.cat([
                src_overlap_list[b][..., 0],
                tgt_overlap_list[b][..., 0],
            ], dim=1))

        pred_pose_weighted = torch.stack([
            compute_rigid_transform(
                corr_all[b][..., :3],
                corr_all[b][..., 3:],
                overlap_prob[b])
            for b in range(batch_size)], dim=1
        )

        outputs = {
            # Predictions
            'src_feats': src_feats_list, # List(B) of (N_pred, N_src, D)
            'tgt_feats': tgt_feats_list, # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz,
            'tgt_kp_warped': tgt_corr_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted # [num_layer, batch_size, 3, 4]
        }

        return outputs


class CorrespondenceDecoder(nn.Module):
    def __init__(
        self,
        pos_emb_dim,
        use_pos_emb,
        pos_embed=None,
        num_neighbors=0
    ) -> None:
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'
        
        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(pos_emb_dim)

        self.q_proj = nn.Linear(pos_emb_dim, pos_emb_dim)
        self.k_proj = nn.Linear(pos_emb_dim, pos_emb_dim)
        self.conf_logits_decoder = nn.Linear(pos_emb_dim, 1)
        self.num_neighbors = num_neighbors

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """
        Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred, ] Q, B, D)
            key: ([N_pred,], S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)
        
        Returns:
        Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k) # [B, N_query, N_src]

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :] # ([N_pred,], B, Q, S)
        
        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask
        
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    # def forward(self, src_feats_padded, tgt_feats_padded, src_xyz: list, tgt_xyz: list):
    #     """
    #     Args:
    #         src_feats_padded: Source features ([N_pred,] N_src, B, D)
    #         tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
    #         src_xyz: List of ([N_pred,] N_src, 3)
    #         tgt_xyz: List of ([N_pred,] N_tgt, 3)
        
    #     Returns:

    #     """
    #     src_xyz_padded, src_key_padding_mask, src_lens = \
    #         pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
    #     tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
    #         pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

    #     assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
    #            tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

    #     if self.use_pos_emb:
    #         both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
    #         slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
    #         src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
    #         src_pe_padded, _, _ = pad_sequence(src_pe)
    #         tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

    #     # Decode the coordinates
    #     src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
    #     tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
    #     src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
    #                                      tgt_key_padding_mask)
    #     tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
    #                                      src_key_padding_mask)

    #     src_corr_list = unpad_sequences(src_corr, src_lens)
    #     tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)

    #     return src_corr_list, tgt_corr_list


    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz: list, tgt_xyz: list):
        """
        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)
        
        Returns:

        """
        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)
        
        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)
        src_overlap = torch.sigmoid(src_overlap)
        tgt_overlap = torch.sigmoid(tgt_overlap)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


if __name__ == '__main__':
    # Test: python -m conerf.register.nerf_regtr
    src_xyz_rgba = torch.load(
        '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_0/grid_features.pt',
        map_location=torch.device('cuda:0')
    ).permute(3, 2, 0, 1).unsqueeze(dim=0)
    src_mask = torch.load(
        '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_0/mask.pt',
        map_location=torch.device('cuda:0')
    )

    tgt_xyz_rgba = torch.load(
        '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_1/grid_features.pt',
        map_location=torch.device('cuda:0')
    ).permute(3, 2, 0, 1).unsqueeze(dim=0)
    tgt_mask = torch.load(
        '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_1/mask.pt',
        map_location=torch.device('cuda:0')
    )

    data = {
        'src_xyz_rgba': src_xyz_rgba,
        'tgt_xyz_rgba': tgt_xyz_rgba,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_nerf_path': '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_0/model.pth',
        'tgt_nerf_path': '/home/chenyu/Datasets/nerf_synthetic/out/chair/block_1/model.pth',
    }

    nerf_regtr = NeRFRegTr().cuda()
    outputs = nerf_regtr(data)

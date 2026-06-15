# XSimGCL

Lightweight PyTorch implementation for this course project.

Data and evaluation convention follows the RecBole-GNN ML-1M setup used for
comparison:

- filter interactions with `rating < 3`
- split filtered interactions by `8:1:1`
- train with user-item graph propagation, BPR loss, and contrastive noise
- evaluate `Precision@10`, `Recall@10`, `NDCG@10`, and `MAP@10`

This is a compact project wrapper, not a full RecBole-GNN replacement.

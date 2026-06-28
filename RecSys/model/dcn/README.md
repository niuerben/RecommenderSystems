# DCN

本目录是只使用 `ratings.dat` 的 MovieLens 1M DCN 实现，结构和超参按 RecBole Context DCN 的 ML-1M 橙色建议调整。

模型输入只包含：

- `user_id`
- `movie_id`
- `rating`

不会使用 `users.dat` 的性别、年龄、职业，也不会使用 `movies.dat` 的标题或类型。

统一入口：

```bash
python main.py --model dcn --config RecSys/config/dcn.yaml
```

直接运行模型文件：

```bash
python RecSys/model/dcn/dcn.py \
  --ratings-file RecSys/data/ml-1m/ratings.dat \
  --epochs 5
```

保留文件：

- `dcn.py`：纯 rating 表 DCN 模型、数据构建、训练、TopN 评估和推荐导出。
- `README.md`：当前说明。

当前默认配置：

- `learning_rate: 0.001`
- `mlp_hidden_size: [512, 512, 512]`
- `reg_weight: 1`
- `cross_layer_num: 6`
- `dropout_prob: 0.2`

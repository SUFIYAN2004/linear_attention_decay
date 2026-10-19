# 54M-parameter mixer comparison (WikiText-103, 60M tokens per run, 2 seeds)

| mixer   |      params |   seeds |   val_loss |   std |   tok_per_s |   mem_gb |
|:--------|------------:|--------:|-----------:|------:|------------:|---------:|
| decay   | 5.37812e+07 |       2 |      4.211 | 0.006 |     19053.7 |    7.599 |
| idecay  | 5.38324e+07 |       2 |      4.088 | 0     |     18945.5 |    7.763 |
| reset   | 5.38325e+07 |       2 |      4.106 | 0.001 |     18728.6 |    7.764 |
| softmax | 5.37811e+07 |       2 |      4.347 | 0.003 |     29251   |    5.047 |

Same size, data and settings for every mixer.
Caveats: all models undertrained (about 1 token per parameter), 2 seeds, gap to softmax narrowed during training, decay used more memory and was slower at ctx 1024.

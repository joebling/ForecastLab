# cmd_1210.md
## python scripts/reversal_param_search_wf.py --T_list 7 14 21 --X_list 0.06 0.08 0.1 --init_train 500 --oos_window 63 --step 21

GitHub Copilot

简短分析（中文）

功能概述
- 这条命令对指定的候选阈值组 (T, X) 做网格化的 walk‑forward 参数搜索 + 回测：
  - 每个 (T,X) 会：生成标签 → 做特征 → 在多个滚动折（walk‑forward）上训练模型并在 OOS 窗口预测 → 把预测当作交易信号做逐日回放 → 汇总每折的检测与经济指标。

参数含义（逐项）
- --T_list 7 14 21  
  - 持有/观察窗口 T 的候选值（天数）。脚本会把 T 强制为整数，标签根据「未来 T 天」的涨跌幅计算。
- --X_list 0.06 0.08 0.1  
  - 阈值 X 的候选值（比例），分别为 6%、8%、10%。用于判断 future_upswing / future_drawdown 是否≥X 以打标签。
- --init_train 500  
  - 初始训练窗口大小（天数）。第一折用前 500 天作为训练集。
- --oos_window 63  
  - 每折的 OOS（测试/回测）窗口长度，63 天（约 3 个月）。
- --step 21  
  - 折之间的滑动步长，21 天（约 1 个月）。即每次把 start 向前推进 21 天产生下一折。

组合规模与折数
- 共 3 × 3 = 9 个候选 (T,X) 会被测试。
- 对每个候选，折数 = 脚本按 while start + oos_window ≤ n_samples 循环生成的次数（大致 = floor((n_samples - init_train - oos_window)/step) + 1），会根据你的数据长度决定；每折都会训练/回测一次。

输出文件（主要）
- 汇总结果： outputs/backtest/reversal_param_search_wf_results.csv（每个候选的聚合指标）
- 每个候选目录： outputs/backtest/T{T}_X{X*100}/equity_curves.csv、tradelog.csv（每折净值列与逐笔交易日志）
- 可选图表：可用 plot_backtest_summary.py 为 top N 生成图像（outputs/backtest/plots/...）

注意事项与建议（简短）
- 计算量：每候选多折训练/回测，耗时与 CPU 核数、模型类型（LightGBM/XGBoost 等）相关，可能从几十分钟到几小时不等；可并行化加速。  
- 模型优先级：脚本优先使用 LightGBM（如未安装会回退），请确认环境中已安装你偏好的库。  
- 校验输出：运行结束后先查看 summary（查看排序、交易次数、各折稳定性），再进一步放大网格或调整成本假设。  
- 后续操作示例（查看 summary / 画图）：
  - cat reversal_param_search_wf_results.csv | column -t -s, | head
  - python plot_backtest_summary.py --results reversal_param_search_wf_results.csv --top 3

需要我现在帮你：
- A）解析并展示 reversal_param_search_wf_results.csv 的 top3；或  
- B）把前 3 折的具体日期区间和对应 OOS 索引打印出来供你核对？
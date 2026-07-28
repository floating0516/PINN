# Phase39 逐秒波形前缀收敛报告

> 固定 Phase39 Glehman scalar + global invariant、seed42 checkpoint；不训练、不调网络、不换 seed、不调整 0.15 Mw 判据。

## 结论

总体事件等权 MAE 从 **188 秒观测时长**起持续不高于 0.15 Mw，对应最早可发布时间 **193 秒**。

在完整 200 秒时，事件等权 MAE 为 **0.147737 Mw**，与既有 Phase39 外部结果一致；八个事件中有 **5/8** 落在 ±0.15 Mw 误差带内。按“进入后每秒一直保持到 200 秒”的严格定义，共有 **5/8** 个事件在窗口内完成个体收敛。

200 秒端点复现门槛通过：逐站最大差异为 `2.86e-06 Mw`，事件中位数最大差异为 `0 Mw`。

![总体逐秒指标](figures/01_overall_metrics.png)

[PDF 图件](figures/01_overall_metrics.pdf)

## 八个事件轨迹

![事件逐秒轨迹](figures/02_event_trajectories.png)

[PDF 图件](figures/02_event_trajectories.pdf)

![事件收敛时间](figures/03_convergence_times.png)

[PDF 图件](figures/03_convergence_times.pdf)

## 事件级结果

| 事件 | 参考 Mw | 200 s 预测 | 200 s 绝对误差 | 首次进入 | 持续收敛观测时长 | 可发布时间 | 台站数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Luding 2022 M6.6 | 6.6 | 7.068 | 0.468 | 9 s | >200 s | >205 s | 6 |
| Samos 2020 M7.0 | 7.0 | 6.830 | 0.170 | 16 s | >200 s | >205 s | 3 |
| Xizang 2025 M7.1 | 7.1 | 7.253 | 0.153 | 25 s | >200 s | >205 s | 12 |
| Nepal 2015 M7.3 | 7.3 | 7.207 | 0.093 | 32 s | 64 s | 69 s | 5 |
| Sand 2025 M7.3 | 7.3 | 7.238 | 0.062 | 73 s | 73 s | 78 s | 45 |
| Iquique 2014 M7.7 | 7.7 | 7.552 | 0.148 | 63 s | 185 s | 190 s | 11 |
| Mandalay 2025 M7.7 | 7.7 | 7.652 | 0.048 | 187 s | 187 s | 192 s | 12 |
| Kodiak 2018 M7.9 | 7.9 | 7.860 | 0.040 | 179 s | 179 s | 184 s | 64 |

## 评估口径

- 输入仍是单台站 R 分量。每个整数秒 `h=1..200` 只保留完整预处理张量的前 `h` 个 1 Hz 槽位，后续值和有效掩码全部清零。
- 7 点居中 Hamming FIR 需要最多 3 秒未来支撑，因此结果同时报告 `h` 秒观测时长和 `h+5` 秒发布时间。
- 每个台站独立输出 Mw；同一事件对可用台站预测取中位数。参考值来自冻结 USGS 快照的 `mw_selected`。
- 收敛定义为首次满足 `|Event error| <= 0.15 Mw`，并且此后每个整数秒都保持到 200 秒。
- 固定 cohort 为 8 个事件、158 个台站；完整台站可用性从 25 s 起持续保持。

## 必须保留的边界

这是一项固定 checkpoint 的波形前缀敏感性诊断，不是严格因果网络验证。Phase39 的 TCN 使用对称 padding，Transformer 没有 causal mask；固定台站 cohort 也来自完整 200 秒记录的离线筛选。五秒延迟只覆盖预处理 FIR 的未来支撑，不会改变网络内部结构。

这八个事件确实没有进入模型训练，但它们已被多轮方案比较反复使用，因此研究角色仍是 `development_validation`。本报告可以回答“固定 Phase39 在这批开发事件上需要多少秒达到稳定误差水平”，不能单独把结果升级为无偏的最终未见事件泛化证据。

## 工件

- [总体摘要](summary.json)：总体结论、端点复现和事件收敛时间。
- [事件收敛表](event_convergence.csv)：首次进入和 suffix-stable 收敛时间。
- [事件逐秒预测](event_predictions.csv)：8 个事件 × 200 个观测秒的预测轨迹。
- [逐站逐秒预测](station_predictions.csv)：完整台站级预测。
- [逐秒总体指标](horizon_metrics.csv)：每秒 Event MAE、RMSE、bias、覆盖率和达标事件数。
- [早期不可用台站](unavailable_stations.csv)：pre-P 基线尚未就绪的台站与时刻。
- [固定 cohort](cohort_contract.json) 与 [运行来源](provenance.json)：评估合同和输入哈希。
- [发布清单](publication_manifest.json)：GitHub 工件与生成代码 SHA-256。
- [可复现评估脚本](../../../scripts/evaluation/evaluate_phase39_second_by_second.py) 与 [测试](../../../tests/test_phase39_second_by_second.py)。

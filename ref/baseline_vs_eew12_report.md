# evaluate_baseline_raw 与 EEW_0012 结果对比报告

## 1. 对比范围与口径

- 当前结果来源：`src/evaluation/evaluate_baseline_raw.py` 的终端输出（你提供的 Terminal#4-98）。
- EEW12 参考来源：`EEW_0012/EEW_0012.md` 中：
  - Table 2（走滑+正断层）：含 RP/No RP/Golriz2023
  - Table 4（逆冲）：含 RP/No RP/Golriz2023
- 差值定义：`Δ = 当前 Mwg - EEW12 Mwg`
- 事件映射说明：
  - `Tehuantepec2017` 对应 `Chiapas, Mexico`
  - `Chignic2021` 对应 `Chignik, Alaska`
  - `CapeMendocino2024` 对应 `Cape Mendocino, California`

---

## 2. No RP 对比表

| 当前事件 | EEW12事件 | 当前Mwg | EEW12 Mwg (No RP) | Δ |
|---|---|---:|---:|---:|
| Ridgecrest2019 | Ridgecrest, California | 7.33 | 7.05 | +0.28 |
| ElMayor2010 | El Mayor-Cucapah, Mexico | 7.43 | 7.17 | +0.26 |
| Pazarcik2023 | Pazarcik, Turkey (5 Hz) | 7.88 | 7.68 | +0.20 |
| Elbistan2023 | Elbistan, Turkey (5 Hz) | 7.80 | 7.56 | +0.24 |
| SandPoint2020 | Sand Point, Alaska (Mw7.6) | 7.44 | 7.33 | +0.11 |
| CapeMendocino2024 | Cape Mendocino, California | 7.33 | 7.39 | -0.06 |
| RatIslands2014 | Rat Islands, Alaska | 7.79 | 7.51 | +0.28 |
| Tokachi2003 | Tokachi-Oki, Japan | 8.26 | 8.00 | +0.26 |
| Maule2010 | Maule, Chile | 8.27 | 8.57 | -0.30 |
| Tohoku2011 | Tohoku-Oki, Japan | 8.63 | 8.79 | -0.16 |
| Iquique2014 | Iquique, Chile | 8.21 | 7.61 | +0.60 |
| Illapel2015 | Illapel, Chile | 8.21 | 7.93 | +0.28 |
| Kilauea2018 | Kilauea, Hawaii | 7.18 | 7.18 | +0.00 |
| Simeonof2020 | Simeonof, Alaska | 7.90 | 7.45 | +0.45 |
| Chignic2021 | Chignik, Alaska | 8.14 | 7.68 | +0.46 |
| Tehuantepec2017 | Chiapas, Mexico | 8.11 | 7.89 | +0.22 |

---

## 3. RP ON 对比表

| 当前事件 | EEW12事件 | 当前Mwg | EEW12 Mwg (RP) | Δ |
|---|---|---:|---:|---:|
| Ridgecrest2019 | Ridgecrest, California | 缺失 | 7.18 | — |
| ElMayor2010 | El Mayor-Cucapah, Mexico | 8.10 | 7.19 | +0.91 |
| Pazarcik2023 | Pazarcik, Turkey (5 Hz) | 8.70 | 7.67 | +1.03 |
| Elbistan2023 | Elbistan, Turkey (5 Hz) | 8.50 | 7.55 | +0.95 |
| SandPoint2020 | Sand Point, Alaska (Mw7.6) | 7.45 | 7.63 | -0.18 |
| CapeMendocino2024 | Cape Mendocino, California | 7.88 | 7.76 | +0.12 |
| RatIslands2014 | Rat Islands, Alaska | 缺失 | 7.73 | — |
| Tokachi2003 | Tokachi-Oki, Japan | 8.86 | 8.23 | +0.63 |
| Maule2010 | Maule, Chile | 8.89 | 8.82 | +0.07 |
| Tohoku2011 | Tohoku-Oki, Japan | 8.79 | 9.02 | -0.23 |
| Iquique2014 | Iquique, Chile | 8.20 | 8.23 | -0.03 |
| Illapel2015 | Illapel, Chile | 8.52 | 8.41 | +0.11 |
| Kilauea2018 | Kilauea, Hawaii | 7.92 | 7.28 | +0.64 |
| Simeonof2020 | Simeonof, Alaska | 8.27 | 7.90 | +0.37 |
| Chignic2021 | Chignik, Alaska | 8.53 | 8.12 | +0.41 |
| Tehuantepec2017 | Chiapas, Mexico | 8.74 | 8.11 | +0.63 |

---

## 4. 结论

1. **No RP 更接近 EEW12**：多数事件差值在 ±0.3 附近，但 Iquique、Simeonof、Chignik 偏高明显。  
2. **RP ON 出现系统性偏高**：多事件达到 +0.6 到 +1.0（如 ElMayor、Pazarcik、Elbistan）。  
3. **差异主因是口径不一致**：  
   - EEW12 对部分事件区分 1Hz/5Hz，而当前结果为单事件聚合；  
   - 台站集合与筛选阈值不同（台站数明显不一致）；  
   - RP ON 中个别事件在当前输出缺失（如 Ridgecrest、Rat Islands）。  

---

## 5. 参考位置

- EEW12 Table 2 与相关描述：`EEW_0012/EEW_0012.md` 第 272–293 行  
- EEW12 Table 4 与相关描述：`EEW_0012/EEW_0012.md` 第 356–369 行
- 当前评估脚本入口：`src/evaluation/evaluate_baseline_raw.py` 第 265–270 行


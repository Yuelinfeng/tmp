# Acceptance-Aware Training 原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 6 | Acceptance-Aware Draft Model Training for Speculative Decoding | Tianhua Xia, Mugilan Ganesan, Yifei Feng, Haiyu Wang, Maximilian Egger, Sai Qian Zhang | 2026-09 | `@article{xia2026acceptanceaware, title={Acceptance-Aware Draft Model Training for Speculative Decoding}, author={Xia, Tianhua and Ganesan, Mugilan and Feng, Yifei and Wang, Haiyu and Egger, Maximilian and Zhang, Sai Qian}, year={2026}, eprint={2609.24150}}` | WTV 用窗口中的累计接受概率训练 Draft，采样实验有增量；实际计算采用固定教师前缀上的条件独立近似。 | §3.1–3.4；Eq. 4/14/15；Figure 1；§4 与 Tables 1–3 | [Official PDF](https://arxiv.org/pdf/2609.24150v1) / [Page](https://arxiv.org/abs/2609.24150) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/06_Acceptance_Aware_Training.pdf) | 13 页 v1，方法、训练、采样和 greedy 消融已读取；已选择 P/F/P 窗口，完整范围采集进行中 |

原文 WTV 不使用 KL 项；GRPO 奖励来自 greedy argmax 的当前与参考轨迹差，不能直接当作本项目的非零温度接受推进。标题和编号的公开仓库检索未找到作者训练实现，网站限定发现未返回候选；不据此断言代码从未公开。原始响应另存 `aat_candidate_github_*.json`。

PFSpec 需要保留两 Feature 的完整条件：首 Feature 可复用质量 B(f)，对应 Position 概率 c1/c2(f)，次 Feature 集合接受 α2(f)，目标为 J1+J2。用有完整概率范围的旧提案采集 f 后，B(f)c2(f)α2(f)/q_old(f) 可估计 J2；原 Top-p=0.9 教师数据缺少这项覆盖条件。公式、采集范围和待验证内容在 `aat_candidate_math.md`。

实际实现窗口为 P1/F1/P2，预计接受 Draft 长度为 c1+J1+c1ΣB(f)c2(f)。第二 Feature 分布不参与目标或推理；原双 Feature 候选数学分析保留为候选记录，没有实施为本篇主要方案。

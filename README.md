# PFSpec Lossless Acceptance Handoff

`PFSpec_Lossless_Acceptance_Handoff_20261107.tgz` 原文件为 285 MB，超过 GitHub 的 100 MB 单文件限制，因此拆分为 3 个分卷。

## 重组

```bash
cat PFSpec_Lossless_Acceptance_Handoff_20261107.tgz.part-* > PFSpec_Lossless_Acceptance_Handoff_20261107.tgz
```

校验：

```
c614dc59fb5ea90bcf6b8b614592c7744d0391ff697e45cb1a3cb8ded5928ca4
```

```bash
shasum -a 256 PFSpec_Lossless_Acceptance_Handoff_20261107.tgz
```

## 解包后的目录

压缩包已解包为 `PFSpec_Lossless_Acceptance_20261107/`（literature / project / results 等），目录内容已完整推送。

其中 `results/paper06_dflash/training/diffusion_context_w64/selected.pt`（274 MB）同样超过 100 MB 限制，已拆分为 3 个分卷：

```bash
cd PFSpec_Lossless_Acceptance_20261107/results/paper06_dflash/training/diffusion_context_w64
cat selected.pt.part-* > selected.pt
```

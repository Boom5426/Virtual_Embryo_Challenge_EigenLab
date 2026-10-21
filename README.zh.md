# VEC 新手入门：从注册到第一次提交

这份教程面向第一次参加 [Virtual Embryo Challenge](https://virtualembryo.ai)（VEC）的同学。
跟着走一遍，你会完成：注册并接受规则、下载数据、看懂 `.h5ad` 文件和每个 board 的格式要求、
做出一个最简单的基线预测文件、在本地检查它、先做免费的格式检查再正式提交，最后在终端里看排行榜。

说明：本工具包是社区作品，与主办方无关，也不包含任何建模方法。文中的规则都在 2026-10-21
对照官网核实过；规则以官网（Rules、Tasks 页面、API 指南）为准，有疑问请查官网。

## 0. 比赛在做什么

一共三个任务，官网的说法概括如下：

* **Task 1（T1，单细胞时间预测）**：给定目标时间点之前的阶段，预测未来某个阶段的基因表达分布（一组细胞）。
  用 E8.5 和 E9.5 训练，验证阶段在排行榜上对 E10.5 打分，最终排名用隐藏的 E12.5。
* **Task 2（T2，三维空间多尺度预测）**：同时预测未来阶段每个细胞的基因表达和三维位置。
  心脏（heart）和整个胚胎（embryo）两种设定分开打分。
* **Task 3（T3，敲除扰动预测）**：根据野生型发育数据和一个已观测的敲除，预测另一个没见过的敲除胚胎的表达和三维坐标。
  E9.5 的 Mab21l2 敲除用于训练，E8.75 的 Gata4 敲除是验证目标，E8.75 的 β-catenin 敲除是隐藏测试。

## 1. 注册并接受规则

1. 打开官网的账户页（`/challenge/account`），用 GitHub 或 Google 登录。比赛账户独立于网站其他部分。
2. 如果队友发来了邀请链接，直接打开链接加入他们的队伍；只有自己建队时才填写注册表。
3. 注册时选择赛道：Human Team 或 Agent Team。官网说明，Agent Team 赛道要求在评奖前公开 agent 系统及其完整的演化记录（evolutionary trace）。
4. 把 Challenge Rules 和 Terms of Use 分别滚动到底，复选框才会变为可勾选，勾选即表示接受。
5. 需要用脚本提交时，由队长在账户的 API keys 页面（`/challenge/account/api-keys`）创建 API key。
   密钥只显示一次，请立刻保存。不要把密钥直接写在命令行里（会留在 shell 历史和进程列表中），推荐这样保存：

   ```bash
   umask 077; cat > ~/.vec_key      # 粘贴密钥，回车，再按 Ctrl-D
   ```

   密钥会在创建时选择的有效期后失效，最晚不超过提交截止日期；更换队长也会使普通队伍的密钥失效。

## 2. 下载数据

* **网页下载**：登录后打开 Data 页面（`/challenge/data`）的 Download 部分。训练数据只对已注册的参赛者开放。
* **用 API 下载**：创建密钥时勾选 “Allow data downloads”，然后用
  `GET /challenge/data/manifest` 列出已发布的文件，用 `GET /challenge/data/link?key=...` 拿到下载链接。
  链接最长有效 15 分钟，过期后重新申请即可。下载不消耗提交次数。
* 官网说明这个下载接口不授权再分发或托管数据副本，所以请不要把数据上传到公开仓库。本工具包也不包含任何比赛数据。

文件名以 manifest 为准。官网 Data 页面列出的 T1 训练文件是 `E8.5_RNA.h5ad` 和 `E9.5_RNA.h5ad`；
T2 的文件按阶段命名（`E<stage>.h5ad` 这一形式），确切名称请看 manifest。

## 3. 看懂 h5ad 文件

`.h5ad` 是 AnnData 的文件格式。一个文件就是一个“细胞 × 基因”的表达矩阵，再加上细胞和基因的注释：

| 位置 | 含义 | 本比赛的要求 |
|---|---|---|
| `X` | 表达矩阵，行是细胞，列是基因 | 有限值、非负、已做 log 归一化；稀疏或稠密都可以，评分时会转成 float32 |
| `var` 的索引（`var_names`） | 基因名 | 必须与该 board 的基因列表完全一致，顺序也一致 |
| `obs` 的索引（`obs_names`） | 细胞名 | 不与真实细胞一一对应，指标都是分布层面的 |
| `obs['celltype']` | 细胞类型 | 可选，评分时不读取，评分器会自己给每个细胞定类型 |
| `obsm['spatial_3D']` | 每个细胞的 x, y, z 坐标 | T2 和 T3 必需，形状为（细胞数, 3 或更多），只读前三列，必须是有限值，坐标系任意 |

各任务的区别：

* **T1**：真实的解离单细胞，全转录组 32,285 个基因，log1p 归一化，没有空间坐标，提交文件里也不需要坐标。
  官网特别提醒：把 T1 当成 500 基因的 MERFISH 面板，是 T1 最常见的格式错误。
* **T2 / T3**：3D MERFISH，测的是一个 500 基因的面板（当前 index 中 embryo 设定的 board 是 498 个基因），
  log 归一化，带每个细胞的三维坐标。T3 提交的是突变胚胎本身，不是它与野生型的差值。

在 Python 里快速查看一个文件：

```python
import anndata as ad
a = ad.read_h5ad("E9.5.h5ad")
print(a)                       # 细胞数 x 基因数，以及 obs / var / obsm 里有哪些键
print(a.var_names[:5])         # 前几个基因名
print(a.obsm["spatial_3D"][:3])  # T2/T3 文件的前三个细胞坐标
```

关于归一化：官网要求数值已经做过 log 归一化。原始计数矩阵是非负且有限的，所以能通过所有格式检查，
但会被当成 log 尺度的数据打分，服务器不会提示你，分数只会悄悄变差。只做了归一化、没取 log 的矩阵同样不对。
官网没有写明归一化的目标总和。`vec_check.py` 会报告 `expm1(X)` 每行求和的分布，偏离 1e4 时给出警告（不是失败），
你可以用同样的方法检查一个官方训练文件，看看它是怎样缩放的，再让你的预测保持一致。

## 4. 每个 board 的“合同”

每个可以提交的榜单叫一个 board，规格写在公开的
`https://virtualembryo.ai/challenge/panels/index.json` 里，基因列表文件放在它旁边。主要字段：

| 字段 | 含义 |
|---|---|
| `key` | board 名，例如 `T2:heart:val_extrap` |
| `n_genes` / `genes_file` | 基因数和基因列表文件（每行一个基因，顺序就是要求的顺序） |
| `needs_coords` / `obsm_required` | 是否需要 `obsm['spatial_3D']` |
| `min_cells` / `max_cells` | 允许的细胞数范围，两端都包含 |
| `floor_model` | 这个 board 的保底模型（floor），得分定为 50 |
| `anchors` | 每项指标在 floor 和 ceiling 处的原始数值，也就是分数尺度两端的锚点（详见官网 Evaluation 页面） |

2026-10-21（验证阶段）的 board：

| board | 基因数 | 细胞数范围 | 需要坐标 | floor 模型 |
|---|---|---|---|---|
| `T1:val` | 32,285 | 1000..5118 | 否 | copy_last |
| `T2:heart:val_interp` | 500 | 1000..17616 | 是 | copy_last |
| `T2:heart:val_extrap` | 500 | 1000..25179 | 是 | copy_last |
| `T2:embryo:val_interp` | 498 | 583..5000 | 是 | copy_last |
| `T3:gata4` | 500 | 1000..7449 | 是 | wt_identity |

阶段切换时 board 会变。随时可以用下面的命令看当前的 index：

```bash
python vec_check.py --list-boards
```

## 5. 做一个最简单的基线：复制上一个阶段

官网公布的 floor 模型就是“什么都不变”：

* **copy_last**（T1、T2）：原样复制上一个观测阶段的所有细胞；T2 连同三维坐标一起复制。
* **wt_identity**（T3）：原样返回同一阶段的野生型，也就是假设敲除没有任何效果。

`examples/copy_last_baseline.py` 按照 board 的公开合同，把一个官方训练文件整理成合格的预测文件：
按面板选出基因并排好顺序、细胞数超过上限时用固定随机种子抽样、需要时带上坐标。它只用到官方发布的早期阶段数据和公开合同。

先问它每个 board 应该复制哪个阶段（依据官网 Tasks 页面的阶段顺序）：

```bash
python examples/copy_last_baseline.py --board T2:heart:val_extrap --hint
```

| board | 复制哪个文件 |
|---|---|
| `T1:val`（目标 E10.5） | E9.5 单细胞文件 |
| `T2:heart:val_interp`（目标 E8.5） | E8.25 心脏 MERFISH 文件 |
| `T2:heart:val_extrap`（目标 E10.5） | E9.5 心脏 MERFISH 文件 |
| `T2:embryo:val_interp`（目标 E7.5） | E7.25 胚胎 MERFISH 文件 |
| `T3:gata4`（Gata4 敲除，E8.75） | E8.75 野生型 MERFISH 文件 |

然后生成文件：

```bash
python examples/copy_last_baseline.py --board T2:heart:val_extrap \
    --input E9.5.h5ad --out floor_heart_extrap.h5ad
```

注意：这个文件只是用来打通流程的，不是有竞争力的提交。因为可能做了细胞抽样，它的分数不保证与官方 floor 的 50 分完全相同。

## 6. 本地检查：vec_check.py

```bash
python vec_check.py floor_heart_extrap.h5ad --board T2:heart:val_extrap
```

它会下载并缓存（`~/.cache/vec-kit`）公开 index 和对应的基因文件，然后逐项检查：
文件后缀和大小（上限 1,200 MiB）、基因名和顺序、细胞数范围、X 是否有 NaN 或无穷、是否有负数、
是否像 log 尺度、`obsm['spatial_3D']` 的形状和数值。每一项显示 PASS / FAIL / WARN，最后给出结论：

* 退出码 `0`：没有失败项（可能有警告）；
* 退出码 `1`：至少一项失败（加 `--strict` 时警告也算失败）；
* 退出码 `2`：用法或环境问题，比如 board 名写错、无法下载 index。

如果装了官方的 `veckit`（`pip install veckit`），它还会调用 veckit 自己的读取函数再检查一遍。

本地通过不代表服务器一定接受。下一步的服务器格式检查才是权威，而且官网也说明，
即使服务器格式检查通过，也不保证评分器一定接受。

## 7. 提交：先免费检查，再正式打分

```bash
# 只做免费的服务器格式检查（不创建提交，不消耗次数）
python vec_submit.py floor_heart_extrap.h5ad --board T2:heart:val_extrap \
    --model "copy-last test" --key-file ~/.vec_key

# 先看看会发什么请求（什么都不发送）
python vec_submit.py floor_heart_extrap.h5ad --board T2:heart:val_extrap \
    --model "copy-last test" --key-file ~/.vec_key --scored --dry-run

# 确认无误后，正式消耗一次打分机会
python vec_submit.py floor_heart_extrap.h5ad --board T2:heart:val_extrap \
    --model "copy-last test" --key-file ~/.vec_key --scored
```

`vec_submit.py` 的流程：

1. 先跑一遍 `vec_check`，本地失败就停下，什么都不发送；
2. 确认当前阶段接受提交、这个 board 处于开放状态，并打印你的配额；
3. 发送 `format_only=true` 的格式检查（按 API 指南的要求不带 Idempotency-Key），失败就停下；
4. 只有加了 `--scored` 才正式上传。Idempotency-Key 由 board、模型名和文件的 SHA-256 推导出来，
   同样的文件重试时 key 不变，服务器会返回第一次的结果，不会重复扣次数。遇到超时或 5xx 会按 2、4、8 秒退避后原样重发，
   遇到 4xx 绝不自动重试；
5. 每 20 秒查询一次状态（API 指南建议 15 到 30 秒），直到 `scored` 或 `rejected`，然后打印总分、各项指标、排名，
   核对服务器计算的 SHA-256 是否与本地文件一致，并再次打印配额。

所有请求和返回都会追加到本地的 `vec_ledger.jsonl` 里，方便日后查账；密钥永远不会被打印或写进日志。
如果日志里已经有同一个 key 的提交记录，再次运行同样的命令只会继续查询那次提交，不会重新上传。
T2 会自动带上 `setting` 和 `mode`，T3 会带上 `target`，与官网网页上传器的做法一致。

Agent Team 的提交会先进入 `awaiting_evidence` 状态，需要补充证据（例如运行轨迹和 prompts）。
本工具不上传证据，遇到这个状态会停下并给出提交 id，请在网页或通过证据接口补交。

## 8. 配额和最常见的拒绝原因

以下均来自官网（Rules 和 API 指南），2026-10-21 核实：

* **配额**按队伍、按任务、按 UTC 自然日计算：P1 开发阶段每天 20 次打分，P2（基线发布后）每天 8 次，
  P3 最终测试阶段每个 board 整个阶段只有 2 次正式提交，不按天重置。`GET /challenge/phase` 会显示当前阶段和 `daily_quota`。
* 网页和 API 共用同一份配额，多建几个密钥也不会多出次数。
* 被拒绝的文件不消耗打分次数，但会计入另一个每日上传上限；格式检查两者都不消耗。
* 返回 `429` 说明配额用完了。每日上限在 UTC 零点重置，最终阶段的 board 预算不会重置。
* 被拒绝的预测也可能返回 HTTP 200，要看返回里的 `status` 和 `problems`，不能只看状态码。

最常见的拒绝原因：

1. **基因不对**：基因名或顺序与该 board 的基因列表不完全一致；T1 误用了 500 基因面板。
2. **细胞数超出范围**：在服务器打分前就会被拒绝，不消耗打分次数。
3. **X 里有 NaN 或无穷**：哪怕只有一个，整个文件都会被拒绝。
4. **X 有负数**：通常是在流程里做了中心化或 z-score。
5. **T2/T3 缺少坐标**：没有 `obsm['spatial_3D']`，或者形状不是（细胞数, 3 及以上），或者含 NaN。
6. **文件太大**：超过 1,200 MiB 会返回 `413`，格式检查也一样。
7. **`409`**：需要重新接受规则或条款（看 `GET /challenge/me` 的 `stale_consents`）、提交已关闭，
   或者同一个 Idempotency-Key 配了不同的内容。
8. **`401`**：密钥缺失、无效、过期或已被吊销。

还有一个不会被拒绝、但会让分数悄悄变差的错误：提交了原始计数或没取 log 的矩阵。用 `vec_check.py` 的 WARN 提示来发现它。

## 9. 看排行榜：vec_board.py

排行榜接口是公开的，不需要密钥：

```bash
python vec_board.py overall              # 按赛道分别显示总榜，并核对总分
python vec_board.py task T2              # T2 每个 board 一张表，带各项指标
python vec_board.py team "你的队名"       # 某个队伍的总排名和各 board 成绩
python vec_board.py snapshot             # 把所有公开接口保存到带时间戳的文件夹
```

计分规则（官网 Rules）：每个 board 上 floor 模型为 50 分，可达到的上限（ceiling）为 100 分；
队伍在一个 board 上的成绩取最好的一次，不是最近的一次；总分是三个任务各自的平均分（任务内对各 board 取平均）之和，
没有提交过的 board 按 0 分计；排名在各自赛道内进行；最终排名由隐藏测试集决定，不是开发阶段的排行榜。
`vec_board.py overall` 会按这个公式重新计算每一行，并报告与 API 给出的总分不一致的行。

## 10. 安装与测试

```bash
python -m pip install -r requirements.txt     # numpy、scipy、anndata、h5py，需要 Python 3.10 及以上
python -m unittest discover -s tests -v       # 33 个离线测试，不联网
```

祝比赛顺利。发现问题欢迎提 issue。

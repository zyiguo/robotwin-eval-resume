# RoboTwin 50 任务 × 10 次闭环评测与多次续跑计划

> **2026-10-16 部署更新：** 请先读[最新 agent 交接文档](../deployment/2026-10-16-agent-handoff.md)。当前迁移模型为 `formal_muon_mb2_ga50_20261016_010215` 第 150 步；本文中的 10 月 15 日模型/批次示例属于历史配置，不能直接作为本次默认值。完整环境归档已通过检查，补丁、仿真资产和渲染/设备验收仍待完成。

更新：2026-10-15。本文同时是实现说明与云端操作手册。**云端命令全部在 Linux/Bash 终端执行。** 本机 Windows 仅负责打包和软件测试。

## 1. 目标、当前实现与验收边界

最终累计完成 **50 个任务、每任务 10 个 episode，共 500 次**。每次会话最多 8 小时，允许同一实例重新启动或更换相同配置实例后续跑。8 小时不是完成全部评测的承诺；直到 500 个有效结果齐全才算完成。

模型来源固定为：

```text
/mnt/workspace/runtime/formal_muon_ga50_150_20261015_085411/best.json
```

`prepare` 首次读取其 `path`，核验导出并保存独立模型快照。以后 `run` 只读取本批次，不再追踪变化后的 `best.json`。如果 best 指向前一轮权重，实际快照以前一轮为准。

已实现入口为 `scripts/robotwin_eval.py`，提供 `prepare`、`run`、`status`、`report`。本项目的旧 `eval.sh` 和 `collect_results.py` 不作为这一方案的执行、续跑或汇总入口。上游评测循环仍负责场景、专家检查和策略执行；审计适配器在内存中增加记录，不修改共享上游源码或成功判据。

**验证范围必须分开：** 本地测试检查调度、审计、持久化、迁移和异常行为；实际 MI300X 模型加载、Vulkan/SAPIEN 渲染、真实 episode、Linux 强杀清理及跨实例续跑需要云端验收。不能把模拟的 500 项结果当作真实成绩。

## 2. 固定评测口径

| 项目 | 固定设置 |
|---|---|
| 任务 | LingBot 固定提交 `951475ae1b1d87553e7dc47c97b53a3d695c0d13` 启动器中的 50 个任务 |
| 场景、语言 | `demo_clean`、`seen` |
| 每任务次数 | 10；episode 编号 0～9 |
| 场景 seed | `seed_start = 100000 + episode编号`；每任务同样的 10 个 seed |
| 推理 | FP32、`use_bf16=False`、chunk 长度 50、关闭编译 |
| 规划 | MPLib；`ROBOTWIN_DISABLE_CUROBO=1`、`ROBOTWIN_EE_PLANNER=mplib` |
| 专家检查 | `expert_check=true`、`accept_expert_info_on_failure=true` |
| 并发 | 一张 GPU、一个驻留模型服务、一个仿真子进程 |
| 视频 | 关闭 |
| 默认端口 | 13400；占用时不终止已有进程，可指定其他空闲端口 |
| 每次预算 | 28800 秒；7 小时 45 分后不领取新 episode，8 小时到期结束本次拥有的服务与仿真进程 |
| 启动、单 episode 超时 | 默认分别 1800、3600 秒；仍受本次总预算约束 |

优先顺序为 `adjust_bottle`、`click_bell`、`handover_block`、`lift_pot`、`place_empty_cup`，每个任务各完成 10 次，然后按固定启动器顺序执行其余任务。首个 `adjust_bottle` episode 就是正式进度，不额外重复。500 项任务清单在准备时冻结。

本方案每次调用上游 `--test_num 1 --seed_start N`，不是一次启动连续 10 次；会增加仿真进程初始化开销。代价换来 episode 级提交和恢复。场景 seed 固定不代表 GPU 推理或随机语言选择具有逐位确定性；中断中的 episode 可能产生不同的重试轨迹，已提交结果不重跑。

## 3. 新实例恢复和依赖准备

### 3.1 保存并迁移哪些内容

沿用旧实例的**精确镜像 ID、系统架构和 Python 环境**。虚拟环境复用镜像中的 ROCm/PyTorch 和系统库，不能当作跨镜像的完整环境包。

| 必需内容 | 默认位置或说明 |
|---|---|
| 已安装的模型/仿真 venv | `/mnt/workspace/venvs/robotwin-env`，含其指向的基础解释器和共享系统依赖 |
| 已修改源码及本地依赖 | `/mnt/workspace/src`，包括 LingBot、RoboTwin、XPolicyLab 及符号链接实际目标 |
| Qwen 配置与 tokenizer | `/mnt/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer` |
| 选择记录 | 上述训练运行中的 `best.json` |
| 完整自训导出 | best 实际指向的全部权重分片、索引、配置、统计量和导出来源记录 |
| 仿真资产 | RoboTwin 的 `assets/objects`、`assets/embodiments`、`assets/background_texture` |
| 本次代码包 | `robotwin_eval_upload.zip` |

纯评测不要求迁移训练轨迹、原始 ZIP、训练优化器状态和验证缓存。训练环境迁移的分片打包与恢复方法见项目 `01_工程/正式训练说明.md` 第 5 节，按上表缩小保存范围。旧实例源文件保留，传输成功前不删除。

**尚未 prepare 时**，一定要复制 best 指向的完整权重，而不只是 best.json；训练程序可能清理旧导出。**已经 prepare 后**，迁移完整批次目录即可带上模型快照和进度。

### 3.2 上传独立工具包

本地生成包：

```powershell
python 01_工程/scripts/build_robotwin_eval_package.py
```

产物是 `02_运行归档/闭环评测/robotwin_eval_upload.zip`，附 SHA256 文件。它只含工具、任务清单、本文和固定 AMD 参考补丁，不含模型、数据、凭据或 Python 环境。

新实例上传至 `/mnt/workspace` 后，第一次解压：

```bash
cd /mnt/workspace
test ! -e /mnt/workspace/robotwin_eval || { echo '工具目录已存在，请使用独立目录解压，勿覆盖正在运行的代码'; exit 1; }
python -m zipfile -e robotwin_eval_upload.zip /mnt/workspace
source /mnt/workspace/venvs/robotwin-env/bin/activate
cd /mnt/workspace/robotwin_eval
python scripts/robotwin_eval.py --help
```

工具除 PyYAML 外使用 Python 标准库，不重新安装完整上游 requirements，也不替换镜像中的 torch。若当前 venv 没有 PyYAML，才执行 `python -m pip install 'PyYAML>=6,<7'`。

### 3.3 系统渲染和资产

MI300X 上的推理可用不证明相机渲染可用。本流程默认检查此前使用的 CPU 软件 Vulkan ICD：

```bash
test -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

在当前 Ubuntu 镜像尚未安装这些包时，执行：

```bash
apt-get update
apt-get install -y libvulkan1 vulkan-tools mesa-vulkan-drivers
```

工具只给自己启动的子进程设置 ICD、EGL 和私有 `XDG_RUNTIME_DIR`；不修改全局 shell 配置。若系统实际 ICD 文件名不同，准备与续跑均用 `--vulkan-icd /实际路径.json`。ICD 内容参与身份检查，不可静默切换渲染后端。

缺少仿真资产时，先激活 venv，在 RoboTwin 根目录执行固定版本下载。已有完整资产则跳过，不重复解压覆盖：

```bash
cd /mnt/workspace/src/RoboTwin
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import snapshot_download

names = ['background_texture.zip', 'embodiments.zip', 'objects.zip']
root = Path('/mnt/workspace/robotwin_asset_download')
snapshot_download(
    repo_id='TianxingChen/RoboTwin2.0', repo_type='dataset',
    revision='a967b852afa21a9cbf19a198f7e653109042e87c',
    allow_patterns=names, local_dir=str(root))
for name in names:
    with ZipFile(root / name) as archive:
        archive.extractall('assets')
PY
python scripts/update_embodiment_config_path.py
```

ModelScope 实例不能访问下载源时，可在本机下载同一版本再上传。先核对个人磁盘配额；下载压缩包、解压资产和模型快照可能同时占空间。不要以文件系统总容量代替个人剩余配额。

每次 `run` 在预算开始前检查 GPU、NumPy `1.26.4`、Transformers `4.57.3`、Vulkan 枚举、SAPIEN 离屏 PNG 和 RoboTwin 自身的渲染初始化。基础离屏示例通过仍不代表真实场景通过；真实场景由第一个正式 episode 验收。

### 3.4 缺少 AMD 补丁时

准备会检查推理端的 `LINGBOTVLA_TRAINING_CONFIG`，仿真端的 `lingbot_vla_v2`、`--seed_start`、专家检查参数以及 MPLib/CuRobo 开关。缺失则停止，不带病运行。

上传包 `reference_patches` 中提供原固定提交的 LingBot 与 RoboTwin 补丁。在**新实例上未被使用的恢复副本**中，先检查补丁是否完整可应用，再应用：

```bash
git -C /mnt/workspace/src/RoboTwin apply --check \
  /mnt/workspace/robotwin_eval/reference_patches/robotwin-rocm-reproduction.patch
# 只有上一条成功才执行：
git -C /mnt/workspace/src/RoboTwin apply \
  /mnt/workspace/robotwin_eval/reference_patches/robotwin-rocm-reproduction.patch

git -C /mnt/workspace/src/lingbot-vla-v2 apply --check \
  /mnt/workspace/robotwin_eval/reference_patches/lingbot-vla-v2-rocm.patch
# 只有上一条成功才执行：
git -C /mnt/workspace/src/lingbot-vla-v2 apply \
  /mnt/workspace/robotwin_eval/reference_patches/lingbot-vla-v2-rocm.patch
```

对源码快照可使用 Git 的补丁应用功能，无需重新克隆。若已完整应用，`git apply --reverse --check` 应通过，可跳过。若正向和反向检查都失败，说明存在部分补丁或其他修改：保留输出，针对差异修复；不要强行覆盖已跑通的训练源码。所有兼容修复应在 `prepare` 前结束；改变冻结后的评测代码必须新建批次。

## 4. 第一次冻结模型与清单

选择固定批次目录，后续不要每次创建新名字：

```bash
source /mnt/workspace/venvs/robotwin-env/bin/activate
cd /mnt/workspace/robotwin_eval
export BATCH=/mnt/workspace/runtime/robotwin_eval/clean50_best_20261015

python scripts/robotwin_eval.py prepare \
  --parent-run /mnt/workspace/runtime/formal_muon_ga50_150_20261015_085411 \
  --batch-dir "$BATCH"
```

若权重在新实例改了位置，可添加 `--model-source /实际迁移的/hf_ckpt`。工具核验 best 的步数、运行身份及导出结构，记录显式映射；不得用另一个模型冒充 best。

`prepare` 执行以下操作：

1. 检查配置、统计量、导出记录、权重索引和 safetensors 数据完整性；检查 50 任务与固定版本相符。
2. 把完整权重保存到 `$BATCH/model`。同一文件系统优先为不可变权重分片创建硬链接，非权重文件单独复制；硬链接不支持时复制分片。因此训练端删除原目录后，快照仍可存在。**不得原地改写快照或原硬链接分片。**
3. 保存 best、原路径、模型哈希、统计量、Qwen 文件、相关源码、仿真资产、工具代码、关键 Python 包版本和 ICD 身份。
4. 最后发布 `manifest.json`。没有该文件的准备目录不算有效批次，不能运行。

模型和全部资产的内容校验会读取较多文件，每次续跑也会校验；这是预算开始前的检查时间，单独体现在会话的准备阶段。若准备失败，保留原目录诊断，使用新的空批次目录重新准备，不覆盖半成品。

## 5. 每次运行、查看进度与停止

### 5.1 前台运行

```bash
source /mnt/workspace/venvs/robotwin-env/bin/activate
cd /mnt/workspace/robotwin_eval
export BATCH=/mnt/workspace/runtime/robotwin_eval/clean50_best_20261015

python -u scripts/robotwin_eval.py run --batch-dir "$BATCH"
```

每次运行先核验正式结果、输入身份和渲染环境，然后启动模型服务，**从启动服务开始计 8 小时**。服务驻留，仿真每次只跑一个指定 seed 的 episode。成功提交时输出 `COMMITTED: 任务/编号 success或failure total=N/500`。

`run` 正常退出不等于 500 项完成，必须检查输出中的 `complete` 和 `completed`。基础设施异常、时限退出或未完成项都可能留下后续工作。

### 5.2 后台运行

```bash
export SESSION_LOG="$BATCH/session_$(date +%Y%m%d_%H%M%S).console.log"
nohup python -u scripts/robotwin_eval.py run --batch-dir "$BATCH" \
  > "$SESSION_LOG" 2>&1 </dev/null &
echo $! > "${SESSION_LOG}.pid"
printf 'scheduler pid=%s\nlog=%s\n' "$(cat "${SESSION_LOG}.pid")" "$SESSION_LOG"
```

日志和 PID 文件每次单独保存。关闭 Notebook 页面不应作为停止手段。需要人工停止时，核对本次 PID 对应的进程和命令后向**该调度器**发送 SIGTERM；不要使用 `pkill python` 等广泛终止命令。前台 Ctrl+C 也会结束本次子进程并保存已有结果。

7 小时 45 分后停止领取新 episode，最多用余下 15 分钟收尾；8 小时到期结束本次子进程，清理可能额外需要约一秒。当前 episode 没有正式提交时，下次重跑它，已提交结果不动。默认单 episode 一小时超时；若实测正常 episode 确实更慢，可用 `--episode-timeout-seconds 7200`，仍不能超过单次总预算。

工具不会调用 ModelScope 关闭实例接口。预算结束后如需停止平台计费，由用户在平台操作。

### 5.3 查看与汇总

另一个终端中：

```bash
cd /mnt/workspace/robotwin_eval
python scripts/robotwin_eval.py status --batch-dir "$BATCH"
```

`status` 是只读快照，可以在运行中使用，不需要 GPU 或加载模型。会校验已发布结果及其证据；如果正在提交，快照可能暂时少一个结果，下次查看会更新。

会话结束后，可重建报告：

```bash
python scripts/robotwin_eval.py report --batch-dir "$BATCH"
```

报告为 `$BATCH/report.json` 与 `$BATCH/report.md`。`report` 写文件，需要获得批次锁；运行中使用 `status`，不要同时重建报告。

### 5.4 下次续跑

重新激活环境，使用**同一条命令和同一个 BATCH**：

```bash
source /mnt/workspace/venvs/robotwin-env/bin/activate
cd /mnt/workspace/robotwin_eval
export BATCH=/mnt/workspace/runtime/robotwin_eval/clean50_best_20261015
python -u scripts/robotwin_eval.py run --batch-dir "$BATCH"
```

无需再次 `prepare`。新一轮独立获得最多 8 小时，正式结果累计保留。正常失败与成功一样都跳过，不能为提升成绩重跑失败 episode。

## 6. 续跑如何避免丢进度和误计成绩

正式结果以 `(批次, task, episode编号)` 唯一定位，文件在 `results/<task>/<编号>.json`。结果先写临时文件、刷新到磁盘，再通过不覆盖已有文件的原子发布生成正式文件。只有正式文件计数；`.pending`、内存计数和终端输出不计数。

每次尝试保存独立的 `request.json`、`worker.log`、`guard.json`、`audit.json`、上游 `_result.txt` 和可用的原始输出。正式结果附这些证据的内容摘要，读取时重新核验。

提交必须同时满足：

- 子进程正常退出，且没有被守护进程超时或中断结束。
- 恰好审计到一个 episode，实际 seed 与本项 seed 一致。
- 结束前 `check_success()` 与上游 episode 成功标志一致。
- 上游最终统计恰好一次，分母为 1，与审计结果相符。
- 没有被上游捕获的执行异常或 rollout traceback，原始结果文件齐全。

评测循环的异常分支在内存中记录；不依赖“出现最终统计就算完成”。被捕获的通信异常即使输出 `0/1` 也不能成为模型正常失败。正常执行结束而 `check_success()` 为 false，才计为 `failure`。

若上游尝试跳过指定 seed，工具停止该尝试，不擅自改用其他 seed。每个 episode 每次会话最多自动重试两次，三次均异常则暂缓并继续缺项；连续三个 episode 出现可识别的共同渲染错误，或模型服务故障时，停止本次会话，避免大量无效运行。普通任务或 seed 异常不阻止其他任务获得评测机会。

同一批次使用操作系统文件锁保护。锁文件存在不等于锁仍被持有；进程退出后操作系统释放锁，因此不需要手工删除“旧锁”。不得删除正在使用的锁文件。Linux 守护进程在调度器失联、信号或预算到期时结束它自己创建的进程组，不终止其他用户进程。

若调度器被强杀：已发布正式结果仍有效；未提交的一项下次重新执行。若仿真已完成但调度器尚未来得及提交就被杀，该项也重跑，不凭不完整现场推断成绩。旧守护进程可能需要一两秒清理；若端口仍占用，先确认旧进程收尾，不强杀未知占用者。

结果损坏、重复、缺证据或模型身份变化时会拒绝继续，报告对应位置。不要手工改正式结果或删正常失败；保留整个批次供诊断。恢复进度不得依赖单份 report.json。

## 7. 更换实例后的续接

1. 停止旧调度器，确认其模型和仿真子进程已退出，保存一次报告。不要在正在写入时打包，也不要同时在两台实例启动同一批次的两个副本。
2. 备份**完整 `$BATCH`**，包括 `model`、`manifest.json`、`selection.json`、`results`、`sessions`、原始证据及报告。跨机器传输必须复制硬链接目标的真实内容；普通 tar 对批次内唯一的权重链接会包含文件内容。文件系统内部硬链接不会要求新实例保留原训练目录。
3. 迁移相同版本工具包、源码、环境、Qwen 文件和资产。符号链接指向的外部依赖也要迁移；批次本身不包含完整源码和环境。
4. 新实例恢复后，先执行 `status`，确认完成数、成功数和失败数与备份前一致，再 `run`。
5. 若目录变化，通过参数指定新位置，模型快照默认仍在批次内部：

```bash
python -u scripts/robotwin_eval.py run \
  --batch-dir /新位置/clean50_best_20261015 \
  --lingbot-root /新位置/src/lingbot-vla-v2 \
  --robotwin-root /新位置/src/RoboTwin \
  --qwen-path /新位置/models/Qwen3-VL-4B-Instruct-config-tokenizer \
  --python /新位置/venvs/robotwin-env/bin/python
```

源码中已知根目录会归一化后比较，因此仅改变绝对位置可以恢复。依赖版本、代码行为、统计量、资产和权重内容变化仍会拒绝接续。RoboTwin 机器人配置中的绝对路径确需更新时，只改成新根目录的对应路径，不能改其他语义参数。工具包解压后的脚本需保持原样；修改工具会改变身份，已有批次不混用。

完整批次可使用与训练手册相同的流式 tar 分片方式备份。例如在目录稳定、配额足够时：

```bash
set -o pipefail
export BACKUP_DIR=$(mktemp -d /mnt/workspace/eval_backup.XXXXXX)
tar -cf - -C "$(dirname "$BATCH")" "$(basename "$BATCH")" \
  | split -b 2000000000 -d -a 3 - "$BACKUP_DIR/eval.tar.part-"
printf '%s\n' "$BACKUP_DIR"
```

下载全部分片到外部备份；新实例在**空的目标父目录**恢复：

```bash
set -o pipefail
cat /实际上传目录/eval.tar.part-* | tar -xf - -C /实际恢复父目录
```

所有分片必须齐全且从 000 连续，恢复后 `status` 会校验逐项证据，`run` 还会校验模型等输入。确认恢复成功前保留旧副本。

## 8. 累计报告和最终完成标准

报告分别列出任务覆盖数、完成全部 10 次的任务数、已完成 episode／500、成功、正常失败、异常尝试、缺项、每次会话耗时和累计耗时。

剩余时间依据已完成 episode 的平均耗时估算；它不含未来模型启动和重试成本，且任务差异可能很大。被强杀会话的耗时只能由持久化的最后进度或守护记录恢复，可能是下界，不能称为精确计费时间。

未完成时显示“部分完成，等待续跑”；逐任务分母为该任务已完成次数，并明确目标为 10。未完成项不能计为模型失败，也不能让已有结果冒充 500 次成绩。

只有同时满足下列条件才标记 `complete=true`：

- 50 个固定任务均存在 10 个校验通过的正式结果。
- 总数恰好 500，无重复、缺项和外来模型结果。
- 每个结果均有匹配身份与完整证据。

此时计算每任务成功率、50 任务宏平均和总体成功率。正常失败也计入分母，只有成功数不同。全部任务样本数相等时，宏平均与总体成功率相同。

## 9. 本地和云端验收

本地命令：

```powershell
python -m unittest discover -s 01_工程/tests -p test_robotwin_eval.py -v
```

2026-10-15 本地验收：Windows / Python 3.12.7，共 29 项，28 项通过，1 项 Linux 专属 SIGKILL 守护测试跳过，0 失败、0 错误。完整日志与结构化记录保存在项目 `02_运行归档/闭环评测`。这不是云端硬件验收。

本地验收覆盖：中断后不重跑已完成项；正常失败保留；捕获异常后的最终 0/1 拒绝；seed 和成功判据不一致拒绝；原子写入、磁盘满、证据损坏与重复结果；身份变化；进程锁；目录迁移；缩短预算退出；累计 500 项含失败仍能正确完成；固定上游和真实 AMD 补丁的 AST 兼容；真实子进程超时清理。使用模拟子进程和轻量环境/模型替身的测试结果不作为真实性能或成功率。

Linux 上另测 SIGKILL 调度器后的守护清理；Windows 本地会显式跳过这一项，不能写成通过。

云端逐项验收并保留日志：

1. `prepare` 成功，记录 batch ID、模型来源和 fingerprint。
2. 第一个真实 episode 正常完成并出现一份正式结果。成功或正常失败均可，基础设施异常不算。
3. 人为在另一个 episode 进行中停止本次会话，确认已提交结果不变。
4. 再次 `run`，确认只执行未提交项，正常失败不重跑。
5. 重新进入同一实例再续跑；若换实例，恢复备份后再验证一次。
6. 多次累计到 500 个正式结果后核对报告。若硬件渲染或长任务始终阻塞，保留缺项和证据，不能宣称已完成。

## 10. 常见阻塞的处理

| 情况 | 处理 |
|---|---|
| `ERROR_INCOMPATIBLE_DRIVER`、无 PNG 或 RoboTwin 渲染失败 | 看本次 `sessions/.../preflight` 日志；核对 ICD、系统库和软件渲染。基础示例通过不能替代真实相机。留在当前实例定位，不自动切换硬件。 |
| 缺 AMD 参数或协议 | 按第 3.4 节核对补丁；在冻结前修复，不能边评测边改共享源码。 |
| `Evaluation identity changed` | 检查报告中的模型、代码、资产、包版本或 ICD 哪一类变化；恢复原内容或建立新批次，不修改 manifest 绕过。 |
| 端口占用 | 确认占用者，等待旧会话清理，或使用另一个端口；工具不杀未知进程。 |
| 单个 seed 不稳定 | 保留三次尝试，后续会话再试同一 seed；不静默替换成容易成功的 seed。 |
| 磁盘满、写盘失败 | 不生成虚假进度，保留原始数据；释放安全的外部传输临时文件或扩大可用存储后续跑，不删正式结果。 |
| 8 小时未完成 | 正常现象，使用同一 BATCH 再启动；累计进度不清零。 |
| 工具正常退出但 `complete=false` | 查看 pending 和异常记录，不把退出码 0 当成全量完成。 |

参考来源：项目固定版本源码备份；[LingBot RoboTwin 入口](https://github.com/Robbyant/lingbot-vla-v2/tree/951475ae1b1d87553e7dc47c97b53a3d695c0d13/experiment/robotwin)；[固定 AMD 复现说明](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/Reproduce_Guide.md)。

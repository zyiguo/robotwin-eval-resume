# RoboTwin 新实例部署交接：2026-10-16

这是给接手部署 agent 的当前状态和执行入口。请先读本文，再读[完整操作手册](../plans/2026-10-15-robotwin-closed-loop-evaluation.md)。本文更新了模型选择、迁移文件检查结果和待办；旧手册、旧清单及上传 ZIP 内说明的 10 月 15 日默认路径属于历史配置，不能直接沿用。

## 1. 用户目标和接手边界

- 在相同配置的 ModelScope MI300X 实例上，累计完成 RoboTwin 50 个任务 × 10 个 episode，共 500 项正式闭环评测。
- 固定 `demo_clean + seen`、每任务 seed 100000～100009、FP32、chunk/use_length=50、关闭编译、MPLib、关闭 CuRobo、关闭视频；单模型服务与单仿真进程。
- 每次会话最多 8 小时，可跨会话、实例重启及迁移继续；8 小时不是完成 500 项的承诺。
- 只跳过已提交的正式结果，正常任务失败也计入完成；基础设施异常保留为待重试。
- 现有调度工具已经实现；当前工作是恢复部署和设备端验收，不是重新训练或重写评测系统。
- 此仓库公开，只含工具、文档和小型检查证据。模型、环境归档、完整修改版源码及仿真资产不在 GitHub 中；接手时要确认它们已经上传或挂载到新实例，不能从 `git clone` 成功推断迁移完成。

## 2. 当前模型：必须显式指定

```text
PARENT_RUN=/mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215
MODEL_SOURCE=$PARENT_RUN/checkpoints/global_step_150/hf_ckpt
run_id=60e52e4910a94139883bf819ff7deffc
global_step=150
```

本地 `best.json` 与 `checkpoints/latest.json` 都指向本轮 step 150。7 个 safetensors 分片和完整索引、训练配置、归一化统计量、导出记录、来源记录、tokenizer/processor 已具备；现有工具的 `checkpoint_files()` 检查通过，共 22 个导出文件。导出记录的张量字节总数为 12,452,772,206，浮点权重保存为 BF16；评测推理配置仍是 FP32。

全部 22 个导出文件的本地 SHA256 已公布在 [model-sha256-20261016.txt](evidence/model-sha256-20261016.txt)。它是本次本地副本的传输比对基准，不是另行取得的旧实例原始模型校验清单，也不代表实际 GPU 加载成功。

训练记录中的验证损失不是 RoboTwin 闭环成功率。无需迁移初始化模型所属的上一轮权重、原始训练轨迹、优化器或训练验证缓存；本次导出已含统计量和完整推理权重。不要改写原训练配置/来源记录，应让 `prepare` 生成独立评测配置。

**禁止省略 `--parent-run` 使用工具的旧默认路径** `/mnt/workspace/runtime/formal_muon_ga50_150_20261015_085411`。新模型建立新的批次，不与已冻结为其他模型的结果混合。

## 3. 本地迁移材料及验证状态

用户的本地目录是 `C:\Users\root\Desktop\蚂蚁灵波\实例迁移`，它不是新实例路径；不要尝试在 Linux 使用这个 Windows 路径。

| 材料 | 当前状态 | 新实例目标 |
|---|---|---|
| `formal_muon_mb2_ga50_20261016_010215/` | 模型结构通过，含 best、step150 导出 | `/mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215/` |
| `robotwin-env-complete-20261016_114514.tar.gz` | **有效的完整环境归档** | 解压至 `/mnt/workspace/venvs/`，得到 `robotwin-env/` |
| 环境 `.sha256` | 与本地归档一致；原下载文件名带 ` (1)` | 仓库已附标准命名校验文件 |
| 散装 `robotwin-env/` | 历史不完整副本，不用它恢复 | 使用上面的有效归档 |
| `Qwen3-VL-4B-Instruct-config-tokenizer/` | 11 文件，JSON 可解析 | `/mnt/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer/` |
| `src/` | 核心源码已恢复，补丁和 XPolicyLab 缺项待处理 | `/mnt/workspace/src/` |
| `migration_record/` | OS、包、依赖路径、venv 配置及 Vulkan 失败记录已收集 | 保留为部署审计材料 |
| 精确 ModelScope 镜像 ID | **尚未提供** | 向用户取得，或从其旧实例详情确认 |
| 物体/机器人/背景资产 | **本地尚无实际内容** | 从旧实例补传，或在新实例下载固定版本 |
| 正式评测批次 | 当前迁移材料中未发现 | 如用户另有批次，先检查完整 manifest/results/sessions 再决定新建 |

### 已通过的环境归档检查

```text
文件：robotwin-env-complete-20261016_114514.tar.gz
大小：920411674 字节
SHA256：b609bf58ce158dfe484b0cdf576285a99d05914e6461c4eedffcec9b76b34cec
```

- SHA256 与用户从旧实例下载的校验文件一致；gzip/tar 全流可读取，未截断，检查期间文件未变化。
- 有 29,476 个文件/链接条目，普通文件内容合计 2,578,308,774 字节。
- 95 份包 RECORD 的 29,467 个路径均在归档内找到；这是路径存在性检查，没有逐项复算 RECORD 内容哈希。
- `pyvenv.cfg`、解释器软链接、`lib64 -> lib` 均保留；此前漏拷的 Transformers、SAPIEN、Open3D、safetensors、PyYAML、numpydantic 模块入口已找到。
- 先前 591,921,152 字节的 `robotwin-env-complete.tar.gz` 曾经检查到截断错误，不能使用；有效文件是带 `20261016_114514` 的新归档。

## 4. 基础镜像：不能用参考项目的镜像替代

已收集的旧实例信息：

| 项目 | 值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.13，基础解释器 `/usr/bin/python3.12` |
| venv | `/mnt/workspace/venvs/robotwin-env`，`include-system-site-packages = true` |
| Torch | `2.12.0+git6bbd260`，来自 `/usr/local/lib/python3.12/dist-packages` |
| ROCm（torch.version.hip） | `7.2.53211` |
| tokenizers | `0.22.2`，来自系统 dist-packages |
| venv 内依赖 | Transformers 4.57.3、SAPIEN 3.0.3、Open3D 0.19.0、safetensors 0.5.3、NumPy 1.26.4、PyYAML 6.0.2、MPLib 0.2.1、numpydantic 1.6.11 |

完整历史包列表见 [robotwin_packages.json](evidence/robotwin_packages.json)，模块安装位置见 [package_locations.json](evidence/package_locations.json)。341 项包列表包含系统继承包，不应对新镜像直接整体升级/重装；环境归档不包括系统 Torch/ROCm、标准库及系统动态库。参考 AMD 项目的 Ubuntu/PyTorch 组合不能当作本次旧实例镜像。

准确镜像 ID 是当前尚缺的信息。能够并行推进源码和资产准备，但宣称环境恢复成功之前，必须确认系统解释器、Torch/ROCm 和动态库兼容。不要自动换实例类型或升级 GPU 软件栈。

## 5. 部署顺序（以下命令仅在新实例 Linux/Bash 执行）

### A. 上传材料，保留原路径恢复

先核对个人存储配额。新实例应收到模型运行目录、有效环境 tar.gz、Qwen 配置目录和实际源码；不要仅传 `best.json`。若目标已有同名目录，先检查其用途，不覆盖活动环境或已开始的评测。

示例假定环境归档已放在 `/mnt/workspace`：

```bash
cd /mnt/workspace
git clone https://github.com/zyiguo/robotwin-eval-resume.git
export TOOLS_REPO=/mnt/workspace/robotwin-eval-resume

sha256sum -c "$TOOLS_REPO/docs/deployment/evidence/robotwin-env-complete-20261016_114514.tar.gz.sha256"
gzip -t robotwin-env-complete-20261016_114514.tar.gz
```

两项检查成功后，再恢复 venv；目标存在时停止，不能通过删除检查强行覆盖：

```bash
mkdir -p /mnt/workspace/venvs
if [ -e /mnt/workspace/venvs/robotwin-env ]; then
  echo '目标环境已存在，先检查，暂不解压覆盖'
else
  tar -xzf /mnt/workspace/robotwin-env-complete-20261016_114514.tar.gz \
    -C /mnt/workspace/venvs
fi
```

确认解释器、镜像匹配后，激活环境并在模型目录核对传输：

```bash
source /mnt/workspace/venvs/robotwin-env/bin/activate
python -c "import sys,torch,yaml,transformers,sapien,open3d,safetensors,numpydantic; print(sys.version); print(sys.prefix); print(torch.__version__,torch.version.hip); print('GPU:',torch.cuda.is_available())"

cd /mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215/checkpoints/global_step_150/hf_ckpt
sha256sum -c "$TOOLS_REPO/docs/deployment/evidence/model-sha256-20261016.txt"
```

这条导入检查可能暴露真实动态库问题，应保留完整输出，不把 `find_spec()` 的历史路径记录当作真实导入成功。

### B. 解压现有评测工具

仓库已有经过本地软件测试的工具包，不需重新实现。首次解压前确认 `/mnt/workspace/robotwin_eval` 不存在；若存在，检查版本与是否运行中。

```bash
python -m zipfile -e "$TOOLS_REPO/artifacts/robotwin_eval_upload.zip" /mnt/workspace
cd /mnt/workspace/robotwin_eval
python scripts/robotwin_eval.py --help
```

ZIP 内 README 保留旧模型示例；本次所有模型/批次参数以本文显式命令为准。ZIP 不含此次新下载的大环境归档或权重。

### C. 源码、兼容补丁和 editable 路径

源码基线固定版本见 [sources.json](../../02_运行归档/源码备份/sources.json)，不要跟随上游最新分支。

- RoboTwin 核心 `envs/`、`scripts/`、`env_cfg/`、`description/` 已补齐。
- 但当前 `scripts/eval_policy_xpolicylab.py`、`envs/robot/planner.py`、`envs/robot/robot.py` 与固定原版文件逐字节相同，**尚未应用所需 RoboTwin 兼容补丁**。
- 当前缺少 `--seed_start`、`lingbot_vla_v2`、`--accept_expert_info_on_failure`、`ROBOTWIN_DISABLE_CUROBO`、`ROBOTWIN_EE_PLANNER`，会被现有工具预检拒绝。
- LingBot deploy 已有 `LINGBOTVLA_TRAINING_CONFIG`；不要盲目重复应用整个 LingBot 补丁。模型服务包装器会在内存中执行 MoE eval dtype 兼容处理，不应为了它改写旧训练来源文件。
- XPolicyLab `utils/` 已补齐，但 `pyproject.toml`、`setup_policy_server.py`、部分 `scripts/` 尚缺。优先按固定提交补缺失文件；重新安装 editable 包时先确认这些安装文件齐全，保留已有修改版 `client_server/` 等源码。其他 policy 的数千个缺项不全部属于本次 LingBot 评测必需项。
- venv editable 安装仍指向 `/mnt/workspace/src/...`，恢复原路径最省事；换路径则需修复 editable 安装，并重新核验导入。

在未被使用的新实例恢复副本中：

```bash
PATCH=/mnt/workspace/robotwin_eval/reference_patches/robotwin-rocm-reproduction.patch
git -C /mnt/workspace/src/RoboTwin apply --check "$PATCH"
# 仅正向检查成功后应用：
git -C /mnt/workspace/src/RoboTwin apply "$PATCH"
```

若正向检查失败，用 `git apply --reverse --check` 判断是否已完整应用；两者都失败就保留差异针对修复，不能强制覆盖。补丁的存在不等于已应用。代码修复必须在正式 `prepare` 前完成，已冻结后改变行为就应建立新批次。

### D. 渲染环境与资产

旧实例的最新记录显示：三个 Vulkan/Mesa 包查询不到，`vulkaninfo` 不存在，四个常见配置目录不存在，两个指定 ICD 的环境变量为空。这次不是“记录没下载”，而是**尚无可用渲染环境的证据**；不能声称 CPU Vulkan 曾在本次环境验收通过。有限检查也不能排除任意自定义路径下的库。

在新实例按主手册安装所需 Vulkan/Mesa 用户态依赖；不升级 ROCm/PyTorch。CPU Vulkan 是计划中的候选渲染路径，实际能否运行 SAPIEN/RoboTwin 必须实测。准备和续跑的默认 ICD 是 `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`；实际文件名不同就显式传 `--vulkan-icd /实际路径.json`，并保持身份一致。

资产目前只有下载辅助文件，以下目录无实际内容：

```text
/mnt/workspace/src/RoboTwin/assets/objects/
/mnt/workspace/src/RoboTwin/assets/embodiments/
/mnt/workspace/src/RoboTwin/assets/background_texture/
```

可按主手册从 `TianxingChen/RoboTwin2.0` 数据集的固定 revision `a967b852afa21a9cbf19a198f7e653109042e87c` 下载 `objects.zip`、`embodiments.zip`、`background_texture.zip`，解压后运行 `scripts/update_embodiment_config_path.py`。只有空目录或下载脚本不能通过资产就绪判断。

### E. 首次冻结和运行

模型、代码、资产和环境修复结束后，使用一个新的固定批次名。先确认这个目录不存在；若存在正式批次，核对身份与进度后走 `run`，不重复 prepare。

```bash
source /mnt/workspace/venvs/robotwin-env/bin/activate
cd /mnt/workspace/robotwin_eval
export BATCH=/mnt/workspace/runtime/robotwin_eval/clean50_best_20261016_step150

python scripts/robotwin_eval.py prepare \
  --parent-run /mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215 \
  --model-source /mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215/checkpoints/global_step_150/hf_ckpt \
  --batch-dir "$BATCH"

# 仅 prepare 成功、manifest 已发布后执行：
python -u scripts/robotwin_eval.py run --batch-dir "$BATCH"
```

`--model-source` 是显式迁移映射，不允许用其他模型冒充 best。若路径发生变化，按实际恢复位置填写 parent/model/source 路径参数。工具在准备前校验身份并冻结完整模型；首次模型选择成功后，不再跟踪变化的 best。

首个 `adjust_bottle` episode 是正式进度。当前 CLI 没有 `--max-episodes 1`，不要杜撰这个参数；若先观察一个 episode，可前台监看 `COMMITTED` 并使用 Ctrl+C 正常结束会话。调度器可能已开始下一项，未提交项下次重试，已提交项不重跑。第一次闭环先验证成功判定/正常失败与异常区分，然后再持续运行。

```bash
# 运行中另开终端，只读查看
python scripts/robotwin_eval.py status --batch-dir "$BATCH"

# 会话结束后重建报告
python scripts/robotwin_eval.py report --batch-dir "$BATCH"

# 下次续跑：仍用同一批次，不再 prepare
python -u scripts/robotwin_eval.py run --batch-dir "$BATCH"
```

每次 `run` 会先做环境/身份预检；模型服务启动时开始计算会话预算，默认 28,800 秒、预留 900 秒停止领新项。迁移、下载和启动前检查另记耗时。没有 CUDA/ROCm GPU、Vulkan 或真实场景验收时，不能用模拟测试替代。

## 6. 设备端验收与交付

历史 Windows 软件测试为 29 项：28 通过、1 个 Linux SIGKILL/孤儿进程守护测试跳过；本次文档发布没有把它重标为 Linux 已通过。测试入口见[本地软件测试说明](../运行本地软件测试.md)。

接手 agent 应完成并保存实际证据：

1. 匹配系统环境、GPU 检查、全部关键依赖真实导入和模型权重校验。
2. 源码补丁与 50 任务清单预检、资产检查、Vulkan 枚举、SAPIEN 离屏输出和 RoboTwin 渲染初始化。
3. 首个正式 episode 完成，证据中的实际 seed、最终分母和 check_success 与提交一致。
4. 人为中断后的续跑：成功与正常失败都跳过，进行中未提交项可重试；对 Linux 强杀清理及锁保护按主手册和测试验证，不能直接删除活锁或用 `pkill python`。
5. 重启后、以及后续换实例恢复完整批次后的接续检查；缩短预算时验证停止领取、截止清理与再续跑。
6. 记录准确镜像 ID、新实例包/设备/渲染证据、修复差异、部署命令及当前 `status`，明确哪些仍未通过。

换实例需停止旧会话、确认子进程退出后备份整个 `$BATCH`，包括 manifest、selection、model、results、sessions 和原始证据；不能只迁移 report。不得让同一批次的两个副本同时运行。

完成条件：50 个任务各有 10 个经过校验的正式结果，总计恰好 500；报告同时保留正常失败和运行异常，部分结果不称为完整成绩。保持用户要求的单次 8 小时预算，不自动关闭平台实例。

## 7. 证据和优先级

- [机器可读迁移状态](evidence/2026-10-16-migration-audit.json)
- [完整包版本清单](evidence/robotwin_packages.json)
- [依赖实际位置](evidence/package_locations.json)
- [原始 venv 配置](evidence/pyvenv.cfg)
- [模型文件校验清单](evidence/model-sha256-20261016.txt)
- [完整环境归档校验文件](evidence/robotwin-env-complete-20261016_114514.tar.gz.sha256)

优先处理：确认新实例材料位置与准确镜像 → 恢复有效 venv → 完成源码补丁和 XPolicyLab 依赖 → 下载资产及建立渲染链路 → prepare → 首 episode 和恢复验收 → 持续多轮 run。不要再次要求用户重复下载已经核验通过的模型和环境包。

# RoboTwin 50 × 10 评测与断点续跑

面向 ModelScope MI300X / ROCm 的 LingBot-VLA 2.0 闭环评测工具。固定评测 50 个 clean/seen 任务，每任务 10 次；每个 episode 单独保存结果，每次会话最多 8 小时，允许多次运行累计到 500 项。

- **[下载全部迁移文件：模型、环境及源码](https://github.com/zyiguo/robotwin-eval-resume/releases/tag/migration-20261016-step150)**（Release 附件；[下载恢复说明](docs/deployment/migration-release-download.md)）
- **[部署 agent 交接入口（2026-10-16，最新迁移状态）](docs/deployment/2026-10-16-agent-handoff.md)**
- **[闭环评测完整操作手册](docs/plans/2026-10-15-robotwin-closed-loop-evaluation.md)**
- **[旧实例需要补齐的文件清单](docs/旧实例补充文件清单.md)**
- **[下载 ModelScope 上传包](artifacts/robotwin_eval_upload.zip)**（约 43 KB，仅工具和说明）
- [本地验收记录](02_运行归档/闭环评测/验收记录.md)
- [重跑软件测试和重新打包](docs/运行本地软件测试.md)

> **2026-10-16 部署更新：** 本次迁移模型为 `formal_muon_mb2_ga50_20261016_010215` 第 150 步，须显式指定 `--parent-run`。完整环境归档已通过 SHA256 和包文件清单检查；RoboTwin 兼容补丁、仿真资产、渲染和设备端恢复验收仍待完成。先读上面的交接入口，勿沿用旧手册/ZIP 中的 10 月 15 日模型默认路径。

## 已实现

- `prepare`：读取指定运行的 best.json，冻结完整模型、500 项清单和输入身份。
- `run`：模型服务驻留，逐 episode 执行，只补未提交项；每轮默认最多 8 小时。
- `status`：运行中只读查看累计进度，不需要 GPU 或加载模型。
- `report`：从正式结果和证据重建逐任务报告，只有 500 项齐全才标记完整完成。

正常失败也属于已完成结果，不为提高成功率重跑。通信、渲染等执行异常保留为待重试；通过审计 seed、最终统计和 check_success()，防止把被上游捕获的异常误算成模型失败。换实例需要迁移完整评测批次目录，包括模型快照、逐项结果和原始证据。

## 开始使用

先按清单迁移旧实例的环境、已修改源码、完整自训导出和 Qwen 配置/tokenizer，并补齐仿真资产。Git 仓库和工具 ZIP 不内嵌这些大文件；完整迁移 Release 已提供本地目录的全部材料，按下载恢复说明另取附件。原目录未包含的仿真资产仍需补齐。

将 artifacts 中的 ZIP 上传到新实例 `/mnt/workspace`，解压得到 `robotwin_eval/`，激活恢复的 venv。按主手册首次执行 `prepare`，以后针对同一批次反复执行 `run`。不要用原项目旧 eval.sh 代替本工具的续跑入口。

## 验证状态

2026-10-15，Windows / Python 3.12.7：29 项测试，28 通过，1 个 Linux SIGKILL 守护测试跳过，零失败。测试包含真实子进程及在固定 AMD 评测循环中使用轻量环境/模型替身的检查。

**真实 MI300X 推理、Vulkan/SAPIEN 物理闭环、Linux 强杀清理和跨实例续跑尚需云端验收。** 本地模拟的 500 项结果不是模型成绩，8 小时也不是全量评测耗时承诺。

## 来源

上游版本与来源见 [sources.json](02_运行归档/源码备份/sources.json)。上游模型、数据、源码及兼容补丁分别遵循其原有许可。大体积上游源码 ZIP 不进入本仓库，软件测试时按说明下载固定版本。

# 完整迁移材料下载与恢复

发布标签：`migration-20261016-step150`。

发布地址：https://github.com/zyiguo/robotwin-eval-resume/releases/tag/migration-20261016-step150

本发布覆盖用户 `实例迁移` 目录全部 **12,255 个原始文件**，原始大小合计 **13,993,595,246 字节**。原始路径、大小、SHA256 和所在附件见 `MIGRATION_FILES.json`；不是只上传摘要或模型配置。

## 附件分工

- `model-00001-of-00007.safetensors` ～ `model-00007-of-00007.safetensors`：7 个原始模型权重分片。
- `robotwin-env-complete-20261016_114514.tar.gz`：已经核验的完整 Linux venv 归档，恢复环境使用这一份。
- `migration-support-20261016.tar.gz`：其余 12,247 个原始文件，保留原相对路径；包括模型配置、统计量、来源记录、日志、Qwen 配置/tokenizer、源码、环境记录、原校验文件和历史散装环境目录。
- `MIGRATION_FILES.json` / `MIGRATION_FILES.sha256`：原目录全文件清单和恢复后的逐文件校验基准。
- `RELEASE_ASSETS.sha256`：下载附件的 SHA256，包含除自身外的全部发布附件。
- `RESTORE.md` / `release-notes.md`：恢复说明和发布说明。

`migration-support` 中的散装 `robotwin-env/` 是曾经漏拷的历史副本，为完整保存用户目录而保留；**不能用它部署**。完整环境 tar.gz 保留了原 Linux 软链接和权限，但仍依赖镜像中的 Python 3.12、系统 Torch/tokenizers/ROCm 和动态库。

本发布的源码是当前迁移快照，尚需应用 RoboTwin 兼容补丁；物体、机器人和背景资产未存在于用户原目录，因此不包含在这次发布中。准确镜像 ID 也尚未取得。上传全部已有文件不等于这些待办已经完成。

## 1. 在新实例 Linux/Bash 下载并核验附件

先确认个人存储配额；下载、恢复和环境解压可能同时占用空间。在一个独立下载目录中执行：

```bash
mkdir -p /mnt/workspace/migration_download_20261016
cd /mnt/workspace/migration_download_20261016
BASE=https://github.com/zyiguo/robotwin-eval-resume/releases/download/migration-20261016-step150

curl --fail --location --retry 3 "$BASE/RELEASE_ASSETS.sha256" \
  -o RELEASE_ASSETS.sha256

while read -r checksum filename; do
  curl --fail --location --retry 3 --continue-at - \
    "$BASE/$filename" -o "$filename" || break
done < RELEASE_ASSETS.sha256

sha256sum -c RELEASE_ASSETS.sha256
```

只有全部校验通过才继续。若单个文件下载中断，可重新运行循环，支持的服务器会从已有文件位置接续；若服务器拒绝范围请求，保留现场，换一个新文件名完整重下该附件，不能把损坏文件当成完成。附件名称均为 ASCII 且没有空格。

## 2. 恢复完整原目录并逐文件核验

以下目录必须是新的；不要覆盖已经部署或正在运行的目录：

```bash
DOWNLOAD=/mnt/workspace/migration_download_20261016
RESTORED=/mnt/workspace/migration_restored_20261016

test ! -e "$RESTORED" || { echo '恢复目标已存在，请先检查，停止覆盖'; exit 1; }
mkdir -p "$RESTORED"
tar -xzf "$DOWNLOAD/migration-support-20261016.tar.gz" -C "$RESTORED"

MODEL="$RESTORED/formal_muon_mb2_ga50_20261016_010215/checkpoints/global_step_150/hf_ckpt"
mkdir -p "$MODEL"
for number in 00001 00002 00003 00004 00005 00006 00007; do
  file="model-${number}-of-00007.safetensors"
  ln "$DOWNLOAD/$file" "$MODEL/$file" || cp -p "$DOWNLOAD/$file" "$MODEL/$file"
done

ENV_ARCHIVE=robotwin-env-complete-20261016_114514.tar.gz
ln "$DOWNLOAD/$ENV_ARCHIVE" "$RESTORED/$ENV_ARCHIVE" \
  || cp -p "$DOWNLOAD/$ENV_ARCHIVE" "$RESTORED/$ENV_ARCHIVE"

cd "$RESTORED"
sha256sum -c "$DOWNLOAD/MIGRATION_FILES.sha256"
```

在同一文件系统中优先用硬链接减少重复存储；不要原地改写这些权重或环境归档。逐文件核验会读取全部材料，耗时需单独记录。

## 3. 转入部署流程

完整原目录通过检查后，按[部署交接文档](https://github.com/zyiguo/robotwin-eval-resume/blob/main/docs/deployment/2026-10-16-agent-handoff.md)把材料放到新实例标准位置，或向评测工具显式传实际路径。检查目标用途，保留已有内容，不覆盖活动环境。

```text
完整 venv：解压至 /mnt/workspace/venvs/robotwin-env
源码：/mnt/workspace/src
Qwen 配置：/mnt/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer
模型运行：/mnt/workspace/runtime/formal_muon_mb2_ga50_20261016_010215
```

不要从历史散装 venv 启动。恢复完整归档后核对基础镜像、真实依赖导入和 GPU；再补兼容代码、资产与 Vulkan/SAPIEN，完成首 episode、中断和重启接续验收。

当前模型为 run_id `60e52e4910a94139883bf819ff7deffc`、step 150；首次 `prepare` 必须显式指向 10 月 16 日运行，不能沿用旧模型默认值。目标仍为 50 × 10，单次最多 8 小时，多轮累计；正常失败也保留，不能为提高成绩重跑。

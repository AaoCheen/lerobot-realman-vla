#!/usr/bin/env bash
# 激活独立.venv后运行。先浅克隆固定的Transformers，再从本地源码安装。
# 避免pip的普通VCS依赖默认抓取整个Git历史；不修改已运行的pip进程。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
COMMIT=dcddb970176382c0fcf4521b0c0e6fc15894dfe0
SOURCE="$ROOT/.train/transformers-pi05-src"
if [[ "${1:-}" == --help ]]; then
    echo '激活独立.venv后执行 bash scripts/install_pi05_env.sh；--download-only只准备源码。'
    exit 0
fi
# uv会把裸名字“python”当作环境搜索条件；显式传当前解释器的绝对路径，
# 才能准确定位Conda环境，无需--system，也无需重建venv。
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
if [[ "${1:-}" != --download-only ]]; then
    "$PYTHON" - <<'PY'
import os,sys
from pathlib import Path
if not os.environ.get('CONDA_PREFIX') and sys.prefix == sys.base_prefix:
    raise SystemExit('请先激活独立的vla-pi05环境，不要装到base Python。')
if sys.version_info[:2] not in [(3,10),(3,11)]:
    raise SystemExit('训练环境需要Python3.10/3.11。')
print('安装解释器：', sys.executable)
PY
fi
mkdir -p "$ROOT/.train"
if [[ ! -d "$SOURCE/.git" ]]; then
    [[ ! -e "$SOURCE" ]] || { echo "源码目录存在但不是完整Git仓库：$SOURCE，请检查后重试" >&2; exit 1; }
    git -c http.version=HTTP/1.1 clone --depth 1 --single-branch \
        --branch fix/lerobot_openpi --progress \
        https://github.com/huggingface/transformers.git "$SOURCE"
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$COMMIT" ]]; then
    [[ -z "$(git -C "$SOURCE" status --porcelain)" ]] || { echo 'Transformers源码有本地修改，停止避免覆盖' >&2; exit 1; }
    git -C "$SOURCE" -c http.version=HTTP/1.1 fetch --depth 1 origin "$COMMIT"
    git -C "$SOURCE" checkout --detach "$COMMIT"
fi
"$PYTHON" - "$ROOT" "$SOURCE" <<'PY'
from pathlib import Path
import sys
root,source=map(Path,sys.argv[1:])
lines=(root/'requirements-train-pi05.txt').read_text().splitlines()
replacement=0; result=[]
for line in lines:
    if line.startswith('transformers @ git+'):
        result.append(str(source))
        replacement+=1
    else: result.append(line)
if replacement != 1: raise SystemExit('依赖文件不符合预期，请检查Transformers来源')
# 主requirements保持可审阅的固定提交；本地生成文件只替换下载方式。
(root/'.train/requirements-pi05-local.txt').write_text('\n'.join(result)+'\n')
print('源码及依赖已准备：',root/'.train/requirements-pi05-local.txt')
PY
[[ "${1:-}" == --download-only ]] && exit 0
if command -v uv >/dev/null 2>&1; then
    exec uv pip install --python "$PYTHON" \
        --index-url "${PI05_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
        -r "$ROOT/.train/requirements-pi05-local.txt"
fi
exec "$PYTHON" -m pip install -r "$ROOT/.train/requirements-pi05-local.txt"

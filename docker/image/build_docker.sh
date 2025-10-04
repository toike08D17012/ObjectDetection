#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# ==== 設定（必要なら変更）====
USER_NAME="${USER_NAME:-$USER}"
GROUP_NAME="${GROUP_NAME:-$USER_NAME}"

# bash の組込み $UID をそのまま使う（上書きしない）
USER_UID="${UID}"
USER_GID="$(id -g)"
USER_HOME="/home/${USER_NAME}"
IMAGE_NAME="object_detection_practice"
DOCKERFILE="./Dockerfile"
CONTEXT=".."   # レポジトリルートを想定

# ==== 実行 ====
echo "Build args:"
echo "  USER_NAME=${USER_NAME}"
echo "  GROUP_NAME=${GROUP_NAME}"
echo "  USER_UID=${USER_UID}"
echo "  USER_GID=${USER_GID}"
echo "  HOME=${USER_HOME}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE}" \
  --build-arg USER_NAME="${USER_NAME}" \
  --build-arg GROUP_NAME="${GROUP_NAME}" \
  --build-arg USER_UID="${USER_UID}" \
  --build-arg USER_GID="${USER_GID}" \
  --build-arg HOME="${USER_HOME}" \
  "${CONTEXT}"
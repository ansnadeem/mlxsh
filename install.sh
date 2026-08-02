#!/bin/sh
# Install mlxsh, and whatever it needs to run.
#
#   curl -LsSf https://raw.githubusercontent.com/ansnadeem/mlxsh/main/install.sh | sh
#
# Installs uv if it is missing (uv brings its own Python, so nothing else is
# required), then installs mlxsh as a tool. Pass --vision to include mlx-vlm.
#
#   MLXSH_REF=v0.2.1   install a tag or branch instead of PyPI
#   --dry-run          print the commands instead of running them

set -eu

PKG="mlxsh"
EXTRA=""
DRY=""

for arg in "$@"; do
    case "$arg" in
        --vision) EXTRA="[vision]" ;;
        --dry-run) DRY="echo   would run:" ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

case "$(uname -s)" in
    Darwin) ;;
    *) echo "mlxsh needs macOS on Apple silicon (MLX is Apple only)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
    arm64) ;;
    *) echo "mlxsh needs Apple silicon; this is $(uname -m)" >&2; exit 1 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv (it supplies the python mlx needs)"
    $DRY sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    # the installer puts uv in ~/.local/bin
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

REPO="git+https://github.com/ansnadeem/mlxsh"

if [ -n "${MLXSH_REF:-}" ]; then
    TARGET="${REPO}@${MLXSH_REF}"
else
    TARGET="${PKG}${EXTRA}"
fi

echo "installing $TARGET"
if ! $DRY uv tool install --force "$TARGET"; then
    # not on PyPI yet, or the release is behind: take it from the repository
    echo "falling back to $REPO"
    $DRY uv tool install --force "$REPO"
fi

if [ -z "$DRY" ]; then
    echo
    echo "installed. Next:"
    echo "  mlxsh browse     pick a model"
    echo "  mlxsh lm         serve it at http://127.0.0.1:41277/v1"
    echo
    command -v mlxsh >/dev/null 2>&1 || cat <<'EOS'
mlxsh is not on your PATH yet. Add uv's tool directory:
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh
EOS
fi

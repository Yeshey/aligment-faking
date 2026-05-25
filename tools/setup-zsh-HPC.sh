#!/usr/bin/env bash
# Setup zsh on UniBo HPC (no root, no nix).
# Run from anywhere on the cluster.
# Re-running is safe — skips already-installed components.
set -euo pipefail

SCR="/scratch.hpc/joaofilipe.silvade"

# ── dirs ──────────────────────────────────────────────────────────────────────
mkdir -p "$SCR/bin" "$SCR/.config/zsh" "$SCR/.cache/zsh"

# ── 1. Static zsh (romkatv/zsh-bin) ──────────────────────────────────────────
# Installs to $SCR/bin/bin/zsh
if [[ ! -x "$SCR/bin/bin/zsh" ]]; then
    echo "→ Installing static zsh..."
    curl -fsSL https://raw.githubusercontent.com/romkatv/zsh-bin/master/install \
        | sh -s -- -d "$SCR/bin" -e no
else
    echo "✓ zsh already installed"
fi
export PATH="$SCR/bin/bin:$SCR/bin:$PATH"

# ── 2. oh-my-zsh ─────────────────────────────────────────────────────────────
if [[ ! -d "$SCR/.oh-my-zsh" ]]; then
    echo "→ Installing oh-my-zsh..."
    git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git "$SCR/.oh-my-zsh"
else
    echo "✓ oh-my-zsh already installed"
fi

# ── 3. Plugins ────────────────────────────────────────────────────────────────
ZSH_CUSTOM="$SCR/.oh-my-zsh/custom"

for plugin_repo in \
    "zsh-users/zsh-autosuggestions" \
    "zsh-users/zsh-syntax-highlighting" \
    "zsh-users/zsh-history-substring-search"
do
    plugin_name="${plugin_repo##*/}"
    if [[ ! -d "$ZSH_CUSTOM/plugins/$plugin_name" ]]; then
        echo "→ Installing $plugin_name..."
        git clone --depth=1 "https://github.com/$plugin_repo" \
            "$ZSH_CUSTOM/plugins/$plugin_name"
    else
        echo "✓ $plugin_name already installed"
    fi
done

# ── 4. .zshrc ─────────────────────────────────────────────────────────────────
cat > "$SCR/.config/zsh/.zshrc" << 'ZSHRC'
export PATH="/scratch.hpc/joaofilipe.silvade/bin/bin:/scratch.hpc/joaofilipe.silvade/bin:$PATH"

# Explicitly autoload zsh standard functions that static binaries
# sometimes miss — fixes "function definition file not found" warnings
autoload -Uz vcs_info bracketed-paste-magic add-zsh-hook

export ZSH="/scratch.hpc/joaofilipe.silvade/.oh-my-zsh"
export ZSH_CUSTOM="$ZSH/custom"
export ZSH_CACHE_DIR="/scratch.hpc/joaofilipe.silvade/.cache/zsh"

ZSH_THEME="agnoster"

# tmux omitted — not installed on this cluster
plugins=(
    git
    ssh-agent
    zsh-autosuggestions
    zsh-syntax-highlighting
    zsh-history-substring-search
)

source "$ZSH/oh-my-zsh.sh"

HISTFILE="/scratch.hpc/joaofilipe.silvade/.cache/zsh/.zsh_history"
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY HIST_IGNORE_DUPS HIST_IGNORE_SPACE

bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down

command -v direnv >/dev/null 2>&1 && eval "$(direnv hook zsh)"
ZSHRC

# ── 5. .zshenv ────────────────────────────────────────────────────────────────
cat > "$SCR/.config/zsh/.zshenv" << 'ZSHENV'
export PATH="/scratch.hpc/joaofilipe.silvade/bin/bin:/scratch.hpc/joaofilipe.silvade/bin:$PATH"
ZSHENV

# ── 6. nix-enter stub (login shell entry-point) ───────────────────────────────
printf '#!/usr/bin/env bash\nexport ZDOTDIR="%s/.config/zsh"\nexport PATH="%s/bin/bin:%s/bin:$PATH"\nexec "%s/bin/bin/zsh" -l\n' \
    "$SCR" "$SCR" "$SCR" "$SCR" > "$SCR/bin/nix-enter"
chmod +x "$SCR/bin/nix-enter"

# ── 7. bootstrap-home (re-run after cluster wipes real ~/) ────────────────────
printf '#!/usr/bin/env bash\n# Re-run after cluster wipes real ~/\nset -euo pipefail\nSCR="/scratch.hpc/joaofilipe.silvade"\nprintf '"'"'export ZDOTDIR="%%s/.config/zsh"\nexport PATH="%%s/bin/bin:%%s/bin:$PATH"\n'"'"' "$SCR" "$SCR" "$SCR" > "$HOME/.zshenv"\nprintf '"'"'if [[ -x "%%s/bin/nix-enter" ]]; then exec "%%s/bin/nix-enter"; fi\n'"'"' "$SCR" "$SCR" > "$HOME/.bashrc"\nprintf '"'"'Done. Re-login to get zsh.\n'"'"'\n' \
    > "$SCR/bin/bootstrap-home"
chmod +x "$SCR/bin/bootstrap-home"

# ── 8. Fix real home now ──────────────────────────────────────────────────────
printf 'export ZDOTDIR="%s/.config/zsh"\nexport PATH="%s/bin/bin:%s/bin:$PATH"\n' \
    "$SCR" "$SCR" "$SCR" > "$HOME/.zshenv"

printf 'if [[ -x "%s/bin/nix-enter" ]]; then exec "%s/bin/nix-enter"; fi\n' \
    "$SCR" "$SCR" > "$HOME/.bashrc"

echo ""
echo "✓ Done. Log out and back in to get zsh."
echo "  After any home wipe: bash $SCR/bin/bootstrap-home"
SCRIPT
echo "OK"
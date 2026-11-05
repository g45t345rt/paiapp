#!/usr/bin/env sh
# Download and install a supported Python (3.12-3.14) for Pai on Linux/macOS.
#
# Called by run.sh when no supported Python is on PATH and the user agreed to
# install one, but also safe to run directly. It tries the platform package
# manager first (Homebrew on macOS; apt/dnf/zypper/pacman on Linux) and, when
# that is unavailable or fails, falls back to the official python.org macOS
# installer. Linux has no official python.org binary, so there it only reports
# the package manager to use.
#
# Progress is written to stderr; on success the path to a working interpreter is
# printed to stdout (one line) and the exit code is 0. On failure the exit code
# is non-zero and nothing is printed to stdout.
set -eu

# The interpreter range Pai supports; keep in sync with setup_env.py
# (MIN_PYTHON/MAX_PYTHON), run.sh and the Windows installer.
FTP_BASE="https://www.python.org/ftp/python/"

log() {
    printf '==> %s\n' "$1" >&2
}

is_supported() {
    [ -n "$1" ] && command -v "$1" >/dev/null 2>&1 && \
        "$1" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)' >/dev/null 2>&1
}

# Print the first supported interpreter already on PATH, or nothing.
find_supported() {
    for candidate in python3.12 python3.13 python3.14 python3 python; do
        if is_supported "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# A fetch helper: prefer curl, fall back to wget.
have_curl=0
have_wget=0
command -v curl >/dev/null 2>&1 && have_curl=1
command -v wget >/dev/null 2>&1 && have_wget=1

fetch_exists() {
    # HEAD request: succeed only when the URL resolves to a real file.
    if [ "$have_curl" -eq 1 ]; then
        curl -fsIL "$1" >/dev/null 2>&1
    elif [ "$have_wget" -eq 1 ]; then
        wget --spider -q "$1" >/dev/null 2>&1
    else
        return 1
    fi
}

fetch_to() {
    # Download $1 to file $2.
    if [ "$have_curl" -eq 1 ]; then
        curl -fsSL -o "$2" "$1"
    else
        wget -q -O "$2" "$1"
    fi
}

# Echo the newest python.org macOS installer URL, or nothing.
find_macos_pkg_url() {
    [ "$have_curl" -eq 1 ] || [ "$have_wget" -eq 1 ] || return 1
    listing=$(
        if [ "$have_curl" -eq 1 ]; then
            curl -fsSL "$FTP_BASE"
        else
            wget -q -O - "$FTP_BASE"
        fi
    ) || return 1
    # Newest first, then keep only 3.12-3.14. Security-only releases ship no
    # binary, so probe each candidate until one has a .pkg.
    versions=$(printf '%s\n' "$listing" \
        | sed -n 's/.*href="\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)\/".*/\1/p' \
        | sort -t. -k1,1nr -k2,2nr -k3,3nr)
    for version in $versions; do
        case "$version" in
            3.12.*|3.13.*|3.14.*) ;;
            *) continue ;;
        esac
        url="${FTP_BASE}${version}/python-${version}-macos11.pkg"
        if fetch_exists "$url"; then
            printf '%s\n' "$url"
            return 0
        fi
    done
    return 1
}

install_macos() {
    if command -v brew >/dev/null 2>&1; then
        log "Installing python@3.12 with Homebrew"
        if brew install python@3.12; then
            for prefix in "$(brew --prefix 2>/dev/null)" \
                          "$(brew --prefix python@3.12 2>/dev/null)"; do
                for bin in "$prefix/bin/python3.12" \
                           "$prefix/libexec/bin/python"; do
                    if is_supported "$bin"; then
                        printf '%s\n' "$bin"
                        return 0
                    fi
                done
            done
            log "Homebrew installed python@3.12 but no usable interpreter was found"
        else
            log "Homebrew could not install python@3.12"
        fi
    fi

    url=$(find_macos_pkg_url || true)
    if [ -z "$url" ]; then
        log "Could not find a python.org macOS installer"
        return 1
    fi
    log "Downloading $url"
    pkg="${TMPDIR:-/tmp}/pai-python.$$.pkg"
    trap 'rm -f "$pkg"' EXIT
    fetch_to "$url" "$pkg" || { log "Download failed"; return 1; }
    log "Installing Python (this needs your administrator password)"
    if ! sudo installer -pkg "$pkg" -target /; then
        log "The python.org installer failed"
        return 1
    fi
    for version in 3.14 3.13 3.12; do
        bin="/Library/Frameworks/Python.framework/Versions/$version/bin/python$version"
        if is_supported "$bin"; then
            printf '%s\n' "$bin"
            return 0
        fi
    done
    return 1
}

install_linux() {
    log "Installing Python 3.12 with the system package manager"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && \
            sudo apt-get install -y python3.12 python3.12-venv python3-tk
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3.12 python3.12-tkinter
    elif command -v zypper >/dev/null 2>&1; then
        sudo zypper install -y python312 python312-tk
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm python
    else
        log "No supported package manager was found"
        return 1
    fi || {
        log "The package manager could not install Python 3.12"
        return 1
    }

    for version in 3.14 3.13 3.12; do
        bin="/usr/bin/python$version"
        if is_supported "$bin"; then
            printf '%s\n' "$bin"
            return 0
        fi
    done
    find_supported && return 0
    return 1
}

# Already satisfied? Print it and stop.
if existing=$(find_supported); then
    log "Python is already available at $existing"
    printf '%s\n' "$existing"
    exit 0
fi

os=$(uname -s)
result=""
case "$os" in
    Darwin) result=$(install_macos || true) ;;
    Linux) result=$(install_linux || true) ;;
    *)
        log "Unsupported platform: $os"
        exit 1
        ;;
esac

if [ -z "$result" ]; then
    log "Could not install Python 3.12-3.14 automatically."
    if [ "$os" = "Linux" ]; then
        log "Install python3.12 (and its venv/tk packages) with your package manager, then run Pai again."
    else
        log "Install Python 3.12-3.14 from https://www.python.org/downloads/ and run Pai again."
    fi
    exit 1
fi

log "Installed Python at $result"
printf '%s\n' "$result"

#!/bin/bash
# Logs Pi environment details to pi_environment.txt for reproducibility.
# Run this on the Pi: bash log_pi_environment.sh

OUT=./pi_environment.txt

{
    echo "=== date ==="
    date

    echo
    echo "=== Pi model ==="
    cat /proc/device-tree/model 2>/dev/null; echo

    echo
    echo "=== OS ==="
    cat /etc/os-release

    echo
    echo "=== kernel ==="
    uname -a

    echo
    echo "=== CPU ==="
    echo "cores: $(nproc)"
    grep -m1 'Model' /proc/cpuinfo

    echo
    echo "=== Python ==="
    python3 --version

    echo
    echo "=== torch CPU threads (default) ==="
    python3 -c "import torch; print(torch.get_num_threads())" 2>/dev/null || echo "torch not importable"

    echo
    echo "=== metavision-core version (if pip-installed) ==="
    pip show metavision-core 2>/dev/null | grep -E 'Name|Version' || echo "not found via pip -- check separately if installed via system package/installer"

    echo
    echo "=== full pip freeze ==="
    pip freeze

} > "$OUT"

echo "environment logged to $OUT"

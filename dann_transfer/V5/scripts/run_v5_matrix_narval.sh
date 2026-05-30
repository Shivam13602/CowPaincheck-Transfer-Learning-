#!/usr/bin/env bash
# V5 matrix on Alliance Narval (H100). Same protocol and H100 env as Rorqual; only paths/account differ in sbatch.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_v5_matrix_rorqual.sh"

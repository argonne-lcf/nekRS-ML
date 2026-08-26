#!/usr/bin/env bash
# Build the parRSB ctypes shim (libparrsb_shim.so) against a nekRS install.
#
# Usage:
#   NEKRS_HOME=/path/to/nekrs-install ./build_parrsb_shim.sh [outdir]
#
# Requires libparRSB.a and libgs.a from the nek5000 side of the install
# (built by the standard nekRS build; objects are -fPIC on Linux, PIC by
# default on macOS). The output goes next to this script by default so
# repartition/parrsb.py finds it without configuration.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE}
mkdir -p "$OUT"
: "${NEKRS_HOME:?set NEKRS_HOME to a nekRS install prefix}"

PARRSB_DIR="$NEKRS_HOME/nek5000/3rd_party/parRSB"
GSLIB_DIR="$NEKRS_HOME/nek5000/3rd_party/gslib"

for f in "$PARRSB_DIR/lib/libparRSB.a" "$GSLIB_DIR/lib/libgs.a" \
         "$PARRSB_DIR/include/parRSB.h" "$GSLIB_DIR/include/gslib.h"; do
  [[ -f $f ]] || { echo "missing $f (is NEKRS_HOME a full install?)" >&2; exit 1; }
done

CC=${MPICC:-mpicc}
case "$(uname)" in
  Darwin) SHARED=(-dynamiclib) ;;
  *)      SHARED=(-shared) ;;
esac

# -DMPI: parRSB.h's guard macro; gslib's config.h supplies the rest
# (GLOBAL_LONG_LONG, UNDERSCORE, PREFIX=gslib_, ...).
"$CC" -O2 -fPIC -DMPI \
  -I"$PARRSB_DIR/include" -I"$GSLIB_DIR/include" \
  "${SHARED[@]}" -o "$OUT/libparrsb_shim.so" \
  "$HERE/parrsb_shim.c" \
  "$PARRSB_DIR/lib/libparRSB.a" "$GSLIB_DIR/lib/libgs.a" -lm

echo "built $OUT/libparrsb_shim.so"

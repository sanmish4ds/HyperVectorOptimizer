#!/usr/bin/env bash
# Build libacorn.so: the ACORN reference implementation (static, symbols hidden) behind a 4-function
# C ABI, linked against the OpenMP runtime bundled with the pip faiss wheel so both coexist in one process.
# Prerequisite: ACORN cloned at $ACORN_DIR and built statically:
#   cmake -B build-static -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
#     -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
#     -DOpenMP_ROOT=$(brew --prefix libomp) -DCMAKE_CXX_FLAGS="-mcpu=native -fvisibility=hidden" . && make -C build-static faiss
# Apply acorn_fixes.patch to the ACORN checkout first: it wraps the nlohmann_json include in
# $<BUILD_INTERFACE:...> and fixes two heap overflows in ACORN.cpp (add_link at upper levels writes one
# neighbour past the node's slot; ep_per_level is sized max_level but indexed at max_level).
set -euo pipefail
cd "$(dirname "$0")"
A=${ACORN_DIR:-$HOME/hvo-ext/ACORN}
F=$(python3 -c "import faiss,os; print(os.path.join(os.path.dirname(faiss.__file__), '.dylibs'))")
O=$(brew --prefix libomp)
clang++ -O3 -std=c++17 -shared -fPIC -mcpu=native -fvisibility=hidden acorn_wrap.cpp -o libacorn.so \
  -I"$A" -I"$O/include" -Xclang -fopenmp "$A/build-static/faiss/libfaiss.a" -framework Accelerate \
  "$F/libomp.dylib" -Wl,-headerpad_max_install_names -Wl,-exported_symbols_list,acorn_exports.txt
old=$(otool -L libacorn.so | awk '/libomp/{print $1}')
install_name_tool -change "$old" "$F/libomp.dylib" libacorn.so
codesign -f -s - libacorn.so

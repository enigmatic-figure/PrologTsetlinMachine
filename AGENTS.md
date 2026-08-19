# AGENTS.md

## Commands
- `python -m pytest tests/python` - run the full test suite
- `ctest --test-dir out/build -C Release --output-on-failure` - run the full test suite
- `cmake -S . -B out/build -DPTM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release` - build the project
- `cmake --build out/build --config Release` - build the project

## Code Map
- src - application source
- tests - automated tests
- docs - project documentation
- .github - project configuration

## Conventions
- Use `#include "ptm/..."` for native imports.
- Require `.cpp`, `.hpp`, and `.cu` extensions for native source files.
- Name Python test files `test_*.py`.

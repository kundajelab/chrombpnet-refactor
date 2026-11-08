"""Run the optimization kit's `fast` mode on an L40S with its Triton TF32 forward
(run_sections34.sh's `fast_k1` arm; blog 2026-015 Section 3's L40S column).

The kit gives each (GPU, mode) its forward route from its own table,
torch/chrombpnet_k1/arch_tiles.json entries[arch].default_route, and has no
flag or variable to choose one. For an L40S (cuda-89) in `fast` (route mode
"prod") the table says "tf": stock's own TensorFlow forward, so that the files
stay bitwise equal to stock's default run. Its Triton TF32 kernels
(kernels_tc.py) are never run there, and the table lists no TF32 tile for
cuda-89, so they would use kernels_tc's own default tile.

This file, on PYTHONPATH with KIT_L40S_K1_FAST=1, changes exactly that answer:
after the kit's fastdefault.py is executed -- by a normal import in the job
process (chrombpnet_fastkit.fastdefault) or by path in the chrombpnet_opt
launcher (stack.fastdefault) -- its arch_default_route returns "k1" for
cuda-89 in "prod" mode, with a basis that says it was overridden. Everything
downstream is the kit's own code: resolve() then picks the K1 route, TF32
precision for prod, and no tc_tile (kernels_tc's default). Nothing else in the
kit is changed, no kit file is touched, and every other (arch, mode) pair
answers as the table does. The kit passes PYTHONPATH to its job process
(stack.kit_env strips only its own switches), so both processes see the hook.
"""

import os

if os.environ.get("KIT_L40S_K1_FAST") == "1":
    import importlib.machinery as _machinery

    _ARCH, _MODE = "cuda-89", "prod"
    _exec_module = _machinery.SourceFileLoader.exec_module

    def _patch(module):
        original = module.arch_default_route
        if getattr(original, "_kit_l40s_k1_fast", False):
            return

        def arch_default_route(kit_root, arch, mode=None):
            answer = original(kit_root, arch, mode)
            if arch == _ARCH and (mode or module.route_mode()) == _MODE:
                table = answer[1] if answer else "no table entry"
                return ("k1", "[%s/%s] OVERRIDDEN by run_sections34.sh's fast_k1 hook "
                        "(KIT_L40S_K1_FAST=1): the K1 Triton forward at the shipped TF32 numerics, "
                        "kernels_tc's default tile; the kit's own table: %s" % (_ARCH, _MODE, table),
                        "kit_hook_l40s_k1fast/sitecustomize.py over arch_tiles.json")
            return answer

        arch_default_route._kit_l40s_k1_fast = True
        module.arch_default_route = arch_default_route

    def exec_module(self, module):
        _exec_module(self, module)
        path = str(getattr(module, "__file__", "") or "").replace(os.sep, "/")
        if path.endswith("chrombpnet_fastkit/fastdefault.py"):
            _patch(module)

    _machinery.SourceFileLoader.exec_module = exec_module

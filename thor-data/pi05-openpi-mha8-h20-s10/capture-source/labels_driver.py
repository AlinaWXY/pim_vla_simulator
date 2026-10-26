"""Refine one semantic label; never change model execution or kernel order."""
import fine
import runner

original_category = fine.category


def category(name, shapes, scopes):
    if 'rope_generation' in scopes and name in ('aten::matmul', 'aten::mm', 'aten::bmm'):
        return 'rope_frequency_matmul'
    if 'rope_generation' in scopes and name in ('aten::sin', 'aten::cos'):
        return 'rope_' + name.split('::')[-1]
    return original_category(name, shapes, scopes)


fine.category = category
runner.main()

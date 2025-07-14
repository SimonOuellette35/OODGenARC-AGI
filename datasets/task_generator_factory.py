from datasets.generators.flip_generator import FlipGenerator
from datasets.generators.flip_recoloring_generator import FlipRecoloringGenerator
from datasets.generators.flip_shift_generator import FlipShiftGenerator
from datasets.generators.recoloring_generator import RecoloringGenerator
from datasets.generators.rotation_generator import RotationGenerator
from datasets.generators.shift_generator import ShiftGenerator
from datasets.generators.shift_recoloring_generator import ShiftRecoloringGenerator
from datasets.generators.counting_generator import CountingGenerator
from datasets.generators.cropping_generator import CroppingGenerator
from datasets.generators.drawing_generator import DrawingGenerator
from datasets.generators.shearing_generator import ShearingGenerator

class TaskGeneratorFactory:

    @staticmethod
    def create(task_name, DSL):
        if task_name == 'shifts':
            return ShiftGenerator(DSL)
        elif task_name == 'rotations':
            return RotationGenerator(DSL)
        elif task_name == 'flips':
            return FlipGenerator(DSL)
        elif task_name == 'recoloring':
            return RecoloringGenerator(DSL)
        elif task_name == 'flip+recoloring':
            return FlipRecoloringGenerator(DSL)
        elif task_name == 'shift+recoloring':
            return ShiftRecoloringGenerator(DSL)
        elif task_name == 'flip+shift':
            return FlipShiftGenerator(DSL)
        elif task_name == 'counting':
            return CountingGenerator(DSL)
        elif task_name == 'cropping':
            return CroppingGenerator(DSL)
        elif task_name == 'drawing':
            return DrawingGenerator(DSL)
        elif task_name == 'shearing':
            return ShearingGenerator(DSL)
        else:
            print(f"BUG: unknown task {task_name}!")
            exit(-1)
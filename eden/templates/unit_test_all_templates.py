import os, time, random, sys, shutil
import moviepy.editor as mpy
import traceback

def exception_to_string(excp):
   stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)  # add limit=?? 
   pretty = traceback.format_list(stack)
   return ''.join(pretty) + '\n  {} {}'.format(excp.__class__,excp)

sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs


class UnitTestTracker():
    def __init__(self):
        self.functions_tested = []
        self.test_status = []
        self.errors = []
        self.tracebacks = []

    def add(self, function_name, test_status, error, full_traceback):
        self.functions_tested.append(function_name)
        self.test_status.append(test_status)
        self.errors.append(error)
        self.tracebacks.append(full_traceback)

    def print(self):
        # print the results of the unit tests
        delimiter_str = "\n" * 5
        delimiter_str2 = "-" * 50
        print(delimiter_str)
        print(delimiter_str2)
        for i, function_name in enumerate(self.functions_tested):
            print(delimiter_str2)
            padding_spaces = " " * (30 - len(function_name))
            print(f"{function_name}: {padding_spaces}{self.test_status[i]}")
            if len(self.errors[i]) > 0:
                print(f"----- Error:\n{self.errors[i]}")
            if len(self.tracebacks[i]) > 0:
                print(f"----- Full traceback:\n{self.tracebacks[i]}")
        print(delimiter_str2)
        print(delimiter_str2)
        print(delimiter_str)


def test_function(fname, function, args, kwargs, test_tracker):
    try:
        function(*args, **kwargs)
        status, error, full_traceback = "success", "", ""
    except Exception as e:
        status, error, full_traceback = "failed", str(e), exception_to_string(e)

    test_tracker.add(fname, status, error, full_traceback)
    test_tracker.print()
    return



if __name__ == "__main__":

    functions_to_test = ["real2real", "lerp", "generate_basic", "generate_remix"]
    debug  = 0  # optionally overwrite the default render args to make things go FAAAST

    # template inputs:
    outdir = "unit_test_results"
    seed   = 0
    n      = 3  # n_prompts to sample for lerp
    init_image_data = "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"
    input_images = [
        "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp",
        "https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp",
        "https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp"]

    test_tracker = UnitTestTracker()

    if "generate_basic" in functions_to_test:
        from generate_basic import generate_basic
        random.seed(seed)
        args   = (random.choice(text_inputs), outdir)
        kwargs = {"seed": seed, "debug": debug}
        test_function("generate_basic", generate_basic, args, kwargs, test_tracker)

    if "generate_remix" in functions_to_test:
        from generate_remix import remix
        args   = (init_image_data, outdir)
        kwargs = {"seed": seed, "debug": debug}
        test_function("remix", remix, args, kwargs, test_tracker)

    if "real2real" in functions_to_test:
        from interpolate_real2real import real2real
        args   = (input_images, outdir)
        kwargs = {"seed": seed, "debug": debug}
        test_function("real2real", real2real, args, kwargs, test_tracker)

    if "lerp" in functions_to_test:
        from interpolate_basic import lerp
        random.seed(seed)
        args   = (random.sample(text_inputs, n), outdir)
        kwargs = {"seed": seed, "debug": debug}
        test_function("lerp", lerp, args, kwargs, test_tracker)

    print("All template tests completed!")
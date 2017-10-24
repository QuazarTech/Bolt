import arrayfire as af

def timer(self, timing_variable, func, *args):

    def wrapper():

        tic = af.time()
        func(self, *args)
        af.sync()
        toc = af.time()
        timing_variable += toc - tic

    return wrapper

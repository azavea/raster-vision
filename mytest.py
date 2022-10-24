
class MyContextManager():
    def __enter__(self):
        print('enter')

    def __exit__(self, exc_type, exc_value, exc_tb):
        print('exit')

def get_generator():
    with MyContextManager():
        for i in range(10):
            yield i

def foo():
    print('start of foo')
    gen = get_generator()
    next(gen)
    gen = None
    print('end of foo')

print('before foo')
foo()
print('after foo')

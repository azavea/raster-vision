# context manager's exit method is called when the reference to
# the generator is lost, either by setting it to None or moving out of scope

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
    x = next(gen)
    print('end of foo')
    return x

print('before foo')
y = foo()
print('after foo')
y += 1
print('after incrementing y')

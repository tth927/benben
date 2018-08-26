class MyClass:
    """A simple example class"""
    i = 12345

    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def f(self):
        return 'hello world'


# x = MyClass()
# print(x.f())
# print(x.i)

x = MyClass(4,5)
print(x.r)
print(x.i)

x.counter = 1
while x.counter < 10:
    x.counter = x.counter * 2
print(x.counter)
del x.counter



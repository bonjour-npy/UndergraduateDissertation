class Test:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bark(self, name):
        print(name)


test = Test(1, 2)
test.bark('hello')

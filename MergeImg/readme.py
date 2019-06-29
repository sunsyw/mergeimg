class A:
    def __init__(self):
        self.a = 1

    def work(self):
        self.b = 2
        print('1')


a = A()
a.work()  # 调用word之后才有b
print(a.b)

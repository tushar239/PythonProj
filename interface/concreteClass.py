from interface_practice import demoInterface


class concreteclass(demoInterface):
    def method1(self):
        print("This is method1")
        

    def method2(self):
        print("This is method2")



obj = concreteclass()
obj.method1()
obj.method2()
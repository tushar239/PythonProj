from interface_practice import DemoInterface


class ConcreteClass(DemoInterface):
    def method1(self):
        super().method1()
        print("This is method1")
        return

    def method2(self):
        super().method2()
        print("This is method2")
        return


obj = ConcreteClass()
obj.method1()
obj.method2()

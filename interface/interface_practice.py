# https://www.tutorialspoint.com/python/python_interfaces.htm

from abc import ABC, abstractmethod
class demoInterface(ABC):
   @abstractmethod
   def method1(self):
      print ("Abstract method1")


   @abstractmethod
   def method2(self):
      print ("Abstract method1")


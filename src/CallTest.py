# @Time:2023/11/6 20:42 
# @Author:andrew
# @email:zengjunjine1026@163.com
# @File:CallTest.py 
# @veision 1.0
class Person:
    def __call__(self, name):
        print("__call__"+"Hello"+name)
    def hello(self,name):
        print("hello"+name)


person=Person()
person("zhnagsan")
person.hello('lisa')
person()
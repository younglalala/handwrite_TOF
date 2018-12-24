import logging

# def use_logging(func):
#
#     def wrapper():
#         logging.warn('{} is running'.format(func.__name__))
#
#         return func()
#
#     return wrapper
#
# @use_logging
# def foo():
#     print('i am foo')
#
# def foo1():
#     print('i am foo1')


def logged(func):
    def with_logging(*args, **kwargs):
        print(func.__name__)      # 输出 'with_logging'
        print(func.__doc__)       # 输出 None
        return func(*args, **kwargs)
    return with_logging

# 函数
@logged
def f(x):
   """does some math"""
   return x + x * x

a=f(4)
print(a)
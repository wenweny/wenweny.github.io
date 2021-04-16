

# python中的多线程

[TOC]

多线程是操作系统能够进行运算调度的最小单位，它被包含在进程之中，是进程中的实际运作单位。一个进程必须包含一个线程。

在线程中，所有状态在默认情况下都是共享的，比如内存共享。

## 先来试试

python标准库包括：低级模块_thread 和高级模块 threading，绝大多数情况我们使用threading.

```python
from threading import Thread

# Thread 基本参数
class ExplainThread:
    def __init__(self, group=None, target=None, name=None,
                     args=(), kwargs=None, *, daemon=None):
        """
        target： 指定线程由 run () 方法调用的可调用对象。默认为 None, 意味着不调用任何内容。
        name： 指定该线程的名称。 在默认情况下，创建一个唯一的名称。
        args： target调用的实参，元组格式。默认为 ()，即不传参。
        daemon： 为False表示父线程在运行结束时需要等待子线程结束才能结束程序，为True则表示父线程在运行结束时，子线程无论是否还有任务未完成都会跟随父进程退出，结束程序。
        """
        pass
```



```python
def worker(arg):  # 线程执行的目标函数
    print("I'm working {}".format(arg))
    print("worker thread:",current_thread())
    print("Fineshed")


print(current_thread())
t = Thread(target=worker, args=(current_thread(),), name="firstworker")  # 线程对象
t.start()  # 启动线程

"""
<_MainThread(MainThread, started 43488)>
I'm working <_MainThread(MainThread, started 43488)>
worker thread: <Thread(firstworker, started 61856)>
Fineshed
"""
```

最简单的多线程，哪怕只做一件事，也是新开了一个线程。

线程的传参和普通函数没有区别，只是格式必须为元祖格式。

当函数执行完之后，线程也就跟着退出了。

## 线程退出条件

- 线程内的函数语句执行完毕，线程自动结束。（如上面的例子）
- 线程内的函数抛出未处理的异常。

```python
import threading
import time

def worker(arg):
    count = 0
    while True:
        if count > 3:
            raise RuntimeError(count)
        time.sleep(1)
        count += 1
        print("I'm working {},count={}".format(arg, count))
    print("Fineshed")

t = threading.Thread(target=worker, args=(threading.enumerate(),), name="SonThread")
t.start()

print(threading.enumerate())
print("====end===")

"""
[<_MainThread(MainThread, started 63924)>, <Thread(SonThread, started 34916)>]
====end===
[<_MainThread(MainThread, started 63924)>, <Thread(SonThread, started 34916)>]
I'm working [<_MainThread(MainThread, stopped 63924)>],count=1
I'm working [<_MainThread(MainThread, stopped 63924)>],count=2
I'm working [<_MainThread(MainThread, stopped 63924)>],count=3
I'm working [<_MainThread(MainThread, stopped 63924)>],count=4
Exception in thread SonThread:
Traceback (most recent call last):
  File "D:\Anaconda3\lib\threading.py", line 917, in _bootstrap_inner
    self.run()
  File "D:\Anaconda3\lib\threading.py", line 865, in run
    self._target(*self._args, **self._kwargs)
  File "E:/yangwenwen/pythonProject/multi_threading.py", line 35, in worker
    raise RuntimeError(count)
RuntimeError: 4
"""
```

上面例子中，演示了触发异常自动退出线程。但最先打印的是主程序的"===end==="语句，是因为在程序中，主线程启动一个线程后，不会等待子线程执行完毕，就继续执行了后续语句，在执行完主线程语句后，发现还有子线程没有结束，于是等待子线程执行结束，子线程在运行时抛出了未处理的异常，最终子线程结束，主线程也随之结束。



## 方法属性

### Threading属性

- `threading.current_thread()`   返回当前线程对象
- `threading.main_thread()`         返回主线程对象
- `threading.active_count()`       返回处于Active状态的线程个数
- `threading.enumerate()`             返回所有存活的线程的列表，不包括已经终止的线程和未启动的线程
- `threading.get_ident()`             返回当前线程的ID，非0整数

### Thread实例的属性

- `threading.current_thread().name`        线程名，只是一个标识符，可以使用`getName()`、`setName()`获取和运行时重命名。
- `threading.current_thread().ident`       线程ID，非0整数。线程启动后才会有ID，否则为None。线程退出，此ID依旧可以访问。此ID可以重复使用
- `threading.current_thread().is_alive()`  返回线程是否存活，布尔值，True或False。

```python
import threading
import time


def showthreadinfo():
    print("~~~~~~~showthreadinfo~~~~~~")
    print("current thread = {}".format(threading.current_thread()))
    print("main thread  = {}".format(threading.main_thread()))
    print("active thread count = {}".format(threading.active_count()))
    print("active thread list = {}".format(threading.enumerate()))
    print("thread id = {}".format(threading.get_ident()))
    print("~~~~~~~showthreadinfo end~~~~~~")


def worker(arg):
    count = 0
    while True:
        if count > 2:
            raise RuntimeError(count)
        time.sleep(1)
        count += 1
        showthreadinfo()
        print("I'm working {},count={}".format(arg, count))
    print("Fineshed")


showthreadinfo()  # 主线程中调用
time.sleep(1)

t = threading.Thread(target=worker, args=(current_thread(),),name="SonThread")
t.start()

while True:
    time.sleep(1.1)
    if t.is_alive():
        print("{} {} alive".format(t.name, t.ident))
        print("Thread name:", t.getName())
        print("Thread setName")
        t.setName("SonThreadRename")
        print("Thread name:", t.getName())
    else:
        print("{} {} alive".format(t.name, t.ident))
        t.start()

print("====end===")


"""
~~~~~~~showthreadinfo~~~~~~
current thread = <_MainThread(MainThread, started 45476)>
main thread  = <_MainThread(MainThread, started 45476)>
active thread count = 1
active thread list = [<_MainThread(MainThread, started 45476)>]
thread id = 45476
~~~~~~~showthreadinfo end~~~~~~
~~~~~~~showthreadinfo~~~~~~
current thread = <Thread(SonThread, started 30216)>
main thread  = <_MainThread(MainThread, started 45476)>
active thread count = 2
active thread list = [<_MainThread(MainThread, started 45476)>, <Thread(SonThread, started 30216)>]
thread id = 30216
~~~~~~~showthreadinfo end~~~~~~
I'm working <_MainThread(MainThread, started 45476)>,count=1
SonThread 30216 alive
Thread name SonThread
Thread rename
Thread name SonThreadRename
~~~~~~~showthreadinfo~~~~~~
current thread = <Thread(SonThreadRename, started 30216)>
main thread  = <_MainThread(MainThread, started 45476)>
active thread count = 2
active thread list = [<_MainThread(MainThread, started 45476)>, <Thread(SonThreadRename, started 30216)>]
thread id = 30216
~~~~~~~showthreadinfo end~~~~~~
I'm working <_MainThread(MainThread, started 45476)>,count=2
SonThreadRename 30216 alive
Thread name SonThreadRename
Thread rename
Thread name SonThreadRename
~~~~~~~showthreadinfo~~~~~~
current thread = <Thread(SonThreadRename, started 30216)>
main thread  = <_MainThread(MainThread, started 45476)>
active thread count = 2
active thread list = [<_MainThread(MainThread, started 45476)>, <Thread(SonThreadRename, started 30216)>]
thread id = 30216
~~~~~~~showthreadinfo end~~~~~~
I'm working <_MainThread(MainThread, started 45476)>,count=3
Exception in thread SonThreadRename:
Traceback (most recent call last):
  File "D:\Anaconda3\lib\threading.py", line 917, in _bootstrap_inner
    self.run()
  File "D:\Anaconda3\lib\threading.py", line 865, in run
    self._target(*self._args, **self._kwargs)
  File "E:/yangwenwen/pythonProject/multi_threading.py", line 45, in worker
    raise RuntimeError(count)
RuntimeError: 3

SonThreadRename 30216 alive

-------------------------------------------------------------------------------
multi_threading.py 69 <module>
t.start()

threading.py 843 start
raise RuntimeError("threads can only be started once")

RuntimeError:
threads can only be started once
"""
```

线程退出后，尝试再次启动线程时，抛出`RuntimeError`异常，线程对象在定义后只能启动一次。

### run和start方法

#### 单线程情况

```python
import threading

class InputReader(threading.Thread):
    def run(self):
        print("~~~ run ~~~")
        super(InputReader, self).run()
        self.line_of_text = input()
    def start(self):
        print("~~~ start ~~~")
        super().start() #调用父类的start()和run()方法

print("Enter some text and press enter")
t1 = InputReader()

# start以并发模式运行
# t1.start()

# run 只是普通的函数调用
t1.run()

count = result = 1
while t1.is_alive():
    result = count*count
    count+=1

print("{0} *{0} = {1}".format(count,result))
print("while you typed '{}'".format(t1.line_of_text))

"""
以run运行
Enter some text and press enter
~~~ run ~~~
test run
1 *1 = 1
while you typed 'test run'
-------------------------------------
以start运行
Enter some text and press enter
~~~ start ~~~
~~~ run ~~~
test start
4674427 *4674427 = 21850258429476
while you typed 'test start'
"""
```

解释：

- start(): 程序启动，运行到start时，调用start()方法启动了一个新的线程，而初始线程继续进行，判断线程存活执行了while操作，当用户键盘输入完成后，start开启的新线程结束，while亦结束。一旦退出，输出总结信息。

- run(): 程序启动，运行到run时，调用InputReader类的run函数，不开启新的线程，当用户键盘输入完成后，run方法结束，所以再往下执行的时候，while不生效。

- 另一点，可以看出start()方法会先运行start()方法，再运行run()方法；而运行线程的run()方法只能调用到run()方法。

- start() --> run() --> _target()

  run() --> _target()

#### 多线程情况

```python
import threading
import time

def worker():
    count = 1
    while True:
        if count >3:
            break
        time.sleep(1)
        count += 1
        print("thread name = {}, thread id = {}, count = {} ".format(threading.current_thread().name,
                                                        threading.current_thread().ident,count))

t1 = threading.Thread(target=worker, name="t1")
t2 = threading.Thread(target=worker, name='t2')

# t1.start()
# # t1.join()
# t2.start()

t1.run()
t2.run()
print("===end===")

"""
run运行
-----------------
thread name = MainThread, thread id = 75620, count = 2
thread name = MainThread, thread id = 75620, count = 3
thread name = MainThread, thread id = 75620, count = 4
thread name = MainThread, thread id = 75620, count = 2
thread name = MainThread, thread id = 75620, count = 3
thread name = MainThread, thread id = 75620, count = 4
===end===
-------------------
stat运行
-------------------
===end===
thread name = t2, thread id = 58296, count = 2 thread name = t1, thread id = 93012, count = 2 

thread name = t2, thread id = 58296, count = 3 thread name = t1, thread id = 93012, count = 3 

thread name = t1, thread id = 93012, count = 4 thread name = t2, thread id = 58296, count = 4 
-------------------
start+join
-------------------
thread name = t1, thread id = 86160, count = 2 
thread name = t1, thread id = 86160, count = 3 
thread name = t1, thread id = 86160, count = 4 
===end===
thread name = t2, thread id = 66352, count = 2 
thread name = t2, thread id = 66352, count = 3 
thread name = t2, thread id = 66352, count = 4 
"""
```

多线程情况下

- 当两个子线程都用run()方法启动时，会先运行t1.run()，运行完之后才按顺序运行t2.run()，两个线程都工作在主线程，没有启动新线程，因此，run()方法仅仅是普通函数调用。
- 当两个子线程都用start()方法启动时，start()方法启动了两个新的子线程并交替运行，线程名是我们定义的name，每个子进程ID也不同。
- join()方法，正常场景，主线程在起了一个新的子线程后，主线程和子线程是并行的，互不干扰。但是，假如主线程调用了join方法，那它就得等待子线程完全执行完毕才能执行join()之后的语句，相当于又变成了单线程按顺序执行。



## 全局解释器锁GIL(Global Interpreter Lock)

可能究其根本很复杂，但简单理解来说

- GIL 是CPython引入的概念，是设计之初为解决多线程之间数据完整性和状态同步，用了最简单自然的方法-加锁。后来由于代码开发者接受了这种设定，重度依赖这种特性而难以去除。
- CPython中
  - IO密集型，某个线程阻塞，就会调度其他就绪线程；
  - CPU密集型，当前线程可能会连续的获得GIL，导致其它线程几乎无法使用CPU。
- 在CPython中由于有GIL存在
  - IO密集型，多线程
  - CPU密集型，多进程
- 多线程编程，模型复杂，容易发生冲突，必须用锁加以隔离，同时，又要小心死锁的发生。
- Python解释器由于设计时有GIL全局锁，导致了多线程无法利用多核。多线程的并发在Python中就是一个美丽的梦。
- 由于多线程共享内存，线程并发执行的顺序又是随机无法控制的，可能会造成多个线程同时改一个变量，把内容给改乱了。可通过`threading.Lock()`锁保证某段关键代码只能由一个线程从头到尾完整地执行，但可能会造成死锁，[廖雪峰多线程](https://www.liaoxuefeng.com/wiki/1016959663602400/1017629247922688)中有更详细的介绍。

## 参考 

[Python 多线程 使用线程 (二)](https://www.cnblogs.com/i-honey/p/7823587.html)

[Python 多线程 start()和run()方法的区别(三)](https://www.cnblogs.com/i-honey/p/8043648.html)

（这一系列都蛮清晰的）

[廖雪峰多线程](https://www.liaoxuefeng.com/wiki/1016959663602400/1017629247922688)
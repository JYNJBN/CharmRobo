# -*- coding: utf-8 -*-
"""
Python 基础语法 & 常用方法 —— 可运行速查脚本
面向:前端(JS/TS) + Java 背景转 Python
用法: python python_syntax_cheatsheet.py [模块名]
     不带参数则全部运行一遍
"""

from __future__ import annotations  # 让注解可以写 list[str] 而不是 List[str](3.9 之前兼容)

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional


def h(title: str) -> None:
    """打印分节标题,纯装饰"""
    print(f"\n{'=' * 56}\n  {title}\n{'=' * 56}")


# ---------------------------------------------------------------- 1. 数据类型
def demo_types():
    h("1. 数据类型 & 变量")

    # ---- 基本类型(注意:Python 没有 const/let,一切都是直接赋值)
    name = "ARCS"  # str    ← JS 的 string
    count = 42  # int    ← JS 的 number(Python 里 int 任意精度,不会溢出)
    price = 19.99  # float  ← JS 的 number
    is_online = True  # bool   ← 注意首字母大写!Java 的 true → Python 的 True
    nothing = None  # None   ← JS 的 null / Java 的 null(单例)

    print(type(name), type(count), type(is_online), type(nothing))

    # ---- 多变量赋值 + 解包(JS 的解构赋值)
    a, b = 1, 2  # ≈ JS: const [a, b] = [1, 2]
    a, b = b, a  # 交换,不需要临时变量(Java 里做不到)
    print(f"a={a} b={b}")

    # ---- 类型转换
    print(int("100") + 1, str(3.14), float("2.5"), bool(""), bool("0"))
    # 坑:bool("0") == True —— Python 里只有空字符串/0/None/空容器才是 False

    # ---- 真值判断(Python 的 "falsy" 集合,和 JS 很像但不完全一样)
    falsy = [False, 0, 0.0, "", None, [], {}, set()]
    print("falsy 数量:", sum(1 for x in falsy if not x))

    # ---- 多赋值解包进阶:* 收集剩余(JS 的 rest)
    first, *rest, last = [1, 2, 3, 4, 5]
    print(f"first={first} rest={rest} last={last}")  # rest 永远是 list

    # ---- 类型注解(只是提示,运行时不校验!≈ TS 的类型但不会报错)
    device_sn: str = "ARCS-A1-0001"
    # device_sn = 123        # IDE 会画波浪线,但运行照样能过


# ---------------------------------------------------------------- 2. 字符串
def demo_string():
    h("2. str 常用方法(最高频)")

    s = "  Hello, ARCS-Mini  "

    # f-string —— 最推荐的格式化方式(JS 的模板字符串)
    sn = "A1-0001"
    print(f"设备 {sn} 上线,时间 {2026}")  # ≈ JS: `设备 ${sn} 上线`
    print(f"{3.14159:.2f}")  # :.2f = 保留 2 位小数
    print(f"{sn:>15}")  # :>15 = 右对齐宽 15
    print(f"{sn!r}")  # !r = 显示 repr(带引号)

    # 拼接 / 分割
    print("-".join(["a", "b", "c"]))  # "a-b-c"  ← JS: arr.join("-")
    print("a,b,c".split(","))  # ['a','b','c'] ← JS: str.split(",")
    print("a b\nc".splitlines())  # 按行切,比 split("\n") 稳

    # 清洗
    print(repr(s.strip()))  # 去首尾空白 ← JS trim()
    print(repr(s.lstrip()), repr(s.rstrip()))  # trimStart / trimEnd

    # 查找判断
    print("ARCS" in s)  # 包含判断,JS: s.includes("ARCS")
    print(s.find("ARCS"))  # 找不到返回 -1(JS indexOf 同)
    print(s.index("ARCS"))  # 找不到直接抛 ValueError!
    print(s.startswith("  He"), s.endswith("Mini  "))

    # 替换 / 大小写
    print(s.replace("Mini", "Pro"))  # JS replace 只换第一个,Python 默认换全部
    print(s.upper(), s.lower(), s.title())

    # 判断类(Java 里要写正则,Python 直接有)
    print("abc123".isalnum(), "123".isdigit(), "abc".isalpha(), "  ".isspace())

    # 切片 —— Python 的灵魂,str/list/tuple 通用
    t = "HelloWorld"
    print(t[0:5])  # 'Hello'  左闭右开 [start:end)
    print(t[5:])  # 'World'
    print(t[:5])  # 'Hello'
    print(t[::2])  # 'Hlool'  步长 2
    print(t[::-1])  # 'dlroWolleH'  ← 反转字符串,Python 最经典写法
    print(t[-1], t[-3:])  # 负索引:从尾部数


# ---------------------------------------------------------------- 3. 列表
def demo_list():
    h("3. list 常用方法(= JS 的 Array)")

    nums = [3, 1, 2]
    nums.append(4)  # push
    nums.insert(0, 0)  # unshift(插到指定位置)
    nums.extend([5, 6])  # ≈ push(...arr) / concat
    print(nums)

    print(nums.pop())  # pop() 删并返回最后一个
    print(nums.pop(0))  # shift() 删并返回第一个
    nums.remove(3)  # 按"值"删第一个匹配项(JS 没有对应,得用 splice+indexOf)
    print(nums)

    print(nums.index(2))  # indexOf
    print(nums.count(2))  # 出现次数(JS 没有)
    print(2 in nums)  # includes

    nums.sort()  # 原地排序(改自己)
    nums.sort(key=lambda x: -x)  # 按规则排,key ≈ JS sort((a,b)=>...) 的映射版
    print(nums)
    print(sorted(nums, reverse=True))  # 返回新列表,不改原列表
    nums.reverse()  # 原地反转
    print(nums)

    # 切片 = JS 的 slice(但功能更强)
    print(nums[1:3], nums[:2], nums[::-1])

    # 清空 / 复制(深浅拷贝坑!)
    a = [1, [2, 3]]
    b = a.copy()  # 浅拷贝:内层 list 还是共享的!
    import copy
    c = copy.deepcopy(a)  # 深拷贝
    a[1].append(9)
    print(f"浅拷贝也被改了: {b}  深拷贝没事: {c}")

    # 枚举 / 同时遍历(Java/JS 都要写 index 循环,Python 直接来)
    for i, v in enumerate(["a", "b", "c"]):
        print(f"  [{i}] = {v}")
    for x, y in zip([1, 2], ["a", "b"]):  # 拉链合并
        print(f"  {x}->{y}")


# ---------------------------------------------------------------- 4. 字典
def demo_dict():
    h("4. dict 常用方法(= JS 的 Object/Map)")

    d = {"sn": "A1-0001", "online": True}

    # 取值:[] 找不到会 KeyError,get() 找不到返回 None 或默认值
    print(d["sn"])
    print(d.get("not_exist"))  # None
    print(d.get("not_exist", "默认值"))  # JS 里没有,Java Map.getOrDefault 同款
    print(d.setdefault("model", "doubao"))  # 没有就塞进去并返回,有就直接返回

    # 增删改
    d["owner"] = "wugf"
    d.update({"model": "deepseek", "ver": 2})  # ≈ JS Object.assign / {...a, ...b}
    print(d)
    removed = d.pop("ver")  # 删并返回
    d.pop("nothing", None)  # 安全的删,不存在不报错
    last = d.popitem()  # 删并返回最后一对 (k, v)
    print(f"removed={removed} last={last} -> {d}")

    # 遍历(三种,用得最多的是 items())
    for k in d:  # 只拿 key
        pass
    for v in d.values():  # 只拿 value
        pass
    for k, v in d.items():  # ★ 最常用 key + value
        print(f"  {k} = {v}")

    # 判断 / 长度
    print("sn" in d, len(d))

    # 构造技巧
    keys = ["a", "b"]
    print(dict.fromkeys(keys, 0))  # {'a': 0, 'b': 0} 初始化计数器的常用写法

    # 合并(3.9+)
    x, y = {"a": 1}, {"b": 2}
    print(x | y)  # 合并成新 dict


# ---------------------------------------------------------------- 5. 集合 & 元组
def demo_set_tuple():
    h("5. set & tuple")

    # set = 无序、自动去重(JS 的 Set)
    s = {1, 2, 2, 3}
    print(s)  # {1, 2, 3}
    s.add(4)
    s.discard(99)  # 删,不存在也不报错
    # s.remove(99)                  # 删,不存在会 KeyError
    print(s | {9}, s & {1, 2}, s - {1})  # 并集 / 交集 / 差集

    # 去重最快写法(会丢顺序,要保序用 dict.fromkeys)
    print(list(set([1, 1, 2, 2, 3])))

    # tuple = 不可变 list(Java 没有对应;≈ TS 的 readonly [number, string])
    point: tuple[int, int] = (10, 20)
    x, y = point  # 解包
    print(x, y)


# ---------------------------------------------------------------- 6. 推导式
def demo_comprehension():
    h("6. 推导式 —— Python 最优雅的语法,必会")

    nums = [1, 2, 3, 4, 5, 6]

    # list 推导式 ≈ JS 的 map + filter 合一
    evens = [n for n in nums if n % 2 == 0]
    print("偶数:", evens)

    # 带转换
    print("平方:", [n ** 2 for n in nums])  # ** 是幂运算,不是异或!

    # 三元表达式写在里面
    print("标签:", ["大" if n > 3 else "小" for n in nums])

    # 嵌套循环
    print("组合:", [(a, b) for a in "AB" for b in "12"])

    # dict 推导式
    squares = {n: n ** 2 for n in nums}
    print("dict:", squares)
    # 反转 dict
    print("反转:", {v: k for k, v in squares.items()})

    # set 推导式
    print("set:", {n % 3 for n in nums})

    # 生成器表达式:把 [] 换成 (),不占内存(大数据/读文件必用)
    gen = (n ** 2 for n in nums)
    print("生成器:", gen, "->", sum(gen))


# ---------------------------------------------------------------- 7. 函数
def demo_function():
    h("7. 函数")

    # 默认参数 + 关键字参数
    def greet(name: str, prefix: str = "Hello") -> str:
        return f"{prefix}, {name}"

    print(greet("ARCS"))
    print(greet("ARCS", prefix="Hi"))  # 关键字调用(Java 没有)
    print(greet(name="ARCS", prefix="Yo"))  # 顺序都能换

    # ★ 大坑:默认参数别用可变对象!
    def bad(items=[]):  # 这个 [] 只创建一次,所有调用共享!
        items.append(1)
        return items

    def good(items=None):
        items = items if items is not None else []
        items.append(1)
        return items

    r1, r2 = bad(), bad()
    print(f"bad: 第一次={r1} 第二次={r2} 是同一个对象? {r1 is r2}  ← 被污染了")
    g1, g2 = good(), good()
    print(f"good: 第一次={g1} 第二次={g2} 是同一个对象? {g1 is g2}")

    # *args / **kwargs(≈ JS 的 ...rest,但分位置参数和关键字参数两种)
    def log(*args, **kwargs):
        print(f"  位置参数(元组): {args}")
        print(f"  关键字参数(字典): {kwargs}")

    log(1, 2, level="ERROR", tag="arcs")

    def add(a, b):
        return a + b

    print(add(*[1, 2]))  # * 拆包 list 当位置参数传
    print(add(**{"a": 1, "b": 2}))  # ** 拆包 dict 当关键字参数传

    # lambda:只能写一行表达式(JS 的箭头函数,但功能弱很多)
    f = lambda x: x * 2
    print(f(21))
    # 实际项目最常见的用法:当 key / 排序规则
    print(sorted([{"n": 3}, {"n": 1}], key=lambda d: d["n"]))

    # 返回多个值 —— 本质是返回 tuple,外面自动解包
    def minmax(xs):
        return min(xs), max(xs)

    lo, hi = minmax([3, 1, 9])
    print(f"min={lo} max={hi}")


# ---------------------------------------------------------------- 8. 类
@dataclass
class Device:
    """dataclass ≈ TS 的 interface + 自动生成构造器/toString/equals"""
    sn: str
    model: str = "doubao"
    online: bool = False
    tags: list[str] = field(default_factory=list)  # ★ 可变默认值必须这么写

    def label(self) -> str:
        return f"{self.sn}({self.model})"


class DeviceRepo:
    """普通类:注意每个方法第一个参数都是 self(≈ Java 的 this,但要显式写出来)"""

    def __init__(self, name: str):  # 构造器
        self.name = name  # 实例属性(可以随时新增,不用先声明)
        self._cache: dict[str, Device] = {}  # 单下划线 = 约定私有(不强制)

    def add(self, d: Device) -> None:
        self._cache[d.sn] = d

    def get(self, sn: str) -> Optional[Device]:  # Optional[X] = X | None
        return self._cache.get(sn)

    @property  # getter,调用时不用加括号
    def count(self) -> int:
        return len(self._cache)

    @staticmethod  # 不依赖实例(Java static)
    def version() -> str:
        return "1.0"

    @classmethod  # 拿到的是类本身,不是实例(工厂方法常用)
    def empty(cls):
        return cls("empty")

    def __len__(self):  # 魔法方法:让 len(repo) 能用
        return len(self._cache)

    def __repr__(self):  # ≈ Java 的 toString
        return f"<DeviceRepo {self.name} n={len(self._cache)}>"


def demo_class():
    h("8. 类 / dataclass")

    d1 = Device(sn="A1-0001")
    d2 = Device(sn="A1-0001")
    print(d1)  # dataclass 自动生成 __repr__
    print(d1 == d2)  # dataclass 自动生成 __eq__(原生 class 比的是内存地址!)

    repo = DeviceRepo("main")
    repo.add(d1)
    found = repo.get("A1-0001")
    # ★ 注意:Python 没有 JS 的 ?. 可选链!只能用三元表达式
    print(repo, repo.count, len(repo), found.label() if found else "没找到")
    print(DeviceRepo.version(), DeviceRepo.empty())

    # 继承
    class Sensor(Device):
        pass

    print(issubclass(Sensor, Device), isinstance(d1, Device))


# ---------------------------------------------------------------- 9. 异常 & 文件 & with
def demo_error_file():
    h("9. 异常处理 / 文件 / with")

    # try-except-else-finally(≈ Java 的 try-catch-finally)
    try:
        n = int("abc")
    except ValueError as e:  # 捕获指定异常(推荐,别裸 except)
        print("  转换失败:", e)
    except (TypeError, KeyError):  # 多个一起捕获
        print("  其他类型错误")
    else:
        print("  没出异常才走这里")
    finally:
        print("  无论如何都走这里(关资源)")

    # 主动抛出 & 自定义异常
    def bind(sn: str):
        if not sn:
            raise ValueError("sn 不能为空")  # ≈ JS throw / Java throw
        return True

    try:
        bind("")
    except ValueError as e:
        print("  捕获:", e)

    # with = 自动关闭资源(≈ Java try-with-resources / JS 的 using)
    # 不用手动 close(),出了代码块自动关
    with open("_demo_tmp.txt", "w", encoding="utf-8") as f:  # ★ Windows 一定要写 encoding!
        f.write("hello\n")
        f.writelines(["a\n", "b\n"])
    with open("_demo_tmp.txt", encoding="utf-8") as f:
        print("  读全部:", repr(f.read()))
        f.seek(0)
        for line in f:  # 大文件这么读,不一次性加载
            print("  逐行:", line.strip())

    import os
    os.remove("_demo_tmp.txt")

    # JSON(前后端对接天天用)
    payload = {"sn": "A1", "online": True}
    s = json.dumps(payload, ensure_ascii=False)  # 对象 -> 字符串(ensure_ascii=False 才不出 \uXXXX)
    print("  dumps:", s)
    print("  loads:", json.loads(s))  # 字符串 -> 对象


# ---------------------------------------------------------------- 10. 装饰器 & async
def demo_decorator():
    h("10. 装饰器(Java 的注解 / JS 的高阶函数)")

    import time
    from functools import wraps

    def timing(fn):  # 装饰器本质是:接收函数,返回函数
        @wraps(fn)  # 保留原函数的名字和文档,不加会丢元信息
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            r = fn(*args, **kwargs)
            print(f"  {fn.__name__} 耗时 {time.perf_counter() - t0:.4f}s")
            return r

        return wrapper

    @timing  # ≈ 写 slow(2) 实际是 timing(slow)(2)
    def slow(n):
        time.sleep(0.01)
        return n * 2

    print("  结果:", slow(21))


def demo_async():
    h("11. async/await(和 JS 几乎一样,但多一层)")

    async def fetch(sn: str, delay: float) -> str:
        await asyncio.sleep(delay)  # await ≈ JS await,让出控制权
        return f"{sn} ok"

    async def main():
        # await 一个 = 串行
        r = await fetch("A1", 0.02)
        print("  串行:", r)

        # gather = 并发(≈ JS 的 Promise.all)
        rs = await asyncio.gather(fetch("A1", 0.02), fetch("A2", 0.02))
        print("  并发:", rs)

        # create_task = 先起个任务,后面再 await(≈ 不等 await 就发请求)
        task = asyncio.create_task(fetch("A3", 0.02))
        print("  任务中:", await task)

    asyncio.run(main())  # ★ 关键:JS 里顶层直接 await,Python 必须 asyncio.run() 启动事件循环

    print("""
  JS vs Python 对照:
    JS:      async function main(){}; main()          直接跑
    Python:  asyncio.run(main())                      必须有个"发动机"
    并发:     Promise.all([...])   <->  asyncio.gather(...)
    注意:    Python 里 time.sleep() 会阻塞整个事件循环!要用 asyncio.sleep()
    """)


# ---------------------------------------------------------------- 12. 常用内置函数
def demo_builtins():
    h("12. 高频内置函数速查")

    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    print("len/sum/min/max/sorted:", len(nums), sum(nums), min(nums), max(nums), sorted(nums))
    print("abs/round/divmod:", abs(-5), round(3.14159, 2), divmod(17, 5))  # divmod -> (3, 2)
    print("any/all:", any([0, 1]), all([1, 1]))  # ≈ JS some / every
    print("enumerate/zip/range:", list(range(3)), list(enumerate("ab")))
    print("map/filter(返回迭代器,要 list 包一层):",
          list(map(str, [1, 2])), list(filter(lambda x: x > 3, nums)))
    print("isinstance/type:", isinstance(1, int), type(1))
    print("dir(看对象有啥方法):", [m for m in dir("") if m.startswith("is")][:5])
    print("help 的替代品: help(str.split)")


# ---------------------------------------------------------------- 13. 模块 & 项目常见
def demo_module():
    h("13. 模块 / 包 / 习惯写法")

    print("""
  导入方式:
    import os                      -> os.path.join(...)      全名调用,最安全
    from datetime import datetime  -> datetime.now()         直接拿过来
    import numpy as np                                       起别名(约定俗成)
    from .models import Device                               相对导入(包内部用)

  ★ 每个 .py 文件都是一个模块;文件夹里有 __init__.py 才是"包"(3.3+ 可省略,但 FastAPI 项目建议保留)

  ★ 每个文件底部都该有这个(FastAPI 项目里你会天天看到):
      if __name__ == "__main__":
          main()
    含义:被 import 时不执行,直接 python xxx.py 时才执行
    ≈ Node.js 的 require.main === module

  命名习惯(PEP 8,和 Java/JS 都不一样):
    变量/函数  snake_case        get_device_by_sn
    常量       UPPER_CASE        MAX_HISTORY_MESSAGES
    类         PascalCase        DeviceRepo
    "私有"     _leading_under     _cache(约定,不强制)
    魔法方法   __dunder__         __init__ / __repr__
    """)


MODULES = {
    "types": demo_types, "string": demo_string, "list": demo_list,
    "dict": demo_dict, "set": demo_set_tuple, "comp": demo_comprehension,
    "func": demo_function, "class": demo_class, "error": demo_error_file,
    "decorator": demo_decorator, "async": demo_async,
    "builtins": demo_builtins, "module": demo_module,
}


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in MODULES:
        MODULES[sys.argv[1]]()
    elif len(sys.argv) > 1:
        print(f"未知模块。可选: {', '.join(MODULES)}")
    else:
        for fn in MODULES.values():
            fn()
        print("\n全部跑完。想单独看某一节: python python_syntax_cheatsheet.py dict")


if __name__ == "__main__":
    main()

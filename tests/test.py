def mian():
    # d = dict({"name": "bac", "age": 18})
    # print("".join(sorted(d["name"])))
    # a = 'abcdefghjkl'
    # print(a[-3:])
    # arr = ["1", "2", "3", "4", "1"]
    # arr.append(5)
    # print(arr)
    # # arr.pop()
    # # print(arr)
    # arr.pop(0)
    # print(arr)
    # # arr.remove(2)
    # # print(arr)
    # # print(arr.index(1))
    # print(",".join(str(x) for x in arr))
    # print(f"{3.14159:.2f}")
    # s = "Ab-abc"
    # print(s.upper(), s.lower(), s.title())
    # print(zip([1, 2], ["a", "b"]))
    # for i, v in enumerate(["a", "b", "c"]):
    #     print(f"  [{i}] = {v}")
    # for x, y in zip([1, 2], ["a", "b"]):  # 拉链合并
    #     print(f"  {x}->{y}")
    #
    # d = {"sn": "A1-0001", "online": True}
    #
    # print(d.setdefault("model", "doubao"))  # 没有就塞进去并返回,有就直接返回
    # print(d)
    # d.update({"model": "deepseek", "ver": 2})  # ≈ JS Object.assign / {...a, ...b}
    # print(d)
    # d.pop("model")
    # print(d)
    #
    # def bad(items=[]):  # 这个 [] 只创建一次,所有调用共享!
    #     items.append(1)
    #     return items
    #
    # def good(items=None):
    #     items = items if items is not None else []
    #     items.append(1)
    #     return items
    #
    # r1, r2 = bad(), bad()
    # print(f"bad: 第一次={r1} 第二次={r2} 是同一个对象? {r1 is r2}  ← 被污染了")
    # g1, g2 = good(), good()
    # print(f"good: 第一次={g1} 第二次={g2} 是同一个对象? {g1 is g2}")

    # def fn(a, b):
    #     print(a, b)
    #     return a + b

    # print(fn(*[1, 2]))

    #     print(fn(**{"a": 1, "b": 2}))
    #
    #     def test(*args, **kwargs):
    #         print(args)
    #         print(kwargs)
    #
    #     test(*[1, 2], **{"a": 1, "b": 2}, **{"a1": 1, "b1": 2})

    def minmax(xs):
        return min(xs), max(xs)

    print(minmax([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    mi, mx = minmax([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(mi, mx)


if __name__ == "__main__":
    mian()

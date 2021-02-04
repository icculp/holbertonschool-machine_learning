#!/usr/bin/env python3

input = "aaaabbbcca"
def thing(input):
    out = []
    a = 0
    while a <= len(input):
        count = 0
        b = str(input[a])
        print("b", b)
        while input[a] == b:
            print("a", a)
            count += 1
            a += 1
            if a == len(input):
                break
        out.append((b, count))
        if a == len(input):
            break
    return out
print(thing(input))

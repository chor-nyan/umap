import unittest


def solution(a, b):
    return ans


class MyTestCase(unittest.TestCase):
    def test1(self):
        self.assertEqual(solution(a1, b1), c1)

    def test2(self):
        self.assertEqual(solution(a2, b2), c2)

    def test3(self):
        self.assertEqual(solution(a3, b3), c3)


if __name__ == '__main__':
    a = int(input())
    b = list(map(int, input().split()))
    print(solution(a, b))

    unittest.main()
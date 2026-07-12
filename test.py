import torch
import math

print(torch.__version__)
print(torch.cuda.is_available())



def gcd_power(a, b, c, d):
    g = math.gcd(a, c)
    return g ** min(b, d)

# 示例
print(gcd_power(1000000000000000000, 1000000000000000000, 1000000000000000000, 1000000000000000000)%998244353)  # 输出 32，因为 gcd(2^10, 4^5) = gcd(1024, 1024) = 1024 = 2^10
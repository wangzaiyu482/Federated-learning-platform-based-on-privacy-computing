import gmpy2
from gmpy2 import mpz
import binascii

# 生成指定随机状态下二进制位数为1024的随机素数
def gen_prime(rs):
    """
    生成二进制位数为1024的随机素数

    :param rs: gmpy2的随机状态对象
    :return: 生成的1024位随机素数
    """
    # 基于给定的随机状态rs生成一个1024位的随机大整数p
    p = gmpy2.mpz_urandomb(rs, 1024)
    while not gmpy2.is_prime(p):
        # 如果p不是素数，就将其值加1，然后继续检查是否为素数
        p = p + 1
    return p


# 生成RSA算法所需的密钥（两个大素数p和q）
def gen_key():
    """
    生成密钥

    :return: 生成的两个大素数p和q
    """
    # 创建一个gmpy2的随机状态对象
    rs = gmpy2.random_state()
    # 调用gen_prime函数生成第一个大素数p
    p = gen_prime(rs)
    # 调用gen_prime函数生成第二个大素数q
    q = gen_prime(rs)
    return p, q


# 对输入消息进行加密操作
def encrypt(e, n, message):
    """
    将输入消息转换成16进制数字并加密，支持utf-8字符串

    :param e: 公钥指数
    :param n: 模数（由p和q相乘得到）
    :param message: 待加密的消息（utf-8字符串）
    :return: 加密后的密文
    """
    # 将输入的消息先编码为utf-8字节串，再通过binascii.hexlify函数转换成十六进制字符串形式，
    # 最后使用mpz将其转换为gmpy2库中的大整数类型（以16进制解析），得到明文对应的大整数M
    M = mpz(binascii.hexlify(message.encode('utf-8')), 16)
    # 根据RSA加密公式ciphertext = plaintext ^ e mod n，使用gmpy2.powmod函数对M进行加密操作，得到密文C
    C = gmpy2.powmod(M, e, n)
    return C


# 对输入的密文进行解密并还原出原始消息
def decrypt(d, n, C):
    """
    对输入的密文进行解密并解码

    :param d: 私钥指数
    :param n: 模数（由p和q相乘得到）
    :param C: 待解密的密文
    :return: 解密后的原始消息（utf-8字符串）
    """
    # 依据RSA解密公式plaintext = ciphertext ^ d mod n，使用gmpy2.powmod函数对密文C进行解密操作，得到解密后的大整数M
    M = gmpy2.powmod(C, d, n)
    # 将解密后的大整数M通过format函数格式化为十六进制字符串形式，
    # 再使用binascii.unhexlify函数将其转换回字节串，
    # 最后通过decode('utf-8')将字节串解码为utf-8编码的字符串，即还原出原始消息并返回
    return binascii.unhexlify(format(M, 'x')).decode('utf-8')


def main():
    # 密钥生成部分
    # 调用gen_key函数生成两个大素数p和q
    p, q = gen_key()
    # 计算模数n，即p和q的乘积
    n = p * q
    # 计算欧拉函数phi，即(p - 1) * (q - 1)
    phi = (p - 1) * (q - 1)
    # 选择常用的公钥指数e，这里取65537
    e = 65537
    # 通过gmpy2.invert函数计算e关于phi的模逆，得到私钥指数d
    d = gmpy2.invert(e, phi)

    # 输入消息部分
    # 提示用户输入待加密的消息
    message = input('输入待加密的消息：\n')

    # 加密部分
    # 调用encrypt函数对输入的消息进行加密，传入公钥指数e、模数n和待加密消息message，得到密文C
    C = encrypt(e, n, message)
    # 打印出十六进制密文
    print('16进制密文：', hex(C))

    # 解密部分
    # 调用decrypt函数对前面生成的密文C进行解密，传入私钥指数d、模数n和密文C，得到解密后的消息
    print('解密后的消息：', decrypt(d, n, C))


if __name__ == '__main__':
    main()
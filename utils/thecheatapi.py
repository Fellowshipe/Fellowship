import base64

from Cryptodome.Cipher import AES


def decrypt(text, key):
    key = key.encode('utf-8')

    if len(key) < 32:
        key = key.ljust(32, b'\0')  # 키 길이가 32바이트보다 짧으면 0으로 패딩
    elif len(key) > 32:
        key = key[:32]  # 키 길이가 32바이트보다 길면 잘라냄

    key = str_to_bytes(key)
    
    decode = base64.b64decode(text)
    iv = decode[:AES.block_size]
    crypto = AES.new(key, AES.MODE_CBC, iv)
    dec = crypto.decrypt(decode[AES.block_size:]).decode('utf-8')
    return _unpad(dec)


def str_to_bytes(data):
    u_type = type(b''.decode('utf8'))
    if isinstance(data, u_type):
        return data.encode('utf8')
    return data


def _unpad(s):
    return s[:-ord(s[len(s)-1:])]


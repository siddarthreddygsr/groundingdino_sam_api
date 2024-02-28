import secrets
import string

def randomizer(length = 10):
    alphanumeric_chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphanumeric_chars) for _ in range(length))

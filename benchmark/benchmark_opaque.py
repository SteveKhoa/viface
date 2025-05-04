import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from benchmark.constant import N_REPEAT
from lib import opaque
from os import urandom
import time


ids = opaque.Ids("user", "server")
pwdU = "123456789"  # is the user's password.
skS = b'~|}\x97qT\x7f\xc7?\n\x93\x16\x81\xcc\x12\xce\x1f\x0e\x83\xa9\xdc\x1b\xb4\x02"S\xc3n\x03\xdc\x81b'
pwdU2 = "123456789"
context = "MyApp-v0.2"


def benchmark(repeat: int = 500):
    accumulated_registration_client_time = 0.0
    accumulated_registration_server_time = 0.0
    accumulated_login_client_time = 0.0
    accumulated_login_server_time = 0.0

    for i in range(repeat):
        # Registration
        start = time.time()
        secU, client_pub = opaque.CreateRegistrationRequest(pwdU)
        end = time.time()
        accumulated_registration_client_time += end - start

        start = time.time()
        secS, server_pub = opaque.CreateRegistrationResponse(client_pub, skS)
        end = time.time()
        accumulated_registration_server_time += end - start

        start = time.time()
        rec0, export_key = opaque.FinalizeRequest(secU, server_pub, ids)
        end = time.time()
        accumulated_registration_client_time += end - start

        start = time.time()
        rec1 = opaque.StoreUserRecord(secS, rec0)
        end = time.time()
        accumulated_registration_server_time += end - start

        # Login
        start = time.time()
        pub, secU = opaque.CreateCredentialRequest(pwdU2)
        end = time.time()
        accumulated_login_client_time += end - start

        start = time.time()
        resp, sk, secS = opaque.CreateCredentialResponse(pub, rec1, ids, context)
        end = time.time()
        accumulated_login_server_time += end - start

        start = time.time()
        sk, authU, export_key = opaque.RecoverCredentials(resp, secU, context, ids)
        end = time.time()
        accumulated_login_client_time += end - start

        start = time.time()
        opaque.UserAuth(secS, authU)
        end = time.time()
        accumulated_login_server_time += end - start

        print("i=", i)

    client_registration_time = accumulated_registration_client_time / float(repeat)
    server_registration_time = accumulated_registration_server_time / float(repeat)
    client_login_time = accumulated_login_client_time / float(repeat)
    server_login_time = accumulated_login_server_time / float(repeat)

    print("Registration Execution Time")
    print("client (s) =", client_registration_time)
    print("server (s) =", server_registration_time)
    print("Login Execution Time")
    print("client (s) =", client_login_time)
    print("server (s) =", server_login_time)


if __name__ == "__main__":
    benchmark(N_REPEAT)

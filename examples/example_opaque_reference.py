from lib import opaque
from os import urandom
import base64

"""Details: https://pypi.org/project/opaque/
"""

# wrap the IDs into an opaque.Ids struct:
ids=opaque.Ids("user", "server")

# 4-step Registration
pwdU = "123456789"  # is the user's password.
skS = b'~|}\x97qT\x7f\xc7?\n\x93\x16\x81\xcc\x12\xce\x1f\x0e\x83\xa9\xdc\x1b\xb4\x02"S\xc3n\x03\xdc\x81b'
secU, client_pub = opaque.CreateRegistrationRequest(pwdU)
secS, server_pub = opaque.CreateRegistrationResponse(client_pub, skS)
rec0, export_key = opaque.FinalizeRequest(secU, server_pub, ids)
rec1 = opaque.StoreUserRecord(secS, rec0)


# OPAQUE SESSION (AKE)
pwdU2 = "123456789"
pub, secU = opaque.CreateCredentialRequest(pwdU2)
context = "MyApp-v0.2"
resp, sk, secS = opaque.CreateCredentialResponse(pub, rec1, ids, context)
print(sk)
sk, authU, export_key = opaque.RecoverCredentials(resp, secU, context, ids)
print(sk)

opaque.UserAuth(secS, authU)

print("Done")
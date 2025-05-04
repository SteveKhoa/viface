BINARIZER_REPORT_TEMPLATE = """
---
Binarizer Benchmarking    
              
+ n_samples %(n_samples)s
+ bitlength %(bitstring_length)s bits
+ nbits     %(nbits)s bits

* Execution Time (s) : %(execution_time)s
* Overlapped AUC, Euclidean : %(euclid_auc)s
* Overlapped AUC, Hamming   : %(hamming_auc)s
---
"""

KEYGEN_REPORT_TEMPLATE = """
---
Keygen Benchmarking

. FMR should be as low as possible, looking for < 0.001.
. FMNR is expected to be low, but can be compromised, ~ 0.5 would be ok.

+ n_samples %(n_samples)s

* FMR   : %(false_matchrate)s
* FNMR  : %(false_nonmatchrate)s
* Registration (s): %(total_registration_time)s
* Login (s): %(total_login_time)s
---
"""
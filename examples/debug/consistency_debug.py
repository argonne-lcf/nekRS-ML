# debug consistency based on the saved squared_errors and pos files

import numpy as np
import sys

errors_size1 = np.load("squared_errors_local_size1.npy")
errors_rank0_size2 = np.load("squared_errors_local_rank0_size2.npy")
errors_rank1_size2 = np.load("squared_errors_local_rank1_size2.npy")

pos_size1 = np.load("pos_size1.npy").reshape(-1, 3)
pos_rank0_size2 = np.load("pos_rank0_size2.npy").reshape(-1, 3)
pos_rank1_size2 = np.load("pos_rank1_size2.npy").reshape(-1, 3)


# Brute force the analysis
# Loop over the nodes in the size 1 case
for i in range(pos_size1.shape[0]):
    # Find the corresponding node in the size 2 case
    # need to check both the rank 0 and rank 1 case
    # becase the node could be in the halo of both ranks
    is_rank0_found = False
    is_rank1_found = False
    for j in range(pos_rank0_size2.shape[0]):
        if np.allclose(pos_size1[i], pos_rank0_size2[j]):
            print(
                f"Node {i}:{pos_size1[i]} in the size 1 case is the same as node {j}:{pos_rank0_size2[j]} in the rank 0 size 2 case",
                flush=True,
            )
            is_rank0_found = True
            break
    for k in range(pos_rank1_size2.shape[0]):
        if np.allclose(pos_size1[i], pos_rank1_size2[k]):
            print(
                f"Node {i}:{pos_size1[i]} in the size 1 case is the same as node {k}:{pos_rank1_size2[k]} in the rank 1 size 2 case",
                flush=True,
            )
            is_rank1_found = True
            break

    # now check that the error is the same
    # for the case where both ranks are found
    # the value on the two ranks should be half of the value in the size 1 case
    if is_rank0_found and is_rank1_found:
        # assert np.allclose(errors_rank0_size2[j] + errors_rank1_size2[k], errors_size1[i]), "The error on the two ranks should be half of the value in the size 1 case"
        if not np.allclose(
            errors_rank0_size2[j] + errors_rank1_size2[k], errors_size1[i]
        ):
            print(
                f"Node {i}:{pos_size1[i]} has error {errors_size1[i]} on size 1 case, but {errors_rank0_size2[j]} on rank 0 case and {errors_rank1_size2[k]} on rank 1 case on size 2 case",
                flush=True,
            )
    # elif is_rank0_found:
    #    #assert np.allclose(errors_rank0_size2[j], errors_size1[i]), "The error on the rank 0 case should be the same as the value in the size 1 case"
    #    if not np.allclose(errors_rank0_size2[j], errors_size1[i]):
    #        print(f"Node {i}:{pos_size1[i]} has error {errors_size1[i]} on size 1 case, but {errors_rank0_size2[j]} on rank 0 case on size 2 case", flush=True)
    # elif is_rank1_found:
    #    if not np.allclose(errors_rank1_size2[k], errors_size1[i]):
    #        print(f"Node {i}:{pos_size1[i]} has error {errors_size1[i]} on size 1 case, but {errors_rank1_size2[k]} on rank 1 case on size 2 case", flush=True)
    # else:
    #    print(f"Node {i}:{pos_size1[i]} is not found in the rank 0 or rank 1 size 2 case", flush=True)
    #    sys.exit("Error: Node {i} in the size 1 case is not found in the rank 0 or rank 1 size 2 case")

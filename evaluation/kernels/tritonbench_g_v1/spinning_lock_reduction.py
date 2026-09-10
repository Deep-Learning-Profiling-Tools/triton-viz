import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def spinning_lock_kernel(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization

    # Perform reduction for every kth pid
    for iters in range(1, 10):
        if (pid % k == 0):
            next_pid = pid + 1

            while next_pid < pid + k and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                acc += acc1

                next_pid += 1
              
        # Store results using temporary storage P for every k-1 pids
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.atomic_xchg(locks + pid, 1)

        # Store final results in C
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)


def spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,)



##################################################################################################################################################


def test_spinning_lock():
    # Parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    M = 1024
    N = 1024
    num_sms = 304
    k = 3

    # Initialize tensors
    P = torch.zeros((num_sms * BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    locks = torch.zeros(num_sms, dtype=torch.int32, device='cuda')

    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Run the Triton kernel for different branches
    result = {}

    # Test case 1: pid % k == 0
    spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    result['test_case_1'] = C.clone()

    # Test case 2: pid % k != 0
    k = 2  # Change k to ensure pid % k != 0 for some pids
    spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    result['test_case_2'] = C.clone()

    # Test case 3: num_sms < pid + k
    num_sms = 2  # Reduce num_sms to ensure num_sms < pid + k
    spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    result['test_case_3'] = C.clone()

    # Test case 4: next_pid < pid + k and next_pid < num_sms
    num_sms = 5  # Adjust num_sms to ensure next_pid < pid + k and next_pid < num_sms
    spinning_lock(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    result['test_case_4'] = C.clone()

    return result

result_gold = test_spinning_lock()

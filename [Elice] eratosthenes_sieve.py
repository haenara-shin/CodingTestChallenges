def eratosthenes(n):
    sieve = [True] * n
    for i in range(2, n):
        if sieve[i] == True:
            for j in range(i+i, n, i):
                sieve[j] = False                
    return [ i for i in range(2,n) if sieve[i] == True]

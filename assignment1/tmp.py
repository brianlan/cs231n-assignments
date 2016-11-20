import sys

input_lines = sys.stdin.readlines()
for line in input_lines:
    M, N, P = [int(s) for s in line.split(' ')]
    ####################################################
    # TODO: your code goes here                        #
    # Use print() to output your answer                #
    ####################################################
    pass
    ####################################################
    #                END OF YOUR CODE                  #
    ####################################################



import sys

lines = sys.stdin.readlines()
for line in lines:
    M, N, P = [int(s) for s in line.split(' ')]
    if M < N:
        R = M * N + P
    else:
        R = (M + N) * P
    print(R)



ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = ALPHABET.lower()
T = int(sys.stdin.readline())
i = 0
while i < T:
    out_txt = ''
    txt = sys.stdin.readline()
    shift = int(sys.stdin.readline())
    for s in txt:
        if 65 <= ord(s) <= 90:
            out_txt += ALPHABET[(ord(s) - 65 + shift) % 26]
        elif 97 <= ord(s) <= 122:
            out_txt += alphabet[(ord(s) - 97 + shift) % 26]
    print(out_txt)
    i += 1
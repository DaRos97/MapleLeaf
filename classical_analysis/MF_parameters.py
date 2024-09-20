import numpy as np
import sys, os
import functions_MF as fs

order = 'Neel'        #one of the clasical orders

print("Computing LRO ",order)

if order == 'Coplanar':
    print("alpha: ",fs.alpha_)


res = {}
for type_op in ['alpha','beta']:
    res[type_op] = {}
    for loop in ['a','b','c','d','e']:
        pr = 1 if (type_op=='beta' and loop=='e') else 0
        res[type_op][loop] = fs.compute_loop(loop,type_op,order)
        #
        print(loop,': ',type_op)
        print(res[type_op][loop])

print('Bh: ',fs.Bh(res))
print('Bhp: ',fs.Bhp(res))
print('Bt: ',fs.Bt(res))
print('Btp: ',fs.Btp(res))
print('Bd: ',fs.Bd(res))

print('Ah: ',fs.Ah(res))
print('Ahp: ',fs.Ahp(res))
print('At: ',fs.At(res))
print('Atp: ',fs.Atp(res))
print('Ad: ',fs.Ad(res))

exit()

phases_result = {}
#Loop over all phases because we need all of them to compute the longer range ones
list_all = ['a','b','A','B']
for phi in Phis:
    A = A_dic[phi]
    Spins_in_loop = Spins_dic[order][phi]
    ################
    #First compute all the possible terms coming from the operators in the A-list
    r = A[0]
    for i1 in range(len(Spins_in_loop)-1):
        temp = A[i1+1]
        r2 = []
        for i2 in range(len(r)):
            for i3 in range(len(temp)):
                if (r[i2][0] == '-' and temp[i3][0] == '-'):
                    r2.append(r[i2][1:]+temp[i3][1:])
                elif (r[i2][0] == '-' and temp[i3][0] in list_all):
                    r2.append(r[i2]+temp[i3])
                elif (temp[i3][0] == '-' and r[i2][0] in list_all):
                    r2.append('-'+r[i2]+temp[i3][1:])
                else:
                    r2.append(r[i2]+temp[i3])
        r = r2
    #Now calculate their value one by one
    Calculus = complex(0,0)
    result = []
    for num,res in enumerate(r):
        temp_C = complex(1,0)
        if res[0] == '-':       #remove the minus in front if it is there
            res = res[1:]
            result.append(str(num+1)+': -')
            sign = -1
        else:
            result.append(str(num+1)+': ')
            sign = 1
        #Split the terms in the single multiplication
        terms = res.split('*')
        #For each term consider the spin sites one by one
        for i in range(1,len(Spins)+1):        #6 for phi_A1'
            temp = []
            for t in terms:             #extract terms with same lattice position
                if t[1] == str(i):
                    temp.append(t)
            #For each spin site compute the associated alfa, beta or gamma
            if len(temp) > 2:#loops through i have more than two terms in the single product -> passes through 1 two times -> 4 terms
                temp1 = [[temp[0],temp[3]],[temp[1],temp[2]]]
                for temp_ in temp1:
                    if (temp_[0][0] == 'A' and temp_[1][0] == 'a') or (temp_[0][0] == 'a' and temp_[1][0] == 'A'):          #alfa
                        result[num] += 'a' + str(i)
                        temp_C *= alfa(i-1,Spins)
                    elif (temp_[0][0] == 'A' and temp_[1][0] == 'b') or (temp_[0][0] == 'b' and temp_[1][0] == 'A'):        #gamma
                        result[num] += 'g' + str(i)
                        temp_C *= gamma(i-1,Spins)
                    elif (temp_[0][0] == 'B' and temp_[1][0] == 'a') or (temp_[0][0] == 'a' and temp_[1][0] == 'B'):        #gamma_
                        result[num] += 'g_' + str(i)
                        temp_C *= np.conjugate(gamma(i-1,Spins))
                    elif (temp_[0][0] == 'B' and temp_[1][0] == 'b') or (temp_[0][0] == 'b' and temp_[1][0] == 'B'):        #beta
                        result[num] += 'b' + str(i)
                        temp_C *= beta(i-1,Spins)
                    else:
                        print("not recognized 1")
                        exit()
            else:
                if (temp[0][0] == 'A' and temp[1][0] == 'a') or (temp[0][0] == 'a' and temp[1][0] == 'A'):
                    result[num] += 'a' + str(i)
                    temp_C *= alfa(i-1,Spins)
                elif (temp[0][0] == 'A' and temp[1][0] == 'b') or (temp[0][0] == 'b' and temp[1][0] == 'A'):
                    result[num] += 'g' + str(i)
                    temp_C *= gamma(i-1,Spins)
                elif (temp[0][0] == 'B' and temp[1][0] == 'a') or (temp[0][0] == 'a' and temp[1][0] == 'B'):
                    result[num] += 'g_' + str(i)
                    temp_C *= np.conjugate(gamma(i-1,Spins))
                elif (temp[0][0] == 'B' and temp[1][0] == 'b') or (temp[0][0] == 'b' and temp[1][0] == 'B'):
                    result[num] += 'b' + str(i)
                    temp_C *= beta(i-1,Spins)
                else:
                    print("not recognized 2")
                    exit()
        #Sum the term to the result
        Calculus += temp_C * sign

    if np.abs(Calculus) < 1e-10:
        phases_result[phi] = np.nan
        continue

    if phi == 'A1p':
        phases_result[phi] = cmath.phase(Calculus) + np.pi
    elif phi == 'A2':
        phases_result[phi] = cmath.phase(Calculus) + phases_result['B1p'] - np.pi
    elif phi == 'A2p':
        phases_result[phi] = cmath.phase(Calculus) - phases_result['B1p'] - np.pi
    elif phi == 'B2':
        phases_result[phi] = cmath.phase(Calculus) + phases_result['A1p']
    elif phi == 'B2p':
        phases_result[phi] = -cmath.phase(Calculus) - phases_result['A1p']
    elif phi == 'A3':
        phases_result[phi] = cmath.phase(Calculus) + 2*phases_result['A1p'] - p1*np.pi
    elif phi == 'B3':
        phases_result[phi] = cmath.phase(Calculus) - phases_result['A1p'] - phases_result['B1p'] - p1*np.pi
    elif phi in ['B1','B1p']:
        phases_result[phi] = cmath.phase(Calculus)

    if phases_result[phi] >= 4*np.pi:
        phases_result[phi] -= 2*np.pi
    if phases_result[phi] >= 2*np.pi:
        phases_result[phi] -= 2*np.pi
    if phases_result[phi] < -2*np.pi:
        phases_result[phi] += 2*np.pi
    if phases_result[phi] < 0:
        phases_result[phi] += 2*np.pi

print('Result = ',phases_result)

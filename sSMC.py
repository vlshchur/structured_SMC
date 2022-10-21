import numpy as np
from scipy import linalg
from scipy.integrate import odeint
from numpy import dot
import matplotlib.pyplot as plt

class lineage:
    def __init__(self, a, b, p):
        #a and b indicates ancestral material (1) or non-ancestral (0)
        self.a = a
        self.b = b
        self.p = p#0 or 1 population

    def __str__(self):
        return str(self.a)+str(self.b)+str(self.p)

class state:
    def __init__(self, lng):
        self.lng = lng
        self.l = len(lng)
        self.name()

    def name(self):
        lng_s = []
        for l in self.lng:
            lng_s.append( str(l.a) + str(l.b) + str(l.p) )
        lng_s.sort()
        self.n = ','.join(lng_s)

class param:
    def __init__(self):
        self.rho = 1.

numStates =[
#    [ [1,1,0] ],
#    [ [1,1,1] ],

    [ [1,1,0],[1,1,0] ],
    [ [1,1,0],[1,1,1] ],
    [ [1,1,1],[1,1,1] ],

    [ [1,0,0],[1,1,0] ],
    [ [1,0,0],[1,1,1] ],
    [ [1,0,1],[1,1,0] ],
    [ [1,0,1],[1,1,1] ],

    [ [0,1,0],[1,1,0] ],
    [ [0,1,0],[1,1,1] ],
    [ [0,1,1],[1,1,0] ],
    [ [0,1,1],[1,1,1] ],

    [ [0,1,0],[1,0,0] ],
    [ [0,1,0],[1,0,1] ],
    [ [0,1,1],[1,0,0] ],
    [ [0,1,1],[1,0,1] ],

    [ [1,0,0],[0,1,0],[1,1,0] ],
    [ [1,0,0],[0,1,0],[1,1,1] ],
    [ [1,0,0],[0,1,1],[1,1,0] ],
    [ [1,0,0],[0,1,1],[1,1,1] ],
    [ [1,0,1],[0,1,0],[1,1,0] ],
    [ [1,0,1],[0,1,0],[1,1,1] ],
    [ [1,0,1],[0,1,1],[1,1,0] ],
    [ [1,0,1],[0,1,1],[1,1,1] ],

    [ [1,0,0],[0,1,0],[0,1,0] ],
    [ [1,0,0],[0,1,0],[0,1,1] ],
    [ [1,0,0],[0,1,1],[0,1,1] ],
    [ [1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,1],[0,1,1],[0,1,1] ],

    [ [1,0,0],[1,0,0],[0,1,0] ],
    [ [1,0,0],[1,0,0],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,0] ],
    [ [1,0,0],[1,0,1],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,0] ],
    [ [1,0,1],[1,0,1],[0,1,1] ],

    [ [1,0,0],[1,0,0],[0,1,0],[0,1,0] ],
    [ [1,0,0],[1,0,0],[0,1,0],[0,1,1] ],
    [ [1,0,0],[1,0,0],[0,1,1],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,0],[1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,0],[1,0,1],[0,1,1],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,0],[0,1,0] ],
    [ [1,0,1],[1,0,1],[0,1,0],[0,1,1] ],
    [ [1,0,1],[1,0,1],[0,1,1],[0,1,1] ]
]

Mdim = len(numStates)

states = []
stateToNum = {}
i = 0
for st in numStates:
    lng = []
    for l in st:
        lng.append( lineage(l[0], l[1], l[2]) )
    states.append( state(lng) )
    stateToNum[states[-1].n] = i
    i += 1

def params(t):
    return np.array([1.0, 1.0, 2.0, 0.1, 0.1])

def Eq(P, t):
    return dot(dot(M,params(t)),P)

def Coal(st, i, k):
    if i > st.l or k > st.l:
        print("Wrong lineage index.")
    if st.lng[i].p != st.lng[k].p:
        print("Coalescence is impossible.")
    newState = [st.lng[j] for j in range(st.l) if i != j and k != j]
    newState.append(lineage(st.lng[i].a | st.lng[k].a, st.lng[i].b | st.lng[k].b, st.lng[i].p))
    newState = state(newState)
    if newState.l > 1:
        return stateToNum[newState.n]
    else:
        return -1

#Matrix entry [rho, l1, l2, m1, m2]

def Matrix():
    trRate = np.zeros( (Mdim, Mdim, 5), int )
    for s in range(Mdim):
        st = states[s]
        for i in range(st.l):
            #recombination
            if st.lng[i].a & st.lng[i].b:
                newState = [st.lng[j] for j in range(st.l) if i != j]
                newState.append(lineage(1,0,st.lng[i].p))
                newState.append(lineage(0,1,st.lng[i].p))
                newState = state(newState)
                trRate[s,stateToNum[newState.n],0] += 1
                trRate[s,s,0] -= 1
            #coalescence
            for k in range(i+1,st.l):
                if st.lng[i].p == st.lng[k].p:
                    newState_id = Coal(st, i, k)
                    if newState_id != -1:
                        trRate[s, newState_id, st.lng[i].p+1 ] += 1
                    trRate[s, s, st.lng[i].p+1 ] -= 1
            #migration
            newState = [st.lng[j] for j in range(st.l) if i != j]
            newState.append( lineage(st.lng[i].a,st.lng[i].b,(st.lng[i].p+1)%2) )
            newState = state(newState)
            trRate[s, stateToNum[newState.n], st.lng[i].p+3 ] += 1
            trRate[s, s, st.lng[i].p+3 ] -= 1
    return(trRate)

def FirstCoal(locus, popID):
    statesProb = np.zeros( Mdim )
    for s in range(Mdim):
        st = states[s]
        count_a, count_b = 0, 0
        ind = []
        for i, l in enumerate(st.lng):
            if locus == 'a':
                if l.a == 1 and l.p == popID:
                    ind.append(i)
                    count_a += 1
                count_b += l.b
            if locus == 'b':
                if l.b == 1 and l.p == popID:
                    ind.append(i)
                    count_b += 1
                count_a += l.a
        if count_a == 2 and count_b == 2:
            newState_id = Coal(st, ind[0], ind[1])
            statesProb[newState_id] += 1
    return statesProb

Pinit = np.zeros(Mdim)
Pinit[0] = 1.0

T = 1.0

#times = [0.01*i for i in range(0,1001)]
sol = odeint(Eq, Pinit, [T])

Pinit = dot(FirstCoal('a', 0), sol[0])














class MarginalDistr:

    def __init__(self, mu1, mu2, la1, la2):
        #mu1, mu2 - migration  rates
        #la1, la2 - coalescent rates
        #init_cond - vector of length 3
        self.mu1 = mu1
        self.mu2 = mu2

        self.la1 = la1
        self.la2 = la2

        #self.init_cond = np.array(init_cond)/sum(init_cond)

        self.SetMatrix(0.0)

    def m1(self, t):
        return self.mu1

    def m2(self, t):
        return self.mu2

    def l1(self, t):
        return self.la1

    def l2(self, t):
        return self.la2

    def SetMatrix(self, t):
        self.A = np.matrix( [[-2*self.m1(t)-self.l1(t), 0, self.m2(t)],
                             [0, -2*self.m2(t)-self.l2(t), self.m1(t)],
                             [2*self.m1(t), 2*self.m2(t), -self.m1(t) - self.m2(t)]] )

    def CalcProb(self, t, init_cond):
        MET = linalg.expm( dot(self.A, t) )
        return dot(MET, init_cond)

    def CalcDist(self, num_steps, step, init_cond):
        pr_dist = np.zeros(num_steps+1)
        MET = linalg.expm( dot(self.A, step) )
        for i in range(num_steps):
            pr_dist[i] = self.la1*init_cond[0]+self.la2*init_cond[1]
            init_cond = dot(MET, init_cond)
        pr_dist[num_steps] = self.la1*init_cond[0]+self.la2*init_cond[1]
        return pr_dist



def Matrix_old():
    trRate = np.zeros( (Mdim, Mdim, 5) )
    for s in range(Mdim):
        st = states[s]
        for i in range(st.l):
            #recombination
            if st.lng[i].a & st.lng[i].b:
                newState = [st.lng[j] for j in range(st.l) if i != j]
                newState.append(lineage(1,0,st.lng[i].p))
                newState.append(lineage(0,1,st.lng[i].p))
                newState = state(newState)
                trRate[s,stateToNum[newState.n],0] += 1
                trRate[s,s,0] -= 1
            #coalescence
            for k in range(i+1,st.l):
                if st.lng[i].p == st.lng[k].p:
                    newState = [st.lng[j] for j in range(st.l) if i != j and k != j]
                    newState.append(lineage(st.lng[i].a | st.lng[k].a, st.lng[i].b | st.lng[k].b, st.lng[i].p))
                    newState = state(newState)
                    if newState.l > 1:
                        trRate[s, stateToNum[newState.n], st.lng[i].p+1 ] += 1
                    trRate[s, s, st.lng[i].p+1 ] -= 1
            #migration
            newState = [st.lng[j] for j in range(st.l) if i != j]
            newState.append( lineage(st.lng[i].a,st.lng[i].b,(st.lng[i].p+1)%2) )
            newState = state(newState)
            trRate[s, stateToNum[newState.n], st.lng[i].p+3 ] += 1
            trRate[s, s, st.lng[i].p+3 ] -= 1
    return(trRate)

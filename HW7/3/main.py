"""MAE4700 HW7 cubic Hermite Euler-Bernoulli beam solver. Run: python main.py"""
from ast import literal_eval
from pathlib import Path
import math
import numpy as np

INPUTS = ['nodal coordinates.txt','connectivity array.txt','element flexural rigidities.txt','distributed loads.txt','applied loads.txt','essential BCs.txt']
OUTPUTS = ['nodal displacements.txt','reaction forces.txt','displacement.txt','bending moment.txt','shear force.txt']

def read_literal(name):
    return literal_eval(Path(name).read_text().strip())

def validate(coords, conn, eis, ps, loads, bcs):
    if not isinstance(coords,list) or len(coords)<2 or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in coords):
        raise ValueError('Coordinates must be a list of at least two finite numbers')
    if not isinstance(conn,list) or not conn or len(eis)!=len(conn) or len(ps)!=len(conn):
        raise ValueError('Connectivity, EI, and distributed load lists must have matching nonzero lengths')
    last=-float('inf')
    for pair in conn:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(n)!=int or n<0 or n>=len(coords) for n in pair):
            raise ValueError('Invalid two-node connectivity')
        a,b=pair
        if coords[b]<=coords[a] or coords[a]<last:
            raise ValueError('Elements must have positive lengths and be listed left to right')
        last=coords[a]
    if any(not isinstance(e,(int,float)) or not math.isfinite(e) or e<=0 for e in eis):
        raise ValueError('Every EI must be finite and positive')
    if any(not isinstance(p,(int,float)) or not math.isfinite(p) for p in ps):
        raise ValueError('Distributed loads must be finite numbers')
    seen={}
    for label,records in [('load',loads),('BC',bcs)]:
        if not isinstance(records,list): raise ValueError(label+' records must be a list')
        for rec in records:
            if not isinstance(rec,(list,tuple)) or len(rec)!=3: raise ValueError('Each '+label+' must be [node,dof,value]')
            n,d,v=rec
            if type(n)!=int or n<0 or n>=len(coords) or type(d)!=int or d not in (0,1) or not isinstance(v,(int,float)) or not math.isfinite(v):
                raise ValueError('Invalid '+label+' node, DOF, or value')
            if label=='BC':
                g=2*n+d
                if g in seen: raise ValueError('Duplicate prescribed DOF: '+str(g))
                seen[g]=v

def element_dofs(pair):
    i,j=pair
    return [2*i,2*i+1,2*j,2*j+1]

def beam_element(x1,x2,EI,p):
    L=x2-x1
    ke=EI/L**3*np.array([[12,6*L,-12,6*L],[6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],[6*L,2*L**2,-6*L,4*L**2]],float)
    fe=p*L/2*np.array([1,L/6,1,-L/6],float)
    return ke,fe

def assemble_system(coords,conn,eis,ps,loads):
    K=np.zeros((2*len(coords),2*len(coords)))
    F=np.zeros(2*len(coords))
    for pair,EI,p in zip(conn,eis,ps):
        gd=element_dofs(pair)
        ke,fe=beam_element(coords[pair[0]],coords[pair[1]],EI,p)
        for a in range(4):
            F[gd[a]]+=fe[a]
            for b in range(4): K[gd[a],gd[b]]+=ke[a,b]
    for n,d,v in loads: F[2*n+d]+=v
    return K,F

def solve_partitioned(K,F,bcs):
    fixed=np.array([2*n+d for n,d,v in bcs],dtype=int)
    vals=np.array([v for n,d,v in bcs],float)
    free=np.array([i for i in range(len(F)) if i not in set(fixed)],dtype=int)
    result=np.zeros(len(F))
    result[fixed]=vals
    if len(free):
        try: result[free]=np.linalg.solve(K[np.ix_(free,free)],F[free]-K[np.ix_(free,fixed)]@vals)
        except np.linalg.LinAlgError as exc: raise ValueError('Constrained stiffness matrix is singular; check supports and connectivity') from exc
    reactions=K@result-F
    return result,[float(reactions[i]) for i in fixed]

def postprocess_element(x1,x2,EI,de):
    L=x2-x1
    rows=[[],[],[]]
    for r in (0.,0.5,1.):
        N=np.array([1-3*r*r+2*r**3,L*(r-2*r*r+r**3),3*r*r-2*r**3,L*(-r*r+r**3)])
        Nxx=np.array([(-6+12*r)/L**2,(-4+6*r)/L,(6-12*r)/L**2,(-2+6*r)/L])
        Nxxx=np.array([12/L**3,6/L**2,-12/L**3,6/L**2])
        x=float(x1+r*L)
        for bucket,value in zip(rows,(N@de,EI*(Nxx@de),-EI*(Nxxx@de))): bucket.append([x,float(value)])
    return rows

def main():
    coords,conn,eis,ps,loads,bcs=[read_literal(name) for name in INPUTS]
    validate(coords,conn,eis,ps,loads,bcs)
    K,F=assemble_system(coords,conn,eis,ps,loads)
    d,reactions=solve_partitioned(K,F,bcs)
    results=[d.reshape((-1,2)).tolist(),reactions,[],[],[]]
    for pair,EI in zip(conn,eis):
        rows=postprocess_element(coords[pair[0]],coords[pair[1]],EI,d[element_dofs(pair)])
        for j in range(3): results[j+2].extend(rows[j])
    for name,values in zip(OUTPUTS,results): Path(name).write_text(repr(values)+'\n')
    print('Solved',len(conn),'elements; outputs written to current directory')

if __name__=='__main__': main()

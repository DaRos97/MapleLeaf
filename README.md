# Schwinger Boson Mean Field decoupling applied to Maple Leaf lattice

Hamiltonian:

We perform a chiral PSG classification -> 2 ansatze.

We look at classical magnetic orders -> classical phase diagram.

We perform minimization of the mean field parameters in the SBMFT and find the quantum phase diagram.

#Classical analysis
We derived the classification for the regular magnetic orders on the ML lattice. We chose to classify states with T1,T2 and R3 symmetry.
With also R6: FM, Neel, 6 orders with 6-site unit-cell denoted a-Cn, 2 orders with 6x4 sites in the UC called NC2, 1 order with 6x9 sites called NC3.
Only R3: 
    - six site unit cell: FM, Neel, aA3, bA3
    - 54 site unit cell: Coplanar with theta R = 0 or 2 pi/3
    - 24 site unit cell: Non-Coplanar icosahedra states

We derive the formulas for the energies, which may depend on someparameters like the orientation of the initial spin used to construct the order.

In `classical_phase_diagram.py` we computete classical phase digram for 1st nn Heisenberg model with 3 inequivalent bonds (Jh, Jt and Jd).

In `visualize_order.py` we plot a visualization of the orders in the phase diagram (FM, Neel, Coplanar, Non-Coplanar I and II big/small) 

In `MF_parameters.py' we compute the mean field parametersi (pairing and hopping) of the found classical orders.



# Computed
## mafalda
    cpd C3, nn=51, bounds=(-4,0), 'rand2bin', tol=1e-4, popsize = 15
    cpd C3, nn=101, bounds=(-4,0), 'rand2bin', tol=1e-4, popsize = 15

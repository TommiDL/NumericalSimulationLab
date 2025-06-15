## Parallel Random Number Generator

La routine é RANdom New York University (RANNYU). Scritta negli anni 80 con un moltiplicatore suggerito nel 2 volume di Knuth "Art of scientific computing" p 102. Riadattata poi a inizio anni 90 nel gruppo di M.H. Kalos (da parte del Dr. Panoff).

Si tratta di un Linear Congruential Generator with prime addend a 48 bit.
La relazione che genera x_{n+1} da x_n é la seguente: x_{n+1}= ( a x_{n} + c ) mod m
dove a é il moltiplicatore, c é l'addendo (numero primo con certe proprietà), m é il modulo, in questo caso m=2**48.
Con questo moltiplicatore e questo modulo il periodo del generatore risulta P=2**48~10**14. Gli x_n sono numeri interi e corrispondono allo stato salvato nella routine. Il numero random reale in [0,1) si ottiene con x_n/m.

Notare che per ogni c si ha un generatore diverso. In un lavoro di Percus e Kalos (che vi ho aggiunto) sono state mostrate le proprietà che due diversi c (c1 e c2) devono avere affinché le serie generate (x1_n e x2_n) siano effettivamente indipendenti. I 384 numeri primi che già avete in Primes soddisfano questo criterio. Seguendo quel criterio, un mio collaboratore (Dr. G. Bertaina) me ne ha ottenuti 32001 ... e più recentemente 64001. Li trovate in primes32001.in ed in primes64001.in, in ordine decrescente.

Notate che nella routine lo stato a 48 bit é salvato in 4 interi minori di 4096 (=2**12). Il tutto, comprese le operazioni, é scritto in base 4096 (anche in numeri primi nel file Primes, per questo sono coppie).

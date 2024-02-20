*ShapeDataBase.m
A 50x50x85595 matrix ShapeDataBase
ShapeDataBase(:,:,i) store the 50x50 pixelated matrix for the ith micorstructure

*Property.mat
 An 85595x4 matrix
 Property(i,:) store independent entries [C11,C12,C22,C66] of the stiffness matrix of the ith microstructure


Illustration for Property.mat:
----------------------------------------------------
Num for Data|	C11	C12	C22	C66	|
           1|   ...	...	... 	...	|
           2|	...	...	... 	...	|
         ...|	...	...	... 	...	|
       85595|	...	...	... 	...	|
----------------------------------------------------
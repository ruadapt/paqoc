OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[6];
//This initializes 9 quantum and 6 classical registers.

h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
x q[6];
x q[7];
x q[8];

ccx q[5], q[6], q[7];
cz q[7], q[8];
ccx q[5], q[6], q[7];

cu1(-pi/32) q[5], q[0];
cu1(-pi/16) q[5], q[1];
cu1(-pi/8) q[5], q[2];
cu1(-pi/4) q[5], q[3];
cu1(-pi/2) q[5], q[4];
cu1(-pi/16) q[4], q[0];
cu1(-pi/8) q[4], q[1];
cu1(-pi/4) q[4], q[2];
cu1(-pi/2) q[4], q[3];
cu1(-pi/8) q[3], q[0];
cu1(-pi/4) q[3], q[1];
cu1(-pi/2) q[3], q[2];
cu1(-pi/4) q[2], q[0];
cu1(-pi/2) q[2], q[1];
cu1(-pi/2) q[1], q[0];

h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];

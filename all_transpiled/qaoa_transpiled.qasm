OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[10];
h q[7];
h q[8];
h q[11];
h q[12];
h q[13];
cx q[12],q[13];
rz(0) q[13];
cx q[12],q[13];
cx q[12],q[7];
rz(0) q[7];
cx q[12],q[7];
h q[14];
h q[15];
h q[16];
h q[17];
cx q[12],q[17];
rz(0) q[17];
cx q[12],q[17];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[7];
cx q[13],q[8];
rz(0) q[7];
cx q[12],q[7];
cx q[12],q[17];
rz(0) q[17];
cx q[12],q[17];
rz(0) q[8];
cx q[13],q[8];
cx q[13],q[14];
rz(0) q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(0) q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[11],q[16];
cx q[13],q[8];
rz(0) q[16];
cx q[11],q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
rz(0) q[8];
cx q[13],q[8];
cx q[13],q[14];
rz(0) q[14];
cx q[13],q[14];
cx q[13],q[12];
rz(0) q[12];
cx q[13],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[7],q[12];
rz(0) q[12];
cx q[7],q[12];
cx q[7],q[8];
rz(0) q[8];
cx q[7],q[8];
cx q[7],q[12];
cx q[12],q[7];
cx q[7],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[11];
rz(0) q[11];
cx q[12],q[11];
cx q[13],q[14];
rz(0) q[14];
cx q[13],q[14];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[12];
cx q[13],q[12];
rz(0) q[12];
cx q[13],q[12];
cx q[7],q[8];
rz(0) q[8];
cx q[7],q[8];
cx q[9],q[14];
cx q[14],q[9];
cx q[9],q[14];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[7],q[8];
rz(0) q[8];
cx q[7],q[8];
cx q[7],q[12];
rz(0) q[12];
cx q[7],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(0) q[11];
cx q[12],q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[7],q[12];
rz(0) q[12];
cx q[7],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[9],q[8];
rz(0) q[8];
cx q[9],q[8];
cx q[9],q[14];
rz(0) q[14];
cx q[9],q[14];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[13];
rz(0) q[13];
cx q[8],q[13];
cx q[9],q[14];
rz(0) q[14];
cx q[9],q[14];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[8],q[13];
rz(0) q[13];
cx q[8],q[13];
cx q[14],q[13];
rz(0) q[13];
cx q[14],q[13];
h q[21];
cx q[16],q[21];
rz(0) q[21];
cx q[16],q[21];
cx q[16],q[15];
rz(0) q[15];
cx q[16],q[15];
rx(0) q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[16],q[21];
rz(0) q[21];
cx q[16],q[21];
cx q[16],q[15];
rz(0) q[15];
cx q[16],q[15];
rx(0) q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[21];
rz(0) q[21];
cx q[16],q[21];
cx q[16],q[15];
rz(0) q[15];
cx q[16],q[15];
cx q[10],q[15];
cx q[15],q[10];
cx q[10],q[15];
rx(0) q[16];
cx q[16],q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[6],q[11];
cx q[11],q[6];
cx q[6],q[11];
cx q[7],q[6];
rz(0) q[6];
cx q[7],q[6];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
rz(0) q[10];
cx q[5],q[10];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
rx(0) q[5];
cx q[8],q[7];
cx q[7],q[8];
cx q[8],q[7];
cx q[9],q[8];
rz(0) q[8];
cx q[9],q[8];
cx q[7],q[8];
rz(0) q[8];
cx q[7],q[8];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[14],q[9];
cx q[8],q[13];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[12];
rz(0) q[12];
cx q[13],q[12];
rx(0) q[13];
cx q[7],q[12];
rz(0) q[12];
cx q[7],q[12];
rx(0) q[7];
rz(0) q[9];
cx q[14],q[9];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
rz(0) q[12];
cx q[13],q[12];
rx(0) q[13];
cx q[8],q[9];
rz(0) q[9];
cx q[8],q[9];
cx q[14],q[9];
cx q[7],q[8];
cx q[8],q[7];
cx q[7],q[8];
cx q[7],q[12];
rz(0) q[12];
cx q[7],q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rx(0) q[7];
cx q[9],q[14];
cx q[14],q[9];
cx q[14],q[13];
rz(0) q[13];
cx q[14],q[13];
rx(0) q[13];
rx(0) q[14];

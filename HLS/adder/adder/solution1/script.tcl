############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project adder
set_top add
add_files adder.cpp
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e} -tool vivado
create_clock -period 10 -name default
#source "./adder/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -rtl vhdl -format ip_catalog -description "Simple adder for 32-bit floats" -vendor "ChristianBillPedersen" -library "CBP_lib" -display_name "Simple_adder_32f"

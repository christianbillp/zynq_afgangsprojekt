--Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
----------------------------------------------------------------------------------
--Tool Version: Vivado v.2018.2 (win64) Build 2258646 Thu Jun 14 20:03:12 MDT 2018
--Date        : Thu Dec 13 12:09:25 2018
--Host        : cbp-desktop running 64-bit major release  (build 9200)
--Command     : generate_target adder_wrapper.bd
--Design      : adder_wrapper
--Purpose     : IP block netlist
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity adder_wrapper is
end adder_wrapper;

architecture STRUCTURE of adder_wrapper is
  component adder is
  end component adder;
begin
adder_i: component adder
 ;
end STRUCTURE;

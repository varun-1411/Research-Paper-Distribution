\ Model PaperAssignment
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xdeb6c1229e767bff
Maximize
  90 assign_Radhika_thomas2018 + 10 assign_Radhika_wang2019
   + 10 assign_xyz_loachim1998 + 70 assign_xyz_orda1990
   + 20 assign_xyz_thomas2018
Subject To
 OnePerTeam_Radhika: assign_Radhika_ahmadi2025 + assign_Radhika_dean2004
   + assign_Radhika_ding2008 + assign_Radhika_eppstein1998
   + assign_Radhika_fakcharoenphol2006 + assign_Radhika_kilein2005
   + assign_Radhika_loachim1998 + assign_Radhika_orda1990
   + assign_Radhika_thomas2018 + assign_Radhika_wang2019 = 1
 OnePerTeam_xyz: assign_xyz_ahmadi2025 + assign_xyz_dean2004
   + assign_xyz_ding2008 + assign_xyz_eppstein1998
   + assign_xyz_fakcharoenphol2006 + assign_xyz_kilein2005
   + assign_xyz_loachim1998 + assign_xyz_orda1990 + assign_xyz_thomas2018
   + assign_xyz_wang2019 = 1
 OnePerPaper_ahmadi2025: assign_Radhika_ahmadi2025 + assign_xyz_ahmadi2025
   <= 1
 OnePerPaper_dean2004: assign_Radhika_dean2004 + assign_xyz_dean2004 <= 1
 OnePerPaper_ding2008: assign_Radhika_ding2008 + assign_xyz_ding2008 <= 1
 OnePerPaper_eppstein1998: assign_Radhika_eppstein1998
   + assign_xyz_eppstein1998 <= 1
 OnePerPaper_fakcharoenphol2006: assign_Radhika_fakcharoenphol2006
   + assign_xyz_fakcharoenphol2006 <= 1
 OnePerPaper_kilein2005: assign_Radhika_kilein2005 + assign_xyz_kilein2005
   <= 1
 OnePerPaper_loachim1998: assign_Radhika_loachim1998
   + assign_xyz_loachim1998 <= 1
 OnePerPaper_orda1990: assign_Radhika_orda1990 + assign_xyz_orda1990 <= 1
 OnePerPaper_thomas2018: assign_Radhika_thomas2018 + assign_xyz_thomas2018
   <= 1
 OnePerPaper_wang2019: assign_Radhika_wang2019 + assign_xyz_wang2019 <= 1
Bounds
Binaries
 assign_Radhika_ahmadi2025 assign_Radhika_dean2004 assign_Radhika_ding2008
 assign_Radhika_eppstein1998 assign_Radhika_fakcharoenphol2006
 assign_Radhika_kilein2005 assign_Radhika_loachim1998
 assign_Radhika_orda1990 assign_Radhika_thomas2018 assign_Radhika_wang2019
 assign_xyz_ahmadi2025 assign_xyz_dean2004 assign_xyz_ding2008
 assign_xyz_eppstein1998 assign_xyz_fakcharoenphol2006
 assign_xyz_kilein2005 assign_xyz_loachim1998 assign_xyz_orda1990
 assign_xyz_thomas2018 assign_xyz_wang2019
End

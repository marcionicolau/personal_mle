SELECT fd.NDB_No, fg.FdGrp_desc, fd.Long_Desc, fd.Shrt_Desc, fd.Pro_Factor_, fd.Fat_Factor_, fd.CHO_Factor
        FROM FD_GROUP AS fg
INNER JOIN FOOD_DES as fd ON (fg.FdGrp_Cd = fd.FdGrp_cd)
WHERE fg.FdGrp_Cd in (100, 400, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1600, 1800, 1900, 2000, 2100, 2500, 3600)
SELECT fd.NDB_No,
       fg.FdGrp_desc,
       fd.Long_Desc,
       fd.Shrt_Desc,
       fd.Pro_Factor_,
       fd.Fat_Factor_,
       fd.CHO_Factor,
       ninfo.prot_val,
       ninfo.fat_val,
       ninfo.cho_val,
       fd.Pro_Factor_ * ninfo.prot_val AS pro_cal,
       fd.Fat_Factor_ * ninfo.fat_val AS fat_cal,
       fd.CHO_Factor * ninfo.cho_val AS cho_cal
FROM FD_GROUP AS fg
       INNER JOIN FOOD_DES as fd ON (fg.FdGrp_Cd = fd.FdGrp_cd)
       LEFT OUTER JOIN (SELECT nd.NDB_No,
                               max(case when nd.Nutr_No = 203 then nd.Nutr_Val end) as prot_val,
                               max(case when nd.Nutr_No = 204 then nd.Nutr_Val end) as fat_val,
                               max(case when nd.Nutr_No = 205 then nd.Nutr_Val end) as cho_val
                        FROM NUT_DATA as nd
                        GROUP BY nd.NDB_No
                        ORDER BY nd.NDB_No) AS ninfo ON (ninfo.NDB_No = fd.NDB_No)
WHERE fg.FdGrp_Cd in (100, 400, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1600, 1800, 1900, 2000, 2100, 2500, 3600)

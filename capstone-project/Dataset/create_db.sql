CREATE TABLE IF NOT EXISTS "DATA_SRC" (
    "DataSrc_ID" TEXT PRIMARY KEY,
    "Authors" TEXT,
    "Title" TEXT,
    "Year" TEXT,
    "Journal" TEXT,
    "Vol_City" TEXT,
    "Issue_State" TEXT,
    "Start_Page" TEXT,
    "End_Page" TEXT
);
CREATE TABLE IF NOT EXISTS "DATSRCLN" (
    "NDB_No" TEXT,
    "Nutr_No" TEXT,
    "DataSrc_ID" TEXT,
    PRIMARY KEY ( "NDB_No", "Nutr_No", "DataSrc_ID" )
);
CREATE TABLE IF NOT EXISTS "DERIV_CD" (
    "Deriv_Cd" TEXT PRIMARY KEY,
    "Deriv_Desc" TEXT
);
CREATE TABLE IF NOT EXISTS "FD_GROUP" (
    "FdGrp_Cd" TEXT PRIMARY KEY,
    "FdGrp_desc" TEXT
);
CREATE TABLE IF NOT EXISTS "FOOD_DES" (
    "NDB_No" TEXT PRIMARY KEY,
    "FdGrp_Cd" TEXT,
    "Long_Desc" TEXT,
    "Shrt_Desc" TEXT,
    "Com_Name" TEXT,
    "ManufacName" TEXT,
    "Survey" TEXT,
    "Ref_Desc" TEXT,
    "Refuse" INTEGER,
    "Sci_Name" TEXT,
    "N_FActor" DOUBLE,
    "Pro_Factor_" DOUBLE,
    "Fat_Factor_" DOUBLE,
    "CHO_Factor" DOUBLE
);
CREATE TABLE IF NOT EXISTS "FOOTNOTE" (
    "NDB_No" TEXT,
    "Footnt_No" TEXT,
    "Footnt_Typ" TEXT,
    "Nutr_No" TEXT,
    "Footnt_Txt" TEXT
);
CREATE TABLE IF NOT EXISTS "LANGDESC" (
    "Factor" TEXT,
    "Description" TEXT
);
CREATE TABLE IF NOT EXISTS "LANGUAL" (
    "NDB_No" TEXT,
    "Factor" TEXT,
    PRIMARY KEY ( "NDB_No", "Factor" )
);
CREATE TABLE IF NOT EXISTS "NUT_DATA" (
    "NDB_No" TEXT,
    "Nutr_No" TEXT,
    "Nutr_Val" DOUBLE,
    "Num_Data_Pts" INTEGER,
    "Std_Error" DOUBLE,
    "Src_Cd" TEXT,
    "Deriv_Cd" TEXT,
    "Ref_NDB_No" TEXT,
    "Add_Nutr_Mark" TEXT,
    "Num_Studies" INTEGER,
    "Min" DOUBLE,
    "Max" DOUBLE,
    "DF" INTEGER,
    "Low_EB" DOUBLE,
    "Up_EB" DOUBLE,
    "Stat_Cmt" TEXT,
    "AddMod_Date" TEXT,
    PRIMARY KEY ( "NDB_No", "Nutr_No" )
);
CREATE TABLE IF NOT EXISTS "NUTR_DEF" (
    "Nutr_no" TEXT PRIMARY KEY,
    "Units" TEXT,
    "Tagname" TEXT,
    "NutrDesc" TEXT,
    "Num_Dec" TEXT,
    "SR_Order" INTEGER
);
CREATE TABLE IF NOT EXISTS "SRC_CD" (
    "Src_Cd" TEXT PRIMARY KEY,
    "SrcCd_Desc" TEXT
);
CREATE TABLE IF NOT EXISTS "WEIGHT" (
    "NDB_No" TEXT,
    "Seq" TEXT,
    "Amount" DOUBLE,
    "Msre_Desc" TEXT,
    "Gm_Wgt" DOUBLE,
    "Num_Data_pts" INTEGER,
    "Std_Dev" DOUBLE
);

import helper
import pandas as pd

head_files = {'FD_GROUP.txt': ['FdGrp_Cd', 'FdGrp_desc'],
              'LANGUAL.txt': ["NDB_No", "Factor"],
              'LANGDESC.txt': ["Factor", "Description"],
              'FOOD_DES.txt': ["NDB_No", "FdGrp_Cd", "Long_Desc", "Shrt_Desc",
                               "Com_Name", "ManufacName", "Survey", "Ref_Desc",
                               "Refuse", "Sci_Name", "N_FActor", "Pro_Factor_",
                               "Fat_Factor_", "CHO_Factor"],
              'NUT_DATA.txt': ["NDB_No", "Nutr_No", "Nutr_Val", "Num_Data_Pts", "Std_Error",
                               "Src_Cd", "Deriv_Cd", "Ref_NDB_No", "Add_Nutr_Mark",
                               "Num_Studies", "Min", "Max", "DF", "Low_EB", "Up_EB",
                               "Stat_Cmt", "AddMod_Date"],
              'NUTR_DEF.txt': ["Nutr_no",
                               "Units", "Tagname", "NutrDesc", "Num_Dec", "SR_Order"],
              'WEIGHT.txt': ["NDB_No", "Seq", "Amount", "Msre_Desc",
                             "Gm_Wgt", "Num_Data_pts", "Std_Dev"]
              }

base_files = head_files.keys()

all_data = helper.convert_to_csv('SR-Leg_ASC', base_files, head_files, 'Dataset')

nutri_data = pd.read_csv('Dataset/nutritional_usda.csv')
fill_data = nutri_data.dropna()
fill_data.to_csv('Dataset/clean_data.csv', index=False)

ddict = fill_data[['Shrt_Desc', 'Pro_Factor_','Fat_Factor_','CHO_Factor','prot_val','fat_val','cho_val','pro_cal','fat_cal','cho_cal']]
ddict.set_index(keys='Shrt_Desc', inplace=True)
proxy_data = ddict.to_dict('index')

helper.save_proxy_data(proxy_data)


    index=fill_data['Shrt_Desc'])
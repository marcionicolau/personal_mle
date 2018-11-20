import helper

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

all_data = helper.convert_to_csv('SR-Leg_ASC', base_files, head_files, 'DS4')

F_groups = [100, 400, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1600, 1800, 1900, 2000, 2100, 2500, 3600]

fd_group[fd_group.FdGrp_Cd.isin(F_groups)]

# In[25]:


food_des[food_des.FdGrp_Cd.isin(F_groups)]

# In[59]:


food_des[food_des.FdGrp_Cd.isin(F_groups)].shape

# In[103]:


reg_by_grp = pd.DataFrame([{'FdGrp_Cd': f, 'FdGrp_items': food_des[food_des.FdGrp_Cd == f].shape[0]} for f in F_groups])
flt_grp = fd_group[fd_group.FdGrp_Cd.isin(F_groups)]

# In[105]:


sz_by_grp = flt_grp.merge(reg_by_grp, left_on='FdGrp_Cd', right_on='FdGrp_Cd', how='outer')

# In[107]:


sz_by_grp

# In[ ]:

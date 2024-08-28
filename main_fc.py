from generate_static_model import gen_static_model
import os, shutil, subprocess
import concurrent.futures
import numpy as np

def run_eclipse(path_to_Datafile):
    subprocess.run("cmd /c \"eclrun eclipse " + path_to_Datafile + "\"", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Define Variables for Dense DOE
# Variable = [Base, Low, High]
upper_zone_thicknesses = [500, 200, 1000]
lower_zone_thicknesses = [500, 200, 1000]
upper_zone_ntgs = [0.62, 0.15, 0.7]
rock_compressibilities = [3, 1, 10]
P_inits = [18000, 16000, 20000]
lower_zone_permeabilities = [10, 5, 60]
lower_zone_ntgs = [0.9]
P_diffs = [1500]

# # Run Lobe Model
# print("started lobe modeling")
# os.system("python lobe_modeling.py")
# print("done with lobe modeling")

# # Create Entire Directory Tree
# # Fill Subdirectories with Eclipse Files and Start Eclipse Runs
# for P_diff in P_diffs:
#     dir_path_0 = "../P_diff_" + str(P_diff).replace(".","_")
#     if not os.path.exists(dir_path_0):
#         os.makedirs(dir_path_0)

#     for lower_zone_ntg in lower_zone_ntgs:
#         dir_path_1 = "/low_NTG_" + str(lower_zone_ntg).replace(".","_")
#         if not os.path.exists(dir_path_0+dir_path_1):
#             os.makedirs(dir_path_0+dir_path_1)
        
#         for lower_zone_permeability in lower_zone_permeabilities:
#             dir_path_2 = "/low_perm_" + str(lower_zone_permeability).replace(".","_")
#             if not os.path.exists(dir_path_0+dir_path_1+dir_path_2):
#                 os.makedirs(dir_path_0+dir_path_1+dir_path_2)

#             for P_init in P_inits:
#                 dir_path_3 = "/P_init_" + str(P_init).replace(".","_")
#                 if not os.path.exists(dir_path_0+dir_path_1+dir_path_2+dir_path_3):
#                     os.makedirs(dir_path_0+dir_path_1+dir_path_2+dir_path_3)

#                 for rock_compressibility in rock_compressibilities:
#                     dir_path_4 = "/rock_compress_" + str(rock_compressibility).replace(".","_")
#                     if not os.path.exists(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4):
#                         os.makedirs(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4)

#                     for upper_zone_ntg in upper_zone_ntgs:
#                         dir_path_5 = "/up_NTG_" + str(upper_zone_ntg).replace(".","_")
#                         if not os.path.exists(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5):
#                             os.makedirs(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5)

#                         for lower_zone_thickness in lower_zone_thicknesses:
#                             dir_path_6 = "/low_thickness_" + str(lower_zone_thickness).replace(".","_")
#                             if not os.path.exists(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5+dir_path_6):
#                                 os.makedirs(dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5+dir_path_6)

#                             for upper_zone_thickness in upper_zone_thicknesses:
#                                 dir_path_7 = "/up_thickness_" + str(upper_zone_thickness).replace(".","_")
#                                 final_path = dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5+dir_path_6+dir_path_7
#                                 if not os.path.exists(final_path):
#                                     os.makedirs(final_path)
                                
#                                 # Generate static model files
#                                 gen_static_model(upper_zone_thickness=upper_zone_thickness, lower_zone_thickness=lower_zone_thickness, ntg2=upper_zone_ntg,
#                                                     ntg1=lower_zone_ntg, perm1=np.log10(lower_zone_permeability))
                                
#                                 # Edit DATA file
#                                 with open("./eclipse_files/Datafile.DATA", "r") as file:
#                                     lines = file.readlines()
#                                 for i, line in enumerate(lines):
#                                     if "EQUIL" in line:
#                                         lines[i+3] = '  '.join(["{:.2f}".format(P_init) if i == 1 else x for i, x in enumerate(lines[i+3].split())]) + "\n"
#                                         lines[i+5] = '  '.join(["{:.2f}".format(P_init-P_diff) if i == 1 else x for i, x in enumerate(lines[i+5].split())]) + "\n"
#                                     if "ROCK" in line:
#                                         lines[i+1] = ' '.join([str(rock_compressibility)+".00E-6" if i == 1 else x for i, x in enumerate(lines[i+1].split())]) + "\n"
#                                 with open("./eclipse_files/Datafile.DATA", "w") as file:
#                                     file.writelines(lines)
                                    
#                                 # Copy Eclipse files
#                                 shutil.copy("./eclipse_files/COORD", final_path+"/COORD")
#                                 shutil.copy("./eclipse_files/ZCORN", final_path+"/ZCORN")
#                                 shutil.copy("./eclipse_files/ACTNUM", final_path+"/ACTNUM")
#                                 shutil.copy("./eclipse_files/PORO", final_path+"/PORO")
#                                 shutil.copy("./eclipse_files/PERMX", final_path+"/PERMX")
#                                 shutil.copy("./eclipse_files/PERMY", final_path+"/PERMY")
#                                 shutil.copy("./eclipse_files/PERMZ", final_path+"/PERMZ")
#                                 shutil.copy("./eclipse_files/Prod_sch", final_path+"/Prod_sch")
#                                 shutil.copy("./eclipse_files/Grid.GRDECL", final_path+"/Grid.GRDECL")
#                                 shutil.copy("./eclipse_files/Kr.GRDECL", final_path+"/Kr.GRDECL")
#                                 shutil.copy("./eclipse_files/PVT.GRDECL", final_path+"/PVT.GRDECL")
#                                 shutil.copy("./eclipse_files/VFP3000.vfp", final_path+"/VFP3000.vfp")
#                                 shutil.copy("./eclipse_files/Datafile.DATA", final_path+"/Datafile.DATA")

#                                 # # Be careful with \ that stupid Windows wants
#                                 # print("Starting Eclipse run")
#                                 # subprocess.Popen("cmd /c \"eclrun eclipse " + final_path + "/Datafile.DATA\"", shell=True)
#                                 # print("Eclipse run submitted")


max_runs_in_parallel = 14
runs_in_parallel = 0
submitted_num = 0

# Create Entire Directory Tree
# Fill Subdirectories with Eclipse Files and Start Eclipse Runs
with concurrent.futures.ThreadPoolExecutor() as executor:
    final_paths = []
    futures = []
    for P_diff in P_diffs:
        dir_path_0 = "../P_diff_" + str(P_diff).replace(".","_")
        for lower_zone_ntg in lower_zone_ntgs:
            dir_path_1 = "/low_NTG_" + str(lower_zone_ntg).replace(".","_")
            for lower_zone_permeability in lower_zone_permeabilities:
                dir_path_2 = "/low_perm_" + str(lower_zone_permeability).replace(".","_")
                for P_init in P_inits:
                    if P_diff==500 and lower_zone_ntg==0.7 and lower_zone_permeability==10 and P_init==18000:
                        continue
                    dir_path_3 = "/P_init_" + str(P_init).replace(".","_")
                    for rock_compressibility in rock_compressibilities:
                        dir_path_4 = "/rock_compress_" + str(rock_compressibility).replace(".","_")
                        for upper_zone_ntg in upper_zone_ntgs:
                            dir_path_5 = "/up_NTG_" + str(upper_zone_ntg).replace(".","_")
                            for lower_zone_thickness in lower_zone_thicknesses:
                                dir_path_6 = "/low_thickness_" + str(lower_zone_thickness).replace(".","_")
                                for upper_zone_thickness in upper_zone_thicknesses:
                                    dir_path_7 = "/up_thickness_" + str(upper_zone_thickness).replace(".","_")
                                    final_path = dir_path_0+dir_path_1+dir_path_2+dir_path_3+dir_path_4+dir_path_5+dir_path_6+dir_path_7

                                    if submitted_num < max_runs_in_parallel:
                                        submitted_num += 1
                                        futures.append(executor.submit(run_eclipse,final_path+"/Datafile.DATA"))
                                        final_paths.append(final_path)
                                        print("Submitted Eclipse run " + str(submitted_num) + " in " + final_path)
                                    else:
                                        concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                        future = executor.submit(run_eclipse,final_path+"/Datafile.DATA")
                                        for i, f in enumerate(futures):
                                            if f.done():
                                                futures[i] = future
                                                print("Finished Eclipse run in " + final_paths[i])
                                                final_paths[i] = final_path
                                                break
                                        submitted_num += 1
                                        print("Submitted Eclipse run " + str(submitted_num) + " in " + final_path)
                                    

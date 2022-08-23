
import py21cmfast.dm_dtb_tools as db_tools
import example_lightcone_analysis as lightcone_analysis

# Define the database location
database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST/"

db_manager_approx = db_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
db_manager_DH     = db_tools.DHDatabase(path = database_location, cache_path = cache_location)

index_0 = db_manager_approx.search(lifetime=1e+26, approx_params=[[0.3], [0.4, 0.2]], mDM=1e+8)
index_1 = db_manager_DH.search(lifetime=0, primary='none', mDM=0)

# We can charge and remake the analysis for the lightcones in index_0 with
for ind in index_0:
    path = db_manager_approx.path_brightness_temp
    lightcone = lightcone_analysis.LightCone.read(fname= path + str(ind) + '/Lightcone/Lightcone.h5')
    lightcone_analysis.make_analysis(path + str(ind), lightcone, n_psbins=20, nchunks=40)

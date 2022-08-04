
import py21cmfast.dm_dtb_tools as db_tools

# Define the database location
database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST/"

db_manager_approx = db_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
db_manager_DH     = db_tools.DHDatabase(path = database_location, cache_path = cache_location)

print(db_manager_approx.search(lifetime=1e+26, approx_params=[[0.3], [0.4, 0.2]], mDM=1e+8))
print(db_manager_DH.search(lifetime=0, primary='none', mDM=0))